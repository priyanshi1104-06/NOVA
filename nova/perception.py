"""
NOVA Module 1 - Vision Perception.

Turns camera pixels into the same `Track` objects the ground-truth adapter
produced. Everything downstream is unchanged - that is the point of having
fixed the interface first.

    RGB frame ---> YOLOv8 ---> boxes + classes + persistent IDs (ByteTrack)
                                      |
    Depth frame -------------> depth at each box's ground contact point
                                      |
                                      v
                          back-project to 3D --> Track(x, y, vx, vy, ...)

NO CARLA IMPORT IN THIS FILE. It takes numpy arrays and camera intrinsics, so
the identical code runs on a CARLA camera, a dashcam video, or a real camera
on a real car. That is what makes "our perception is sensor-agnostic" a true
statement rather than a slide.

THE THREE HARD PARTS
--------------------
1. WHERE IS IT, IN METRES? A detector gives you a box in pixels. The planner
   needs metres. We read depth at the BOTTOM-CENTRE of the box - where the
   object touches the road - and back-project through the pinhole model.
   Using the box centre instead reads depth off the object's roofline or the
   background behind it, and every distance comes out wrong.

2. IS IT THE SAME OBJECT AS LAST FRAME? Without persistent IDs there is no
   motion history, and without history the predictor cannot estimate a turn
   rate - which is the entire signal that says "this bike is cutting in".
   ByteTrack (built into Ultralytics) supplies the IDs.

3. HOW FAST IS IT MOVING? Nothing measures velocity directly. We difference
   the last two positions of each tracked ID over the known timestep, then
   smooth - raw differences of a jittering box are far too noisy to feed a
   predictor.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .types import AgentClass, Track

# Detector class name -> NOVA class.
#
# Covers BOTH label sets on purpose:
#   * COCO names, which pretrained yolov8n.pt emits - works today on CARLA
#   * IDD names, which your fine-tuned weights will emit - works on real
#     Indian roads, and adds autorickshaw / animal, the two classes no
#     Western dataset contains
# One dict, so switching weight files changes nothing but the filename.
NAME_TO_CLASS = {
    # --- COCO ---
    "car": AgentClass.CAR,
    "truck": AgentClass.TRUCK,
    "bus": AgentClass.BUS,
    "motorcycle": AgentClass.TWO_WHEELER,
    "bicycle": AgentClass.BICYCLE,
    "person": AgentClass.PEDESTRIAN,
    "dog": AgentClass.ANIMAL,
    "cow": AgentClass.ANIMAL,
    "horse": AgentClass.ANIMAL,
    # --- IDD ---
    "autorickshaw": AgentClass.AUTO_RICKSHAW,
    "auto-rickshaw": AgentClass.AUTO_RICKSHAW,
    "rider": AgentClass.TWO_WHEELER,     # IDD labels the human on a bike
    "motorcyle": AgentClass.TWO_WHEELER,  # IDD's own spelling, not a typo here
    "animal": AgentClass.ANIMAL,
    "cart": AgentClass.PUSHCART,
    "pushcart": AgentClass.PUSHCART,
    "vehicle fallback": AgentClass.UNKNOWN,
    "caravan": AgentClass.TRUCK,
    "trailer": AgentClass.TRUCK,
}


@dataclass
class CameraIntrinsics:
    """Pinhole model. CARLA cameras are ideal pinholes - no distortion - so
    focal length follows directly from the field of view."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_fov(cls, width: int, height: int, fov_deg: float) -> "CameraIntrinsics":
        f = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
        return cls(width, height, f, f, width / 2.0, height / 2.0)


def decode_carla_depth(bgra: np.ndarray) -> np.ndarray:
    """CARLA's depth image -> metres.

    CARLA packs depth into 24 bits of colour, so you cannot read it as a
    picture. The formula is fixed by CARLA:

        normalised = (R + G*256 + B*256*256) / (256**3 - 1)
        metres     = 1000 * normalised

    Forget the decode and you get a 'depth' of 0-255 that looks plausible and
    puts every vehicle a few metres away.
    """
    b = bgra[:, :, 0].astype(np.float64)
    g = bgra[:, :, 1].astype(np.float64)
    r = bgra[:, :, 2].astype(np.float64)
    normalised = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0 ** 3 - 1)
    return 1000.0 * normalised


class VisionPerception:
    """YOLOv8 + ByteTrack + depth -> List[Track] in the ego frame.

    Parameters
    ----------
    weights : 'yolov8n.pt' (COCO, works now) or your IDD-fine-tuned 'best.pt'.
    conf : detection confidence floor. 0.35 is deliberately permissive - a
        missed vulnerable road user is far more costly than a spurious box,
        because the risk map treats a low-probability hazard as a small cost
        rather than a wall.
    cam_offset : (forward, left, up) metres from the vehicle origin to the
        camera. Skip this and every measured distance carries a constant
        offset - the classic "it brakes 2 m late" bug.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        weights: str = "yolov8n.pt",
        conf: float = 0.35,
        device: str = "cuda",
        cam_offset: tuple = (1.6, 0.0, 1.6),
        history_len: int = 6,
        max_range: float = 60.0,
    ):
        from ultralytics import YOLO           # imported late: heavy

        self.K = intrinsics
        self.model = YOLO(weights)
        self.conf = conf
        self.device = device
        self.cam_offset = cam_offset
        self.max_range = max_range

        self.history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=history_len))
        self.last_seen: Dict[int, float] = {}
        self.last_boxes = []                   # kept for the HUD to draw

    # ------------------------------------------------------------------
    def _backproject(self, u: float, v: float, depth_m: float):
        """Pixel + depth -> (x, y) metres in the EGO frame.

        CARLA's camera axes are x-right, y-down, z-forward. NOVA's ego frame
        is x-forward, y-LEFT. So z_cam becomes x_ego and x_cam becomes -y_ego,
        then the camera mount offset is added.
        """
        x_cam = (u - self.K.cx) * depth_m / self.K.fx
        z_cam = depth_m
        x_ego = z_cam + self.cam_offset[0]
        y_ego = -x_cam + self.cam_offset[1]
        return x_ego, y_ego

    def _depth_at(self, depth_m: np.ndarray, u: int, v: int) -> Optional[float]:
        """Median depth in a small patch at the box's ground contact point.

        A single pixel is fragile: it can land on a gap between wheels and
        return the depth of a building 80 m away. The median of a patch is
        robust to that, and to the ragged edges YOLO boxes tend to have.
        """
        h, w = depth_m.shape
        u0, u1 = max(0, u - 3), min(w, u + 4)
        v0, v1 = max(0, v - 3), min(h, v + 4)
        if u0 >= u1 or v0 >= v1:
            return None
        patch = depth_m[v0:v1, u0:u1]
        patch = patch[(patch > 0.5) & (patch < 300.0)]   # drop sky and garbage
        if patch.size == 0:
            return None
        return float(np.median(patch))

    # ------------------------------------------------------------------
    def perceive(self, rgb: np.ndarray, depth_m: np.ndarray, dt: float) -> List[Track]:
        """One camera frame -> tracks. `rgb` is HxWx3 BGR, `depth_m` is HxW metres."""
        results = self.model.track(
            rgb,
            persist=True,                 # ByteTrack keeps IDs across frames
            tracker="bytetrack.yaml",
            conf=self.conf,
            verbose=False,
            device=self.device,
        )
        r = results[0]
        tracks: List[Track] = []
        self.last_boxes = []

        if r.boxes is None or r.boxes.id is None:
            return tracks                 # nothing detected, or no IDs yet

        names = r.names
        xyxy = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int)
        clss = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()

        seen = set()
        for (x1, y1, x2, y2), tid, ci, cf in zip(xyxy, ids, clss, confs):
            label = names[int(ci)].lower()
            cls = NAME_TO_CLASS.get(label)
            if cls is None:
                continue                  # a class NOVA does not reason about

            # Ground contact point: bottom-centre of the box.
            u = int((x1 + x2) / 2)
            v = int(y2)
            d = self._depth_at(depth_m, u, v)
            if d is None or d > self.max_range:
                continue

            x, y = self._backproject(u, v, d)

            self.history[tid].append((x, y))
            seen.add(tid)
            hist = list(self.history[tid])

            # Velocity by differencing, smoothed over up to 3 frames. Raw
            # frame-to-frame differences of a jittering box produce velocities
            # that swing wildly, and the predictor turns that into a spinning
            # prediction fan.
            if len(hist) >= 2:
                k = min(3, len(hist) - 1)
                dx = (hist[-1][0] - hist[-1 - k][0]) / (k * dt)
                dy = (hist[-1][1] - hist[-1 - k][1]) / (k * dt)
            else:
                dx = dy = 0.0

            heading = math.atan2(dy, dx) if math.hypot(dx, dy) > 0.3 else 0.0

            tracks.append(Track(
                id=int(tid), cls=cls, x=x, y=y, heading=heading,
                vx=float(dx), vy=float(dy), history=hist, confidence=float(cf),
                dt=dt,
            ))
            self.last_boxes.append(
                (float(x1), float(y1), float(x2), float(y2),
                 int(tid), cls, float(cf), float(d)))

        # Forget objects that left the frame, so a returning vehicle does not
        # inherit a stale history with a gap in it.
        for gone in set(self.history) - seen:
            self.history.pop(gone, None)

        return tracks


class GroundProjector:
    """Camera-image mask -> ego-frame occupancy grid, by flat-ground projection.

    WHY THIS EXISTS
    ---------------
    The segmentation mask is in PIXELS. The risk map is in METRES. Handing the
    raw image mask to RiskMap makes it stretch a camera image over a 60 x 40 m
    grid - the road ends up in completely the wrong place and the planner
    avoids empty tarmac while driving through kerbs.

    So for every grid cell we ask: if that patch of ground were visible, which
    pixel would it land on? Then we sample the mask there. That is a plain
    pinhole projection of a ground plane:

        z_cam = x_ego - cam_x                 (distance in front of the lens)
        x_cam = -y_ego                        (ego +y is LEFT, camera +x RIGHT)
        y_cam = cam_height                    (the ground, below the lens)
        u = fx * x_cam / z_cam + cx
        v = fy * y_cam / z_cam + cy

    ASSUMES FLAT GROUND. Fine on CARLA towns and on most urban roads; it
    degrades on steep slopes, where distant cells project slightly wrong. It
    is also why cells very close to the camera are excluded - z_cam near zero
    sends u and v to infinity.

    The projection is FIXED for a given camera and grid, so the pixel indices
    are computed once here and reused every frame - it costs one array lookup
    per frame instead of a full re-projection.
    """

    def __init__(self, intrinsics: CameraIntrinsics, xs: np.ndarray,
                 ys: np.ndarray, cam_height: float = 1.6,
                 cam_x: float = 1.6, min_range: float = 2.0):
        X, Y = np.meshgrid(xs, ys)                 # both (ny, nx)
        z = X - cam_x
        ahead = z > min_range

        safe_z = np.where(ahead, z, 1.0)
        u = intrinsics.fx * (-Y) / safe_z + intrinsics.cx
        v = intrinsics.fy * cam_height / safe_z + intrinsics.cy

        self.u = np.clip(u, 0, intrinsics.width - 1).astype(np.int32)
        self.v = np.clip(v, 0, intrinsics.height - 1).astype(np.int32)
        self.valid = (ahead
                      & (u >= 0) & (u < intrinsics.width)
                      & (v >= 0) & (v < intrinsics.height))
        self.shape = X.shape

    def project(self, mask: np.ndarray) -> np.ndarray:
        """HxW image mask -> (ny, nx) ego-grid mask, 1 = drivable.

        Cells outside the camera's view (behind, or beyond the frame edge)
        come back as 1, i.e. "assume drivable". Marking them non-drivable
        would surround the car with a wall of cost it cannot see past, and
        the planner would refuse to move.
        """
        out = np.ones(self.shape, dtype=np.uint8)
        out[self.valid] = mask[self.v[self.valid], self.u[self.valid]]
        return out
