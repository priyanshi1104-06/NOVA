"""
NOVA - CARLA adapter. The ONLY file that knows CARLA exists.

Its whole job: turn CARLA's world state into the `Observation` that
pipeline.py already consumes. Nothing downstream imports carla.

THE COORDINATE TRAP - read this before debugging anything
----------------------------------------------------------
CARLA uses a LEFT-HANDED frame:   +x forward, +y RIGHT, +z up, yaw in DEGREES
NOVA uses a RIGHT-HANDED frame:   +x forward, +y LEFT,  +z up, yaw in RADIANS

If you skip the conversion, every lateral sign flips. The car then sees a
motorcycle cutting in from the left, decides to move right - and moves left,
straight into it. It looks exactly like a planner bug, and you can lose hours
in planner.py before suspecting the adapter. Convert here, once, and never
think about it again.

The conversion is two steps:
  1. Mirror the world:  Y = -y_carla,  yaw = -yaw_carla  (handedness fix)
  2. Rotate + translate into the ego's frame                 (ego-relative fix)

WHY GROUND TRUTH FIRST
----------------------
This adapter reads CARLA's actor list directly - perfect positions, perfect
classes. That is deliberate for now. It lets us prove prediction, risk and
planning work in closed loop BEFORE adding perception error on top. In Step E
we swap the actor list for YOLOv8 + depth, and because the output type is
identical, nothing downstream changes. It also gives you an upper bound: if
the car cannot drive with perfect perception, the problem is never perception.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Dict, List, Optional

import numpy as np

from .types import AgentClass, EgoState, Observation, Track

# CARLA blueprint ids that are NOT ordinary cars. Everything else with four
# wheels falls through to CAR. Two-wheelers are split by blueprint because
# CARLA reports both bicycles and motorcycles as number_of_wheels == 2, and
# the difference matters a lot to the predictor: a bicycle tops out at 7 m/s
# and a Ninja at 20.
BICYCLE_IDS = {
    "vehicle.gazelle.omafiets",
    "vehicle.diamondback.century",
    "vehicle.bh.crossbike",
}
BUS_IDS = {
    "vehicle.mitsubishi.fusorosa",
}
TRUCK_IDS = {
    "vehicle.carlamotors.carlacola",
    "vehicle.carlamotors.firetruck",
    "vehicle.tesla.cybertruck",
    "vehicle.ford.ambulance",
    "vehicle.mercedes.sprinter",
    "vehicle.volkswagen.t2",
    "vehicle.volkswagen.t2_2021",
}


def classify(actor) -> AgentClass:
    """CARLA actor -> NOVA AgentClass.

    This is the ONE function to edit if you later add custom vehicles, or when
    YOLO replaces ground truth and starts emitting COCO class names instead.
    """
    tid = actor.type_id

    if tid.startswith("walker.pedestrian"):
        return AgentClass.PEDESTRIAN

    if not tid.startswith("vehicle."):
        return AgentClass.UNKNOWN

    if tid in BICYCLE_IDS:
        return AgentClass.BICYCLE
    if tid in BUS_IDS:
        return AgentClass.BUS
    if tid in TRUCK_IDS:
        return AgentClass.TRUCK

    wheels = int(actor.attributes.get("number_of_wheels", 4))
    if wheels == 2:
        return AgentClass.TWO_WHEELER

    return AgentClass.CAR


class CarlaBridge:
    """Builds an `Observation` from the live CARLA world, every frame.

    Parameters
    ----------
    radius : metres. Actors beyond this are ignored. 60 m comfortably exceeds
        the risk map's 50 m forward range, so nothing relevant is dropped, and
        it keeps per-frame cost proportional to what is actually nearby rather
        than to the total number of actors in the town.
    history_len : how many past positions to keep per track. The predictor's
        yaw_rate() needs at least 3; 5 gives it a little noise immunity.
    sample_dt : seconds between successive observe() calls. This is the
        CALLER's cadence, not the simulator's: step_c and nova_drive observe
        once per planning cycle (plan_every * 0.05 s), step_b observes every
        tick (0.05 s). It is stamped onto every Track so yaw_rate() divides by
        the right number.
    """

    def __init__(self, world, ego, radius: float = 60.0, history_len: int = 5,
                 sample_dt: float = 0.1):
        self.world = world
        self.ego = ego
        self.radius = radius
        self.sample_dt = sample_dt
        self.history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=history_len))
        self._seen_last_frame: set = set()

    # ------------------------------------------------------------------
    # the transform - the important part of this file
    # ------------------------------------------------------------------
    @staticmethod
    def _ego_basis(ego_transform):
        """Return (EX, EY, yaw) describing the ego in a RIGHT-handed world.

        CARLA's y axis points right and its yaw is measured accordingly, so
        mirroring y also requires negating yaw to keep rotations consistent.
        Do one without the other and everything spins the wrong way.
        """
        loc = ego_transform.location
        EX = loc.x
        EY = -loc.y                                  # handedness fix
        yaw = -math.radians(ego_transform.rotation.yaw)
        return EX, EY, yaw

    @staticmethod
    def _to_ego(EX, EY, yaw, wx, wy):
        """World point (CARLA coords) -> ego-frame metres, +y LEFT."""
        X, Y = wx, -wy                               # handedness fix
        dx, dy = X - EX, Y - EY
        c, s = math.cos(yaw), math.sin(yaw)
        x_ego = dx * c + dy * s
        y_ego = -dx * s + dy * c
        return x_ego, y_ego

    @staticmethod
    def _vec_to_ego(yaw, vx, vy):
        """World VELOCITY (CARLA coords) -> ego frame. Rotation only, no
        translation - a velocity has direction and magnitude but no position."""
        VX, VY = vx, -vy
        c, s = math.cos(yaw), math.sin(yaw)
        return VX * c + VY * s, -VX * s + VY * c

    # ------------------------------------------------------------------
    # drivable area WITHOUT a camera
    # ------------------------------------------------------------------
    # WHY THIS EXISTS. The drivable mask normally comes from the semantic
    # camera. With --no-cameras there is none, and seeding it to "everything
    # is road" removes riskmap.offroad_cost entirely - the planner then has
    # no idea where the road is, pins the steering to one side and drives
    # into a wall. Measured: ego reached 19.4 km/h, then wedged, 147
    # collisions, steer stuck at -0.50 for the whole run.
    #
    # In ground-truth mode the tracks already come from simulator state, so
    # taking the road geometry from the same source is consistent, not a
    # short cut. On real hardware this is what an HD map layer provides.
    #
    # The raster is built ONCE (a few seconds) and sampled per frame with
    # numpy, because get_waypoint() per grid cell per frame would be ~38 000
    # calls a frame.
    def _build_road_raster(self, res: float = 1.0, pad: float = 20.0):
        wps = self.world.get_map().generate_waypoints(1.5)
        if not wps:
            return None
        pts = np.array([[w.transform.location.x, -w.transform.location.y,
                         max(w.lane_width, 2.0) * 0.5] for w in wps],
                       dtype=np.float64)          # y mirrored to NOVA frame

        self._r_res = res
        self._r_x0 = pts[:, 0].min() - pad
        self._r_y0 = pts[:, 1].min() - pad
        nx = int((pts[:, 0].max() + pad - self._r_x0) / res) + 1
        ny = int((pts[:, 1].max() + pad - self._r_y0) / res) + 1
        raster = np.zeros((ny, nx), dtype=np.uint8)

        # Stamp a disc of the lane's half-width at every waypoint.
        for px, py, half in pts:
            r = int(half / res) + 1
            cx = int((px - self._r_x0) / res)
            cy = int((py - self._r_y0) / res)
            x0, x1 = max(0, cx - r), min(nx, cx + r + 1)
            y0, y1 = max(0, cy - r), min(ny, cy + r + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            yy, xx = np.ogrid[y0:y1, x0:x1]
            raster[y0:y1, x0:x1] |= (((xx - cx) ** 2 + (yy - cy) ** 2)
                                     <= r * r).astype(np.uint8)
        return raster

    # Static things the car can hit that are NOT actors and therefore never
    # appear in observe()'s track list. Every collision in a 3-minute run was
    # one of these: fences, guardrails, poles, sign posts. The planner could
    # not see any of them - the drivable mask said "not road" but nothing said
    # "solid steel object here", and off-road cost alone is not enough when
    # the car is already off the road.
    STATIC_LABELS = ("Fences", "GuardRail", "Poles", "TrafficSigns", "Walls")

    def static_obstacles(self, radius: float = 60.0):
        """(x, y, r) of nearby static obstacles, in the ego frame.

        Fetched ONCE and cached - these never move, so re-querying every frame
        would be pure RPC waste. The world-space list is cached and only the
        ego-frame transform is redone per call.
        """
        import carla

        if getattr(self, "_static_world", None) is None:
            boxes = []
            for name in self.STATIC_LABELS:
                label = getattr(carla.CityObjectLabel, name, None)
                if label is None:
                    continue
                try:
                    for bb in self.world.get_level_bbs(label):
                        e = bb.extent
                        boxes.append((bb.location.x, bb.location.y,
                                      max(float(e.x), float(e.y))))
                except Exception:                      # noqa: BLE001
                    continue
            self._static_world = boxes

        ego_tf = self.ego.get_transform()
        EX, EY, yaw = self._ego_basis(ego_tf)
        out = []
        for wx, wy, r in self._static_world:
            x, y = self._to_ego(EX, EY, yaw, wx, wy)
            if -10.0 < x < radius and abs(y) < radius:
                out.append((x, y, r))
        return out

    def drivable_from_map(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """(ny, nx) mask over the risk grid, 1 = road. No camera needed."""
        if getattr(self, "_road_raster", None) is None:
            self._road_raster = self._build_road_raster()
        raster = self._road_raster
        if raster is None:                        # no map data - assume open
            return np.ones((len(ys), len(xs)), dtype=np.uint8)

        ego_tf = self.ego.get_transform()
        EX, EY, yaw = self._ego_basis(ego_tf)
        gx, gy = np.meshgrid(xs, ys)              # ego frame, +y LEFT

        # Ego frame -> world, the exact inverse of _to_ego().
        c, s = math.cos(yaw), math.sin(yaw)
        X = EX + gx * c - gy * s
        Y = EY + gx * s + gy * c

        ix = ((X - self._r_x0) / self._r_res).astype(np.int32)
        iy = ((Y - self._r_y0) / self._r_res).astype(np.int32)
        ok = ((ix >= 0) & (ix < raster.shape[1])
              & (iy >= 0) & (iy < raster.shape[0]))
        # Off the raster means off the known map: treat as NOT road, so the
        # planner stays inside the town instead of driving out of it.
        out = np.zeros(gx.shape, dtype=np.uint8)
        out[ok] = raster[iy[ok], ix[ok]]
        return out

    # ------------------------------------------------------------------
    def observe(self, t: float, goal: Optional[np.ndarray] = None) -> Observation:
        # ONE SNAPSHOT, NOT THREE RPCs PER ACTOR.
        #
        # actor.get_transform() and actor.get_velocity() are each a blocking
        # RPC round trip to the server. The old loop called get_transform()
        # TWICE and get_velocity() once for every actor, so with 30 vehicles
        # and 12 pedestrians that was ~126 round trips per observe(), at
        # 10 Hz - about 1300 RPC calls a second, on top of three camera
        # streams.
        #
        # That is not just slow. CARLA's client aborts under heavy RPC load
        # with an uncaught msgpack exception (carla-simulator/carla#3618):
        # SIGABRT out of C++, so Python reports only
        #     Fatal Python error: Aborted
        #       File "scripts/nova_drive.py", ... in main   <- world.tick()
        # with no traceback and no exit code. That is the crash that ended
        # every run at 25-65 s.
        #
        # world.get_snapshot() returns the transform AND velocity of every
        # actor in a SINGLE call, and snapshot.find() is a local lookup. This
        # takes the per-frame RPC count from ~126 to 2.
        snapshot = self.world.get_snapshot()

        ego_snap = snapshot.find(self.ego.id)
        ego_tf = ego_snap.get_transform() if ego_snap else self.ego.get_transform()
        EX, EY, yaw = self._ego_basis(ego_tf)

        ego_vel = (ego_snap.get_velocity() if ego_snap
                   else self.ego.get_velocity())
        v_fwd, _ = self._vec_to_ego(yaw, ego_vel.x, ego_vel.y)

        ego_state = EgoState(
            x=0.0, y=0.0, heading=0.0,        # ego is the origin of its own frame
            v=max(v_fwd, 0.0),
            steer=float(self.ego.get_control().steer) * 0.55,   # normalised -> rad
        )

        tracks: List[Track] = []
        seen = set()

        for actor in self.world.get_actors():
            if actor.id == self.ego.id:
                continue
            tid = actor.type_id
            if not (tid.startswith("vehicle.") or tid.startswith("walker.pedestrian")):
                continue

            # All three reads below come out of the snapshot taken above -
            # local lookups, no RPC. actor.type_id and actor.attributes are
            # cached client-side, so they are free too.
            snap = snapshot.find(actor.id)
            if snap is None:                 # spawned after the snapshot
                continue
            actor_tf = snap.get_transform()

            loc = actor_tf.location
            x, y = self._to_ego(EX, EY, yaw, loc.x, loc.y)
            if math.hypot(x, y) > self.radius:
                continue

            vel = snap.get_velocity()
            vx, vy = self._vec_to_ego(yaw, vel.x, vel.y)

            # Heading, also mirrored and made ego-relative.
            h_world = -math.radians(actor_tf.rotation.yaw)
            heading = math.atan2(math.sin(h_world - yaw), math.cos(h_world - yaw))

            self.history[actor.id].append((x, y))
            seen.add(actor.id)

            tracks.append(Track(
                id=actor.id,
                cls=classify(actor),
                x=x, y=y, heading=heading, vx=vx, vy=vy,
                history=list(self.history[actor.id]),
                dt=self.sample_dt,
            ))

        # Drop history for actors that left our radius or were destroyed.
        # Without this the dict grows all session and, worse, a vehicle that
        # leaves and returns keeps a stale history whose gap produces a
        # nonsense yaw rate - a wild prediction fan out of nowhere.
        for gone in self._seen_last_frame - seen:
            self.history.pop(gone, None)
        self._seen_last_frame = seen

        return Observation(t=t, ego=ego_state, tracks=tracks, goal=goal)

    def world_loc_to_ego(self, location):
        """A carla.Location -> (x, y) in the ego frame. Used to convert route
        waypoints into the goal the planner searches toward."""
        ego_tf = self.ego.get_transform()
        EX, EY, yaw = self._ego_basis(ego_tf)
        return self._to_ego(EX, EY, yaw, location.x, location.y)

    # ------------------------------------------------------------------
    def ego_to_world(self, x: float, y: float):
        """Ego-frame point -> a carla.Location, for world.debug drawing.

        The exact inverse of _to_ego. Used to paint the planned path and the
        risk field onto the road in the simulator window.
        """
        import carla
        ego_tf = self.ego.get_transform()
        EX, EY, yaw = self._ego_basis(ego_tf)
        c, s = math.cos(yaw), math.sin(yaw)
        X = EX + x * c - y * s
        Y = EY + x * s + y * c
        return carla.Location(x=X, y=-Y, z=ego_tf.location.z + 0.3)


# ======================================================================
# Camera rig - RGB + depth, frame-synchronised with the simulation tick
# ======================================================================
class SensorRig:
    """Attaches an RGB and a depth camera to the ego and hands back matched
    frames, one pair per world.tick().

    WHY QUEUES AND NOT CALLBACKS
    ----------------------------
    CARLA sensors deliver asynchronously through listen() callbacks. If you
    just keep the newest frame in a variable, you end up running detection on
    an image from a different instant than the tick you are planning for -
    at 8 m/s that is a metre or two of error that looks exactly like a
    calibration bug. Pushing frames into a Queue and pulling exactly one per
    tick guarantees image, depth and simulation state all describe the same
    moment.

    RESOLUTION IS A VRAM DECISION
    -----------------------------
    640x480 on two cameras is deliberate. Every extra pixel costs GPU memory
    that CARLA and YOLO are already competing for on a 6 GB card, and YOLOv8
    resizes to 640 internally anyway - so a larger camera buys you nothing but
    a slower copy.
    """

    def __init__(self, world, ego, width=640, height=480, fov=90.0,
                 x=1.6, z=1.6, queue_size=8, sensor_tick=0.0):
        import queue
        import carla

        self.world = world
        self.ego = ego
        self.width, self.height, self.fov = width, height, fov
        self.actors = []
        self._rgb_q = queue.Queue(maxsize=queue_size)
        self._depth_q = queue.Queue(maxsize=queue_size)
        self._seg_q = queue.Queue(maxsize=queue_size)

        bl = world.get_blueprint_library()
        tf = carla.Transform(carla.Location(x=x, z=z))


        # sensor_tick throttles how often the SERVER renders each camera.
        #
        # This is the single biggest cost in the whole loop. Profiled on this
        # machine: prediction+risk+planner 60 ms, lane fit 5 ms, HUD 3.5 ms -
        # about 20 ms of Python per cycle - while the measured frame time was
        # 290 ms (3.4 FPS). The missing ~270 ms is inside world.tick(), where
        # the server renders three 640x480 cameras EVERY tick, at 20 Hz, when
        # the planner only consumes them at 10 Hz. Half of that work was
        # rendered and thrown away.
        #
        # It is not a cosmetic problem: a loop slower than ~120 ms per frame
        # aborts the CARLA client after ~40 s (see CLAUDE.md), so the frame
        # rate IS the stability fix.
        def apply_common(bp):
            bp.set_attribute("image_size_x", str(width))
            bp.set_attribute("image_size_y", str(height))
            bp.set_attribute("fov", str(fov))
            if sensor_tick > 0.0:
                bp.set_attribute("sensor_tick", str(sensor_tick))
            return bp

        rgb_bp = apply_common(bl.find("sensor.camera.rgb"))
        self.rgb = world.spawn_actor(rgb_bp, tf, attach_to=ego)
        self.rgb.listen(self._rgb_q.put)
        self.actors.append(self.rgb)

        # Same transform and FOV as the RGB camera, so pixel (u, v) means the
        # same ray in both. Any mismatch and you read depth for the wrong
        # object entirely.
        d_bp = apply_common(bl.find("sensor.camera.depth"))
        self.depth = world.spawn_actor(d_bp, tf, attach_to=ego)
        self.depth.listen(self._depth_q.put)
        self.actors.append(self.depth)

        # Semantic segmentation. In CARLA this is a perfect ground-truth mask,
        # which is what the drivable-area overlay and the risk map's static
        # layer consume. On real footage it is replaced by DeepLabV3+ emitting
        # the same single-channel mask - the HUD and risk map never know which
        # produced it, exactly like YOLO weights swapping from COCO to IDD.
        s_bp = apply_common(bl.find("sensor.camera.semantic_segmentation"))
        self.seg = world.spawn_actor(s_bp, tf, attach_to=ego)
        self.seg.listen(self._seg_q.put)
        self.actors.append(self.seg)

    def intrinsics(self):
        from .perception import CameraIntrinsics
        return CameraIntrinsics.from_fov(self.width, self.height, self.fov)

    # CARLA semantic tags we treat as drivable: 1 = Roads, 24 = RoadLine.
    # RoadLine is included because a painted line is still road surface - and
    # on Indian roads it is usually absent anyway, which is the whole reason
    # Module 6 fits curves to this mask instead of to painted lines.
    #
    # THESE NUMBERS ARE VERSION-SPECIFIC. CARLA RENUMBERED THE ENTIRE TABLE
    # IN 0.9.14 to follow Cityscapes ordering. Pre-0.9.14 it was Road=7,
    # RoadLine=6; from 0.9.14 on it is Road=1, RoadLine=24, and 6/7 now mean
    # POLE and TRAFFICLIGHT.
    #
    # We ran (6, 7) against 0.9.16, so the "drivable" mask contained the
    # street poles and traffic lights and nothing else - a black frame with
    # two white dots, which is exactly what the DRIVABLE MASK inset showed.
    # Two consequences, and the second is the one that wasted an evening:
    #   1. LaneAnalyzer found no road edges -> "CORRIDOR NOT FOUND" forever.
    #   2. GroundProjector marked every grid cell the camera could see as
    #      OFF-ROAD, so RiskMap's static layer put a 5.0 penalty across the
    #      whole road ahead while the unseen cells behind stayed free. The
    #      planner did the correct thing with that map and REFUSED TO DRIVE
    #      FORWARD. The car sat at 0 km/h with PLANNER: NOMINAL - no error,
    #      no crash, just a car that would not move.
    # If you ever change CARLA version, re-check this table first.
    DRIVABLE_TAGS = (1, 24)

    def grab(self, timeout: float = 2.0):
        """Pull one matched (bgr, depth_metres, drivable_mask) triple.

        Call exactly once per world.tick(). Returns (None, None, None) on
        timeout rather than raising - a dropped frame should cost one planning
        cycle, not the run.
        """
        import time as _time

        import numpy as _np
        from .perception import decode_carla_depth

        # BLOCKING get(), NOT a poll loop. This matters more than it looks.
        #
        # A previous version peeked with `while any queue empty: sleep(0.001)`
        # to make the three-way read atomic. On Windows time.sleep(0.001)
        # actually sleeps 10-15 ms (timer granularity), so every wait cost
        # ~15 ms per iteration and grab() measured 108 ms - which then got
        # blamed on GPU throughput and camera resolution. Dropping the camera
        # to 320x240, a 4x cut in pixels, moved it only 108 -> 102 ms, which
        # is what finally gave the poll loop away.
        #
        # Queue.get() waits on a condition variable and wakes the moment the
        # sensor callback delivers. Keep it that way.
        try:
            rgb_img = self._rgb_q.get(timeout=timeout)
            depth_img = self._depth_q.get(timeout=timeout)
            seg_img = self._seg_q.get(timeout=timeout)
        except Exception:                              # noqa: BLE001
            return None, None, None

        rgb = _np.frombuffer(rgb_img.raw_data, dtype=_np.uint8)
        rgb = rgb.reshape((rgb_img.height, rgb_img.width, 4))[:, :, :3].copy()

        d = _np.frombuffer(depth_img.raw_data, dtype=_np.uint8)
        d = d.reshape((depth_img.height, depth_img.width, 4))
        depth_m = decode_carla_depth(d)

        # The semantic camera stores the class tag in the RED channel of the
        # raw buffer (BGRA order, so index 2). Convert it to a plain 0/1 mask
        # so nothing downstream needs to know CARLA's tag numbering.
        sg = _np.frombuffer(seg_img.raw_data, dtype=_np.uint8)
        sg = sg.reshape((seg_img.height, seg_img.width, 4))
        tags = sg[:, :, 2]
        drivable = _np.isin(tags, self.DRIVABLE_TAGS).astype(_np.uint8)

        return rgb, depth_m, drivable

    def destroy(self):
        for a in self.actors:
            try:
                a.stop()
            except Exception:                          # noqa: BLE001
                pass
            try:
                a.destroy()
            except Exception:                          # noqa: BLE001
                pass
