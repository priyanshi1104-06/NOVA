"""
NOVA 2.0 - the same autonomy stack, on REAL Indian road video. No simulator.

    mp4 frame
        |
        v
  [1] YOLOv8 (best.pt, IDD-tuned)  ->  boxes + classes + track IDs
        |
        v  flat-ground range from the box's contact point
      List[Track]  <- the SAME type carla_bridge emits
        |
        v
  [2] multimodal intent prediction
  [3] spatio-temporal risk map
  [4] time-aware Hybrid A*
  [6] the same HUD

WHY THIS EXISTS
---------------
Two reasons, and the second is the interesting one.

1. PRACTICAL. On this machine CARLA will not render camera sensors, so the
   perception half of NOVA cannot be shown in the simulator at all. Video
   needs no simulator, so it cannot be broken by one.

2. IT IS A BETTER ANSWER TO THE PROBLEM STATEMENT. CARLA's towns look
   European. This runs the identical prediction, risk and planning code over
   genuine Indian traffic - autos, filtering two-wheelers, unmarked lanes,
   pedestrians crossing anywhere - which is what the brief is actually about.

NOTHING IN nova/ WAS CHANGED TO MAKE THIS WORK. That is the architecture's
whole claim: every stage knows only the SHAPE of its input. perception.py
already emits List[Track], exactly as carla_bridge.py does, so the pipeline
downstream cannot tell which produced it.

WHAT THIS IS NOT
----------------
OPEN LOOP. The footage is fixed, so the car cannot act on the plan - NOVA
draws what it WOULD do. Closed-loop driving is the CARLA demo. Say so
plainly when showing it; a judge who assumes otherwise and then works it out
will discount everything else you said.

DEPTH IS ESTIMATED, NOT MEASURED. CARLA has a depth camera; video does not.
Range comes from where a box meets the ground plus the camera geometry - the
standard monocular approach. It is good enough to rank hazards and poor at
absolute distance, and it needs a rough idea of the source camera's field of
view (--fov). Dashcams are typically 60-90 degrees.

RUN
---
    python scripts\\nova_video.py --video road.mp4
    python scripts\\nova_video.py --video road.mp4 --weights best.pt --fov 70
    python scripts\\nova_video.py --video road.mp4 --record out.mp4
"""

import argparse
import dataclasses
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nova.hud import HUD, LaneAnalyzer
from nova.perception import CameraIntrinsics, VisionPerception
from nova.pipeline import NovaPipeline
from nova.types import EgoState, Observation


def flat_ground_depth(K: CameraIntrinsics, cam_height: float) -> np.ndarray:
    """A depth image assuming the whole scene is flat ground.

    perceive() reads depth at the BOTTOM-CENTRE of each box - the point where
    the object touches the road. For that pixel the flat-ground assumption is
    not an approximation, it is exactly right, which is why this cheap trick
    is the standard monocular range estimate.

    For a pixel row v below the horizon:

        depth = fy * camera_height / (v - cy)

    Rows at or above the horizon get +inf: nothing there is on the ground, so
    perceive() discards them via max_range rather than inventing a distance.
    """
    v = np.arange(K.height, dtype=np.float32).reshape(-1, 1)
    below = v - K.cy
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(below > 1.0, K.fy * cam_height / below, np.inf)
    return np.repeat(d.astype(np.float32), K.width, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="path to an mp4")
    ap.add_argument("--weights", default="yolov8n.pt",
                    help="best.pt once the IDD fine-tune is done")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fov", type=float, default=75.0,
                    help="horizontal field of view of the SOURCE camera, "
                         "degrees. Dashcams are usually 60-90. This scales "
                         "every distance, so a wrong value makes the risk "
                         "map uniformly too near or too far.")
    ap.add_argument("--cam-height", type=float, default=1.35,
                    help="height of the source camera above the road, metres")
    ap.add_argument("--horizon", type=float, default=None,
                    help="where the horizon sits, as a fraction of frame "
                         "height (0.5 = the middle). Range is fy*h/(v-cy), "
                         "so cy IS the horizon; a CARLA camera puts it at "
                         "0.5, but a dashcam aimed slightly up puts it lower "
                         "in frame. Find it by looking at one frame: the row "
                         "where the road disappears. Getting it wrong scales "
                         "every distance at once.")
    ap.add_argument("--ego-speed", type=float, default=8.0,
                    help="starting speed of the filming vehicle, m/s. Video "
                         "carries no odometry, and the predictor needs a "
                         "closing speed to reason about time-to-collision. "
                         "Most dashcams burn the real speed into the corner "
                         "of the frame - use that number.")
    ap.add_argument("--fixed-speed", action="store_true",
                    help="hold ego speed constant instead of letting it "
                         "follow the planner's own accel command. Constant "
                         "speed means NOVA closes on traffic that never "
                         "recedes, so it sits in EMERGENCY for most of a "
                         "clip - which reads as broken and misrepresents "
                         "what the planner decided.")
    ap.add_argument("--v-target", type=float, default=8.0)
    ap.add_argument("--every", type=int, default=1,
                    help="process every Nth frame; 2 halves the load")
    ap.add_argument("--record", default=None, help="write the dashboard to mp4")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole file")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="enlarge the dashboard window by this factor. The "
                         "HUD renders at a fixed size, so 1.0 is a small "
                         "window on a 1080p screen; 1.8 roughly fills one.")
    ap.add_argument("--fullscreen", action="store_true",
                    help="fill the whole screen, ignoring --scale. Press Q "
                         "to quit - there is no window chrome to close.")
    ap.add_argument("--fast", action="store_true",
                    help="process flat out instead of pacing to the source "
                         "frame rate. Playback speed then depends on how "
                         "expensive the scene is - dense traffic crawls, an "
                         "empty road races. Useful with --record, wrong for "
                         "anything a person is watching.")
    ap.add_argument("--loop", action="store_true",
                    help="restart at the end instead of exiting. A short clip "
                         "otherwise runs out mid-sentence while you are "
                         "presenting.")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        raise SystemExit(f"video not found: {args.video}")

    try:
        import ultralytics                                 # noqa: F401
    except ImportError:
        raise SystemExit(
            "\nultralytics is not installed. Run:\n"
            "    pip install ultralytics\n")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.video}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"video: {src_w}x{src_h} @ {src_fps:.0f} fps")

    # Work at 640 wide: YOLO resizes to 640 anyway, and the HUD is laid out
    # for roughly this size. Keep the aspect ratio so the geometry stays sane.
    W = 640
    H = max(2, int(round(src_h * W / max(src_w, 1))))
    K = CameraIntrinsics.from_fov(W, H, args.fov)
    if args.horizon is not None:
        if not 0.05 < args.horizon < 0.98:
            raise SystemExit("--horizon is a fraction of frame height, "
                             "roughly 0.4 to 0.8 for a real dashcam")
        K = dataclasses.replace(K, cy=args.horizon * H)
        print(f"horizon at row {K.cy:.0f} of {H}")
    depth = flat_ground_depth(K, args.cam_height)

    percep = VisionPerception(K, weights=args.weights, conf=args.conf,
                              device=args.device,
                              cam_offset=(0.0, 0.0, args.cam_height))
    pipe = NovaPipeline(v_target=args.v_target)
    lanes = LaneAnalyzer(width=W, height=H)
    hud = HUD(cam_w=W, cam_h=H)
    print(f"perception: {args.weights} on {args.device}   fov {args.fov} deg")

    writer = None
    v_ego = args.ego_speed          # evolves unless --fixed-speed
    frame_i = 0
    processed = 0
    fps, last_t = 0.0, time.perf_counter()
    dt = args.every / float(src_fps)

    # NO SEGMENTATION ON RAW VIDEO, AND WE DO NOT PRETEND OTHERWISE.
    #
    # The three corridor panels (DRIVABLE AREA / BIRD'S EYE / CORRIDOR FIT)
    # are all derived from a segmentation mask. YOLO gives boxes, not a
    # per-pixel road mask, so on video there is nothing real to put in them
    # until DeepLabV3+ is wired in. Feeding them an all-ones stub produces
    # two white rectangles, a meaningless green blob, and a curvature number
    # computed from noise - which is worse than showing nothing, because it
    # looks like a result.
    #
    # So they are labelled offline, and obs.drivable stays None: RiskMap then
    # uses its geometric road model instead of a fabricated mask.
    drivable_stub = np.ones((H, W), dtype=np.uint8)
    SEGMENTATION_AVAILABLE = False

    WIN = "NOVA 2.0 - real road video"
    if args.fullscreen:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
    elif args.scale != 1.0:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    print("\nNOVA 2.0 running on video. Q to stop.\n")
    try:
        while True:
            tick = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                if args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    if not ok:
                        print("could not rewind - stopping")
                        break
                else:
                    print("end of video")
                    break
            frame_i += 1
            if frame_i % args.every:
                continue
            if args.max_frames and processed >= args.max_frames and not args.loop:
                break

            bgr = cv2.resize(frame, (W, H))

            t_p = time.perf_counter()
            tracks = percep.perceive(bgr, depth, dt=dt)
            perc_ms = (time.perf_counter() - t_p) * 1000.0

            obs = Observation(
                t=processed * dt,
                ego=EgoState(x=0.0, y=0.0, heading=0.0,
                             v=v_ego, steer=0.0),
                tracks=tracks,
                goal=np.array([40.0, 0.0], dtype=np.float64),
            )
            res = pipe.step(obs)

            # Close the loop on speed, then pull back towards the seed.
            #
            # THE PULL IS NOT COSMETIC. Integrating accel alone ratchets to
            # zero and stays there: the footage keeps rushing at NOVA at the
            # filming vehicle's real speed no matter what NOVA decides, so
            # hazards keep arriving, every emergency plan commands -5 m/s^2,
            # and clip() floors it. Measured on the night clip: 43 km/h to 0
            # in 16 frames, then dead for the rest of the run.
            #
            # The filming vehicle DID keep moving at roughly --ego-speed, so
            # relaxing towards it is the physically honest thing to do. With
            # tau = 1.5 s the same clip reads 43 -> 24 km/h as hazards appear
            # and recovers to 35: NOVA visibly slowing for things the human
            # drove past, which is the point of the demo.
            if not args.fixed_speed and res.plan is not None:
                v_ego = float(np.clip(v_ego + res.plan.accel * dt,
                                      0.0, args.ego_speed * 1.5))
                v_ego += (args.ego_speed - v_ego) * min(1.0, dt / 1.5)

            lane = lanes.analyse(drivable_stub)
            now = time.perf_counter()
            inst = 1.0 / max(now - last_t, 1e-6)
            fps = 0.9 * fps + 0.1 * inst if fps else inst
            last_t = now

            gaps = [float(np.hypot(t.x, t.y)) for t in tracks]
            stats = {
                "speed_label": "SPEED" if args.fixed_speed else "NOVA WANTS",
                "kph": v_ego * 3.6,
                "plan_ms": res.t_total_ms,
                "perc_ms": perc_ms,
                "n_tracks": len(tracks),
                "min_gap": min(gaps) if gaps else 999.0,
                "collisions": 0,
                "fps": fps,
            }
            canvas = hud.render(bgr, lane, percep.last_boxes, tracks,
                                pipe.risk, res.plan, res.predictions, stats,
                                cameras_ok=SEGMENTATION_AVAILABLE)

            if args.record:
                if writer is None:
                    writer = cv2.VideoWriter(
                        args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                        max(src_fps / args.every, 1.0), (hud.W, hud.H))
                    print(f"recording -> {args.record}")
                writer.write(canvas)

            shown = canvas
            if args.scale != 1.0 and not args.fullscreen:
                shown = cv2.resize(
                    canvas, None, fx=args.scale, fy=args.scale,
                    interpolation=cv2.INTER_CUBIC)
            cv2.imshow(WIN, shown)

            # waitKey is the sleep: it both paces the loop and pumps the
            # window's event queue, which imshow needs to redraw at all.
            wait_ms = 1
            if not args.fast:
                spare = dt - (time.perf_counter() - tick)
                wait_ms = max(1, int(spare * 1000.0))
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                break
            processed += 1
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"saved {args.record}")
        cv2.destroyAllWindows()
        print(f"processed {processed} frames")


if __name__ == "__main__":
    main()
