"""
NOVA - STEP D: live vision. YOLOv8 on the CARLA camera, in an OpenCV window.

WHAT THIS PROVES
----------------
That Module 1 works: objects detected from PIXELS, classified, given stable
IDs, and placed in metres in the ego frame. The car is still on autopilot -
we are proving perception alone, before letting it steer anything.

WHAT YOU SEE
------------
An OpenCV window with the camera feed. Every detection gets a box coloured by
class, and a label reading:

    motorcycle #7  0.81  14.2m

    class          confidence  distance measured from the depth camera

Under it, an ego-frame readout: where each object is in metres, forward and
left/right. That is exactly the `Track` list handed to the predictor - so if
this window looks right, prediction gets good input.

CHECK THESE THREE THINGS
------------------------
1. Are IDs STABLE? A car keeps #7 while it is visible. If numbers churn every
   frame, ByteTrack is failing and the predictor gets no motion history - it
   is then no better than constant-velocity.
2. Are DISTANCES sane? A car two lengths ahead should read ~10-12 m. If
   everything reads 3 m or 300 m, the depth decode or the intrinsics are off.
3. Are CLASSES right? Motorcycles must not come back as bicycles - the class
   sets erraticness, and 0.65 versus 0.50 changes how wide the prediction fan
   opens.

RUN
---
    python scripts\\reset_carla.py
    python scripts\\step_d_vision.py

    --weights best.pt     use your IDD-fine-tuned model instead of COCO
    --conf 0.3            lower = more detections, more false positives
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step_a_traffic import Scenario
from nova.carla_bridge import SensorRig
from nova.perception import VisionPerception
from nova.types import AgentClass

# Per-class colours, BGR. Vulnerable road users get hot colours because they
# are the ones you must never miss.
CLASS_COLOUR = {
    AgentClass.CAR:           (200, 200, 200),
    AgentClass.TRUCK:         (180, 140,  60),
    AgentClass.BUS:           (200, 160,  40),
    AgentClass.TWO_WHEELER:   ( 60, 140, 255),
    AgentClass.BICYCLE:       ( 60, 220, 255),
    AgentClass.PEDESTRIAN:    ( 80,  80, 255),
    AgentClass.AUTO_RICKSHAW: ( 90, 200, 255),
    AgentClass.ANIMAL:        (140,  90, 255),
    AgentClass.PUSHCART:      (160, 160, 120),
    AgentClass.UNKNOWN:       (120, 120, 120),
}


def draw_detections(frame, perception, tracks, fps, n_tracks):
    """Boxes, labels and a small status bar. This is the beginning of Module 6."""
    by_id = {t.id: t for t in tracks}

    for (x1, y1, x2, y2, tid, cls, conf, dist) in perception.last_boxes:
        colour = CLASS_COLOUR.get(cls, (150, 150, 150))
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, colour, 2)

        label = f"{cls.value} #{tid}  {conf:0.2f}  {dist:0.1f}m"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(frame, (p1[0], p1[1] - th - 6),
                      (p1[0] + tw + 6, p1[1]), colour, -1)
        cv2.putText(frame, label, (p1[0] + 3, p1[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)

        # Ego-frame position under the box - the numbers the planner receives.
        tr = by_id.get(tid)
        if tr is not None:
            side = "L" if tr.y > 0 else "R"
            sub = f"x{tr.x:+.0f} y{abs(tr.y):.0f}{side} v{tr.speed*3.6:.0f}"
            cv2.putText(frame, sub, (p1[0] + 2, p2[1] + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA)

    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 26), (25, 25, 30), -1)
    cv2.putText(frame, f"NOVA  perception  |  {n_tracks} tracked  |  {fps:4.1f} FPS",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (240, 240, 240), 1, cv2.LINE_AA)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm-port", type=int, default=8000)
    ap.add_argument("--town", default="Town01")
    ap.add_argument("--n-vehicles", type=int, default=30)
    ap.add_argument("--n-walkers", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weights", default="yolov8n.pt",
                    help="yolov8n.pt (COCO) or your IDD best.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    scn = Scenario(args)
    rig = None
    try:
        scn.setup_sync()
        scn.spawn_ego()
        scn.ego.set_autopilot(True, args.tm_port)   # perception only, for now
        scn.spawn_traffic()
        scn.spawn_walkers()

        rig = SensorRig(scn.world, scn.ego)
        percep = VisionPerception(rig.intrinsics(), weights=args.weights,
                                  conf=args.conf, device=args.device)
        print(f"\nYOLO loaded: {args.weights} on {args.device}")
        print("OpenCV window open - press Q there to stop\n")

        frame_i = 0
        fps, last_t = 0.0, time.perf_counter()

        while True:
            scn.world.tick()
            frame_i += 1

            # grab() returns (rgb, depth, drivable). The segmentation mask is
            # unused here - this step is proving detection only.
            bgr, depth_m, _ = rig.grab()
            if bgr is None:
                continue

            tracks = percep.perceive(bgr, depth_m, dt=0.05)

            now = time.perf_counter()
            inst = 1.0 / max(now - last_t, 1e-6)
            fps = 0.9 * fps + 0.1 * inst if fps else inst
            last_t = now

            view = draw_detections(bgr.copy(), percep, tracks, fps, len(tracks))
            cv2.imshow("NOVA - perception", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if frame_i % 60 == 0:
                census = {}
                for t in tracks:
                    census[t.cls.value] = census.get(t.cls.value, 0) + 1
                near = sorted(tracks, key=lambda t: t.x)[:3]
                print(f"frame {frame_i:5d}  {fps:4.1f} FPS  "
                      f"{len(tracks):2d} tracks  {census}")
                for t in near:
                    print(f"    {t.cls.value:<12} #{t.id:<4} "
                          f"x={t.x:6.1f} y={t.y:+6.1f}  "
                          f"v={t.speed*3.6:5.1f} km/h")

    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        cv2.destroyAllWindows()
        if rig is not None:
            rig.destroy()
        scn.cleanup()


if __name__ == "__main__":
    main()
