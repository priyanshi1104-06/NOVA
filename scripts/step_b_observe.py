"""
NOVA - STEP B: prove the CARLA -> Observation adapter is correct.

WHAT THIS PROVES
----------------
That `CarlaBridge` reports the world the way NOVA expects it: right things,
right classes, right SIGNS. The car is still on stock autopilot; we are only
watching what NOVA would see.

HOW TO VERIFY (this is the whole point - do not skip it)
--------------------------------------------------------
The printout labels each nearby object AHEAD/BEHIND and LEFT/RIGHT. Watch the
CARLA window and compare. If a bus is visibly on the ego's left but the
terminal says RIGHT, the handedness conversion in carla_bridge.py is wrong and
NOTHING after this point can work - the planner would steer into hazards while
believing it was avoiding them.

Sixty seconds of eyeballing here saves a whole evening in Step C.

RUN
---
Terminal 1:  CarlaUE4.exe (already running)
Terminal 2:  cd C:\\NOVA
             .venv\\Scripts\\activate
             python scripts\\step_b_observe.py
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step_a_traffic import Scenario                    # reuse the spawn logic
from nova.carla_bridge import CarlaBridge
from nova.types import AgentClass


def describe(tr):
    """Human-readable bearing, so you can check it against the CARLA window."""
    lon = "AHEAD " if tr.x >= 0 else "BEHIND"
    lat = "LEFT " if tr.y > 0.5 else ("RIGHT" if tr.y < -0.5 else "CENTRE")
    return f"{lon} {lat}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm-port", type=int, default=8000)
    ap.add_argument("--town", default="Town01")
    ap.add_argument("--n-vehicles", type=int, default=60)
    ap.add_argument("--n-walkers", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    scn = Scenario(args)
    try:
        scn.setup_sync()
        scn.spawn_ego()
        scn.spawn_traffic()
        scn.spawn_walkers()

        # observe() is called every tick here, not once per planning cycle,
        # so the history interval is one simulator step.
        bridge = CarlaBridge(scn.world, scn.ego, sample_dt=0.05)
        print("\nwatching what NOVA would see - Ctrl+C to stop")
        print("compare AHEAD/BEHIND and LEFT/RIGHT against the CARLA window\n")

        frame = 0
        t0 = time.perf_counter()
        while True:
            scn.world.tick()
            scn.follow_with_spectator()
            frame += 1

            obs = bridge.observe(t=frame * 0.05)

            # Print a snapshot every 2 seconds of sim time.
            if frame % 40 == 0:
                real = time.perf_counter() - t0
                sim = frame * 0.05
                near = sorted(obs.tracks, key=lambda tr: tr.x**2 + tr.y**2)[:6]

                print(f"--- sim {sim:6.1f}s | real {real:6.1f}s | "
                      f"ratio {sim/max(real,1e-6):4.2f}x | "
                      f"ego {obs.ego.v*3.6:5.1f} km/h | "
                      f"{len(obs.tracks)} tracks in range")
                for tr in near:
                    print(f"      {tr.cls.value:<12} id={tr.id:<5} "
                          f"x={tr.x:+6.1f} y={tr.y:+6.1f}  "
                          f"{describe(tr)}  "
                          f"speed={tr.speed*3.6:5.1f} km/h  "
                          f"yaw_rate={tr.yaw_rate():+5.2f}")

                # Class census - confirms the fleet mix actually spawned and
                # that classify() is not lumping everything into CAR.
                if frame % 200 == 0:
                    census = {}
                    for tr in obs.tracks:
                        census[tr.cls.value] = census.get(tr.cls.value, 0) + 1
                    print(f"      classes: {census}")
                print()

    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        scn.cleanup()


if __name__ == "__main__":
    main()
