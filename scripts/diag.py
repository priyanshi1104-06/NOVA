"""
NOVA - crash bisector.

step_b_observe.py runs fine. step_c_drive.py kills the CARLA server. Between
them there are only three new things:

    1. ego taken off autopilot
    2. a collision sensor attached
    3. apply_control() called every tick

This script does them ONE AT A TIME, 100 ticks each, printing before and
after every phase. Whichever phase it dies in names the culprit - which beats
restarting the simulator once per guess.

RUN
---
    (CARLA already running)
    python scripts\\reset_carla.py
    python scripts\\diag.py

Read the LAST LINE printed. That is the phase that killed it.
"""

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import carla
import numpy as np


HOST, PORT, TM_PORT = "127.0.0.1", 2000, 8000
TICKS = 100


def banner(n, text):
    print(f"\n{'='*60}\nPHASE {n}: {text}\n{'='*60}", flush=True)


def run_ticks(world, label, n=TICKS,every=25):
    for i in range(n):
        world.tick()
        if (i + 1) % every == 0:
            print(f"    {label}: {i+1}/{n} ticks OK", flush=True)


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(20.0)
    world = client.get_world()
    print(f"connected to CARLA {client.get_server_version()}", flush=True)

    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    tm = client.get_trafficmanager(TM_PORT)
    tm.set_synchronous_mode(True)
    print("synchronous mode on", flush=True)

    ego = None
    col = None
    try:
        # ---------------------------------------------------------- 1
        banner(1, "spawn ego, autopilot ON, no traffic")
        bp = world.get_blueprint_library().find("vehicle.tesla.model3")
        for sp in world.get_map().get_spawn_points():
            ego = world.try_spawn_actor(bp, sp)
            if ego is not None:
                break
        if ego is None:
            raise RuntimeError("could not spawn ego")
        ego.set_autopilot(True, TM_PORT)
        world.tick()
        print(f"  ego {ego.id} at {ego.get_transform().location}", flush=True)
        run_ticks(world, "autopilot")
        print("PHASE 1 SURVIVED", flush=True)

        # ---------------------------------------------------------- 2
        banner(2, "autopilot OFF, no control sent")
        ego.set_autopilot(False)
        run_ticks(world, "coasting")
        print("PHASE 2 SURVIVED", flush=True)

        # ---------------------------------------------------------- 3
        banner(3, "apply_control() with a constant, definitely-valid control")
        ctrl = carla.VehicleControl(throttle=0.3, steer=0.0, brake=0.0)
        for i in range(TICKS):
            ego.apply_control(ctrl)
            world.tick()
            if (i + 1) % 25 == 0:
                v = ego.get_velocity()
                print(f"    driving: {i+1}/{TICKS}  "
                      f"{3.6*math.hypot(v.x, v.y):.1f} km/h", flush=True)
        print("PHASE 3 SURVIVED", flush=True)

        # ---------------------------------------------------------- 4
        banner(4, "attach collision sensor")
        cbp = world.get_blueprint_library().find("sensor.other.collision")
        col = world.spawn_actor(cbp, carla.Transform(), attach_to=ego)
        hits = {"n": 0}
        col.listen(lambda e: hits.__setitem__("n", hits["n"] + 1))
        world.tick()
        for i in range(TICKS):
            ego.apply_control(ctrl)
            world.tick()
            if (i + 1) % 25 == 0:
                print(f"    with sensor: {i+1}/{TICKS}  "
                      f"collisions={hits['n']}", flush=True)
        print("PHASE 4 SURVIVED", flush=True)

        # ---------------------------------------------------------- 5
        banner(5, "NOVA planning + its real control values")
        from nova.carla_bridge import CarlaBridge
        from nova.pipeline import NovaPipeline

        bridge = CarlaBridge(world, ego)
        pipe = NovaPipeline(v_target=8.0)
        carla_map = world.get_map()

        for i in range(TICKS):
            wp = carla_map.get_waypoint(ego.get_transform().location)
            nxt = wp.next(25.0)
            goal_loc = nxt[0].transform.location if nxt else ego.get_transform().location
            gx, gy = bridge.world_loc_to_ego(goal_loc)

            obs = bridge.observe(t=i * 0.05,
                                 goal=np.array([gx, gy], dtype=np.float64))
            res = pipe.step(obs)

            steer = float(np.clip(res.plan.steer / 0.55, -1.0, 1.0))
            if res.plan.accel >= 0:
                throttle, brake = float(np.clip(res.plan.accel / 3.0, 0, 1)), 0.0
            else:
                throttle, brake = 0.0, float(np.clip(-res.plan.accel / 5.0, 0, 1))

            # THE CHECK THAT MATTERS. A NaN or infinity reaching CARLA's
            # physics is a documented way to take the server down with a bare
            # "Fatal error!" - exactly the symptom we are chasing.
            bad = [n for n, v in (("steer", steer), ("throttle", throttle),
                                  ("brake", brake), ("plan.steer", res.plan.steer),
                                  ("plan.accel", res.plan.accel))
                   if not math.isfinite(v)]
            if bad:
                print(f"  !!! NON-FINITE CONTROL at tick {i}: {bad}", flush=True)
                print(f"      steer={steer} throttle={throttle} brake={brake}",
                      flush=True)
                print(f"      plan.steer={res.plan.steer} "
                      f"plan.accel={res.plan.accel} "
                      f"emergency={res.plan.emergency} "
                      f"cost={res.plan.cost}", flush=True)
                break

            ego.apply_control(carla.VehicleControl(
                throttle=throttle, steer=steer, brake=brake))
            world.tick()

            if (i + 1) % 25 == 0:
                v = ego.get_velocity()
                print(f"    NOVA: {i+1}/{TICKS}  "
                      f"{3.6*math.hypot(v.x, v.y):5.1f} km/h  "
                      f"steer={steer:+.2f} thr={throttle:.2f} brk={brake:.2f}  "
                      f"plan {res.t_total_ms:.0f}ms  "
                      f"tracks={len(obs.tracks)}", flush=True)
        print("PHASE 5 SURVIVED", flush=True)

        print("\n" + "="*60)
        print("ALL PHASES SURVIVED - the crash is not in any of these")
        print("="*60, flush=True)

    finally:
        print("\ncleaning up...", flush=True)
        try:
            world.apply_settings(original)
            tm.set_synchronous_mode(False)
        except Exception:                              # noqa: BLE001
            pass
        time.sleep(0.3)
        for a in (col, ego):
            if a is not None:
                try:
                    a.destroy()
                except Exception:                      # noqa: BLE001
                    pass
        print("done", flush=True)


if __name__ == "__main__":
    main()
