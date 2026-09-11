"""
NOVA - STEP C: the car drives itself.

THIS IS THE MILESTONE. Everything before it was plumbing.

The ego comes off CARLA's autopilot. From here every steering angle and every
throttle command is produced by YOUR pipeline:

    CarlaBridge.observe()   ->  Observation      (what the car can see)
    ManeuverPredictor       ->  weighted futures (what others might do)
    RiskMap                 ->  cost field       (where is dangerous, and WHEN)
    HybridAStarPlanner      ->  steer + accel    (safest move right now)
    VehicleControl          ->  the car moves    (world responds, repeat)

Nothing is scripted. The other 30 vehicles are still on CARLA's Traffic
Manager and behave independently - the planner has no access to what they
intend and must infer it from what it can observe.

WHAT YOU SHOULD SEE
-------------------
A green line on the road ahead of the car - the planned path, redrawn ten
times a second. A scatter of coloured dots - the risk field, green where it
is safe to be and red where the predictor expects something to be. When a
motorcycle drifts toward your lane, the red should appear BEFORE the bike
arrives, and the green line should bend away from it.

That "before" is the whole thesis of the project. Point at it.

RUN
---
Terminal 1:  .\\start_carla.ps1
Terminal 2:  python scripts\\reset_carla.py        (always, before a run)
             python scripts\\step_c_drive.py

Useful flags:
    --no-debug        turn off the overlays, for honest timing numbers
    --plan-every 2    plan every Nth tick (2 = 10 Hz at a 20 Hz sim)
    --n-vehicles 30   traffic density
"""

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import carla
from step_a_traffic import Scenario
from nova.carla_bridge import CarlaBridge
from nova.pipeline import NovaPipeline

# Must match HybridAStarPlanner's max_steer, or the steering command is
# silently scaled wrong and the car understeers everywhere.
MAX_STEER_RAD = 0.55


def to_control(plan, prev_steer: float) -> "tuple[carla.VehicleControl, float]":
    """Planner output -> CARLA VehicleControl.

    Two conversions, both easy to get wrong:

    STEERING. The planner works in radians of front-wheel angle; CARLA wants
    a normalised [-1, 1]. Divide by the same max_steer the planner searched
    with. A mismatch here makes the car understeer and looks like a bad cost
    function.

    ACCELERATION. The planner emits m/s^2, CARLA wants separate throttle and
    brake pedals in [0, 1]. Positive -> throttle, negative -> brake. They are
    never both non-zero; sending both fights the physics and the car crawls.

    The steering is also rate-limited. The planner may legitimately jump from
    one committed manoeuvre to another between cycles, but a real steering
    rack cannot, and an instantaneous jump both looks wrong on video and
    upsets CARLA's tyre model.
    """
    # NOTE THE MINUS SIGN. It is not cosmetic.
    #
    # NOVA's frame has +y LEFT, and the planner integrates
    #     heading += (v / L) * tan(steer) * dt ;  y += v * sin(heading) * dt
    # so a POSITIVE steer angle turns LEFT.
    #
    # CARLA's VehicleControl.steer is the other way round: -1.0 is full LEFT,
    # +1.0 is full RIGHT. Without this negation the car turns the opposite way
    # to every decision the planner makes - it steers INTO whatever it is
    # avoiding, which reads exactly like a broken cost function and is not.
    #
    # Symptom this fixed: on an empty road with 8 vehicles, the ego pinned
    # steer at -0.50 from the first frame, accelerated to 21 km/h, and drove
    # straight off the road into scenery every single run.
    steer_cmd = float(np.clip(-plan.steer / MAX_STEER_RAD, -1.0, 1.0))
    max_delta = 0.15                       # per control tick
    steer_cmd = float(np.clip(steer_cmd, prev_steer - max_delta,
                              prev_steer + max_delta))

    if plan.accel >= 0:
        throttle = float(np.clip(plan.accel / 3.0, 0.0, 1.0))
        brake = 0.0
    else:
        throttle = 0.0
        brake = float(np.clip(-plan.accel / 5.0, 0.0, 1.0))

    return carla.VehicleControl(
        throttle=throttle, steer=steer_cmd, brake=brake,
        hand_brake=False, reverse=False, manual_gear_shift=False,
    ), steer_cmd


def draw_plan(world, bridge, plan, life):
    """Green line = the path the planner just chose."""
    pts = plan.states
    for a, b in zip(pts[:-1], pts[1:]):
        world.debug.draw_line(
            bridge.ego_to_world(a[0], a[1]),
            bridge.ego_to_world(b[0], b[1]),
            thickness=0.12,
            color=carla.Color(40, 255, 40),
            life_time=life,
        )


def draw_risk(world, bridge, risk, life, step=6, thresh=0.25):
    """Risk field, first slice, as coloured dots. Green safe -> red lethal.

    grid[0] is the hazard at t = +dt (0.25 s), not at t = now - the predictor's
    first sample is one step into the future. There is no "now" slice.

    Subsampled hard on purpose: the grid is 240x160 and drawing every cell
    every frame would cost more than the planner does. A few hundred points
    reads better anyway - a solid carpet of dots hides the road.
    """
    g = risk.grid[0]
    dyn = g - risk.static              # traffic only; the road edge is not news
    for iy in range(0, risk.ny, step):
        for ix in range(0, risk.nx, step):
            r = float(dyn[iy, ix])
            if r < thresh:
                continue
            t = min(r / 1.5, 1.0)
            world.debug.draw_point(
                bridge.ego_to_world(float(risk.xs[ix]), float(risk.ys[iy])),
                size=0.06,
                color=carla.Color(int(255 * t), int(255 * (1 - t)), 30),
                life_time=life,
            )


def draw_predictions(world, bridge, predictions, life, max_modes=2):
    """Thin lines showing each agent's most likely futures.

    This is the picture of your novelty module. When a judge asks what
    multimodal prediction means, you point at an agent with two lines coming
    out of it going different ways.
    """
    for pred in predictions:
        modes = sorted(pred.modes, key=lambda m: -m.prob)[:max_modes]
        for m in modes:
            if m.prob < 0.12:
                continue
            shade = int(90 + 165 * m.prob)
            pts = m.points[::3]
            for a, b in zip(pts[:-1], pts[1:]):
                world.debug.draw_line(
                    bridge.ego_to_world(float(a[0]), float(a[1])),
                    bridge.ego_to_world(float(b[0]), float(b[1])),
                    thickness=0.04,
                    color=carla.Color(shade, shade // 3, 255),
                    life_time=life,
                )


class RouteFollower:
    """Minimal goal supplier: a point ~25 m ahead along the road.

    Deliberately dumb - Module 5 replaces it with a real global route. The one
    non-obvious detail is that it HOLDS a goal until the car gets near it,
    rather than recomputing every frame. Recomputing at a junction makes
    wp.next() flip between branches, the goal jumps sideways, and the planner
    swings the wheel back and forth. Sticky goals, smooth driving.
    """

    def __init__(self, carla_map, lookahead=25.0, arrive=10.0):
        self.map = carla_map
        self.lookahead = lookahead
        self.arrive = arrive
        self.goal_wp = None

    def goal(self, ego_location):
        if (self.goal_wp is None or
                self.goal_wp.transform.location.distance(ego_location) < self.arrive):
            wp = self.map.get_waypoint(ego_location)
            nxt = wp.next(self.lookahead)
            if nxt:
                self.goal_wp = nxt[0]
        return (self.goal_wp.transform.location if self.goal_wp is not None
                else ego_location)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm-port", type=int, default=8000)
    ap.add_argument("--town", default="Town01")
    ap.add_argument("--n-vehicles", type=int, default=30)
    ap.add_argument("--n-walkers", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plan-every", type=int, default=2,
                    help="plan every Nth tick; 2 = 10 Hz at a 20 Hz sim")
    ap.add_argument("--v-target", type=float, default=8.0, help="m/s")
    ap.add_argument("--no-debug", action="store_true",
                    help="disable overlays for clean timing numbers")
    args = ap.parse_args()

    scn = Scenario(args)
    collisions = {"n": 0}

    # Declared BEFORE the try on purpose. The finally clause reads them, so a
    # failure during setup would otherwise raise UnboundLocalError from the
    # finally and HIDE the real exception. nova_drive.py documents the same
    # trap; this script had not been given the same treatment.
    lat_samples = []
    min_gap = 999.0

    try:
        scn.setup_sync()
        scn.spawn_ego()

        # THE LINE THAT MATTERS: take the ego off CARLA's autopilot.
        # From here NOVA is driving.
        scn.ego.set_autopilot(False)

        scn.spawn_traffic()
        scn.spawn_walkers()

        # Collision sensor - this is the pass/fail metric, so measure it
        # rather than watching for bumps.
        bp = scn.world.get_blueprint_library().find("sensor.other.collision")
        col_sensor = scn.world.spawn_actor(
            bp, carla.Transform(), attach_to=scn.ego)
        col_sensor.listen(lambda e: collisions.__setitem__("n", collisions["n"] + 1))
        scn.vehicles.append(col_sensor)          # so cleanup destroys it

        bridge = CarlaBridge(scn.world, scn.ego,
                             sample_dt=args.plan_every * 0.05)
        pipe = NovaPipeline(v_target=args.v_target)
        route = RouteFollower(scn.map)

        print("\n" + "=" * 62)
        print("NOVA IS DRIVING - Ctrl+C to stop")
        print("green line = planned path | dots = risk field | "
              "blue = predicted futures")
        print("=" * 62 + "\n")

        frame = 0
        prev_steer = 0.0
        control = carla.VehicleControl()
        last = None
        t0 = time.perf_counter()
        life = args.plan_every * 0.05 + 0.02     # overlay outlives one cycle

        while True:
            scn.world.tick()
            frame += 1
            scn.follow_with_spectator()

            if frame % args.plan_every == 0:
                ego_loc = scn.ego.get_transform().location
                gx, gy = bridge.world_loc_to_ego(route.goal(ego_loc))
                obs = bridge.observe(t=frame * 0.05,
                                     goal=np.array([gx, gy], dtype=np.float64))

                res = pipe.step(obs)
                last = res
                lat_samples.append(res.t_total_ms)

                control, prev_steer = to_control(res.plan, prev_steer)

                if obs.tracks:
                    min_gap = min(min_gap,
                                  min(math.hypot(tr.x, tr.y) for tr in obs.tracks))

                if not args.no_debug:
                    draw_risk(scn.world, bridge, pipe.risk, life)
                    draw_predictions(scn.world, bridge, res.predictions, life)
                    draw_plan(scn.world, bridge, res.plan, life)

            scn.ego.apply_control(control)

            if frame % 60 == 0 and last is not None:
                real = time.perf_counter() - t0
                sim = frame * 0.05
                v = scn.ego.get_velocity()
                kph = 3.6 * math.hypot(v.x, v.y)
                p95 = float(np.percentile(lat_samples[-200:], 95))
                print(f"sim {sim:6.1f}s  ratio {sim/max(real,1e-6):4.2f}x  |  "
                      f"ego {kph:5.1f} km/h  |  "
                      f"plan {last.t_total_ms:5.1f}ms (p95 {p95:5.1f})  |  "
                      f"nodes {last.plan.nodes_expanded:5d}  |  "
                      f"tracks {len(last.predictions):3d}  |  "
                      f"min gap {min_gap:5.1f}m  |  "
                      f"COLLISIONS {collisions['n']}"
                      + ("  <-- EMERGENCY BRAKE" if last.plan.emergency else ""))

    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        if lat_samples:
            print(f"\nlatency: median {np.median(lat_samples):.1f} ms  "
                  f"p95 {np.percentile(lat_samples, 95):.1f} ms  "
                  f"max {np.max(lat_samples):.1f} ms")
            print(f"collisions: {collisions['n']}")
        scn.cleanup()


if __name__ == "__main__":
    main()
