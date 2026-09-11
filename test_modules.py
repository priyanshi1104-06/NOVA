"""
Unit test for NOVA Modules 2, 3, 4 — runs WITHOUT CARLA.

Scenario: a motorcycle ahead-right of us is drifting left across our path.
This is the exact situation the whole project exists to handle.

What we are checking:
  1. Prediction produces multiple weighted futures, not one
  2. The risk map lights up where the bike is GOING, not just where it is
  3. The planner steers around it BEFORE it arrives
  4. All of it runs fast enough to be called 10x/second
"""
import numpy as np
from nova.types import AgentClass, EgoState, Observation, Track
from nova.pipeline import NovaPipeline


def time_aware_clearance(plan, predictions, dt=0.25, p_min=0.10):
    """Closest the plan ever comes to a plausible future, comparing positions
    AT THE SAME INSTANT.

    THIS IS THE METRIC THAT MUST NOT BE REPLACED BY THE OBVIOUS ONE.

    The obvious metric - shortest distance from the path to where an object is
    RIGHT NOW - contradicts the entire reason the risk grid is 3D. The car is
    supposed to drive through space the bike has already left. Judged against
    the bike's current position, a plan that correctly threads the gap behind
    it scores the same as one that drives straight into where it is heading.

    That is not hypothetical. Fixing the planner's risk-slice off-by-one moved
    this number 3.96 -> 6.22 m while the naive one went 4.18 -> 4.17 m. The
    naive metric would have reported the fix as a no-op.

    Index alignment: plan.states[d] is the ego at time d*dt; mode.points[k] is
    that object at time (k+1)*dt. So state d pairs with points[d-1], and
    state 0 (now) has no prediction to pair with.

    Returns (metres, which encounter was worst).
    """
    worst, who = float("inf"), "nothing in range"
    for pred in predictions:
        for m in pred.modes:
            if m.prob < p_min:          # futures we barely believe in
                continue
            for d, st in enumerate(plan.states):
                if d == 0 or d - 1 >= len(m.points):
                    continue
                gap = float(np.hypot(st[0] - m.points[d - 1][0],
                                     st[1] - m.points[d - 1][1]))
                if gap < worst:
                    worst = gap
                    who = (f"{pred.cls.value} '{m.label}' p={m.prob:.2f} "
                           f"at t=+{d * dt:.2f}s")
    return worst, who

# --- build a synthetic frame ------------------------------------------
# Motorcycle 15 m ahead, 3 m to our RIGHT (y is +left, so y=-3),
# moving forward at 9 m/s and already drifting left toward our lane.
# History is OLDEST FIRST. x must INCREASE (it is moving forward) and y must
# curve LEFT with increasing rate, so yaw_rate() reads a genuine left turn.
# Getting this backwards silently gives yaw_rate 0 and equal left/right
# probabilities - which is exactly the bug the first run of this test caught.
# Steps in y GROW (0.05, 0.15, 0.25, 0.35) = accelerating left drift.
hist = [(11.4 + 0.9*i, -3.8 + 0.05*i*i) for i in range(5)]
bike = Track(
    id=1, cls=AgentClass.TWO_WHEELER,
    x=15.0, y=-3.0, heading=0.05, vx=9.0, vy=0.7,
    history=hist,
)
# A bus in the left lane, steady — should barely spread.
bus = Track(
    id=2, cls=AgentClass.BUS,
    x=22.0, y=3.5, heading=0.0, vx=7.0, vy=0.0,
    history=[(22.0 - 7*0.1*i, 3.5) for i in range(5)],
)
# A pedestrian standing at the kerb. Not moving. Should STILL get a cross mode.
ped = Track(
    id=3, cls=AgentClass.PEDESTRIAN,
    x=30.0, y=-6.0, heading=0.0, vx=0.0, vy=0.0,
    history=[(30.0, -6.0)]*5,
)

ego = EgoState(x=0.0, y=0.0, heading=0.0, v=8.0, steer=0.0)
obs = Observation(t=0.0, ego=ego, tracks=[bike, bus, ped],
                  goal=np.array([45.0, 0.0]))

pipe = NovaPipeline(v_target=8.0)
res = pipe.step(obs)

# --- 1. prediction -----------------------------------------------------
print("="*66)
print("MODULE 2 — INTENT PREDICTION")
print("="*66)
for p in res.predictions:
    print(f"\n  track {p.track_id}  ({p.cls.value})")
    for m in sorted(p.modes, key=lambda m: -m.prob):
        end = m.points[-1]
        print(f"    {m.label:<10} p={m.prob:0.2f}  "
              f"ends at x={end[0]:6.1f} y={end[1]:6.1f}  "
              f"sigma_final={m.sigma[-1]:.2f} m")

# --- 2. risk map -------------------------------------------------------
print("\n" + "="*66)
print("MODULE 3 — RISK MAP")
print("="*66)
g = pipe.risk.grid
print(f"  grid shape (t, y, x) = {g.shape}")
print(f"  covers x {pipe.risk.x_min}..{pipe.risk.x_max} m, "
      f"y {pipe.risk.y_min}..{pipe.risk.y_max} m @ {pipe.risk.res} m/cell")
# Where is the bike now vs where does risk peak 2s from now?
print(f"  bike is NOW at x={bike.x:.1f} y={bike.y:.1f}")
print("  where the DYNAMIC hazard peaks over time (static layer excluded):")
for t_idx in (0, 4, 8, 11):
    px, py, pv = pipe.risk.dynamic_peak(t_idx)
    # grid[k] holds the hazard at (k+1)*dt - there is no t=0 slice.
    print(f"    t=+{(t_idx+1)*0.25:.2f}s  peak at x={px:6.1f} y={py:6.1f}  value={pv:.2f}")

# --- 3. planner --------------------------------------------------------
print("\n" + "="*66)
print("MODULE 4 — PLANNER")
print("="*66)
plan = res.plan
xs = [s[0] for s in plan.states]
ys = [s[1] for s in plan.states]
vs = [s[3] for s in plan.states]
print(f"  nodes expanded : {plan.nodes_expanded}")
print(f"  emergency      : {plan.emergency}")
print(f"  first control  : steer={plan.steer:+.3f} rad  accel={plan.accel:+.2f} m/s^2")
print(f"  path lateral   : y goes {ys[0]:+.2f} -> {max(ys, key=abs):+.2f} m")

clear_m, clear_who = time_aware_clearance(plan, res.predictions)
naive = float(min(np.hypot(np.array(xs) - bike.x, np.array(ys) - bike.y)))
print(f"  CLEARANCE      : {clear_m:.2f} m   worst case over every plausible")
print(f"                   future, compared at matching times")
print(f"                   ({clear_who})")
print(f"  naive gap      : {naive:.2f} m   distance to where the bike is NOW -")
print(f"                   shown only because it barely moves when the plan")
print(f"                   genuinely improves. Do not tune against it.")
print(f"  path speed     : {vs[0]:.1f} -> {vs[-1]:.1f} m/s")
print(f"  reaches x      : {xs[-1]:.1f} m in {(len(xs)-1)*0.25:.2f}s")

# --- 4. timing ---------------------------------------------------------
print("\n" + "="*66)
print("TIMING  (budget: 100 ms for 10 Hz)")
print("="*66)
print(f"  predict : {res.t_predict_ms:7.2f} ms")
print(f"  riskmap : {res.t_risk_ms:7.2f} ms")
print(f"  planner : {res.t_plan_ms:7.2f} ms")
print(f"  TOTAL   : {res.t_total_ms:7.2f} ms   "
      f"-> {1000/res.t_total_ms:.1f} Hz")
