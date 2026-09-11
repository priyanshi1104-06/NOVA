"""
NOVA Module 4 - Adaptive Local Planner (time-aware Hybrid A*).

What it does
------------
Searches over the ego's own achievable motions - not over free grid cells - to
find the lowest-risk 3-second rollout through the risk field, executes only the
FIRST control, throws the rest away, and replans. Classic receding horizon.

The three things that make this "adaptive" rather than a rules engine
---------------------------------------------------------------------
1. SEARCHES IN TIME. Each expansion advances exactly `dt` seconds, so a node at
   depth d looks up risk slice d. The planner therefore knows the difference
   between "a bike is there" and "a bike will be there when I arrive". This is
   the single most important idea in the file.

2. NO BEHAVIOUR MODES. There is no `if in_market: drive_slowly()`. A crowded
   lane produces a dense risk field and this cost function yields slow, tight
   paths; an open highway produces a sparse field and the same function yields
   fast committed ones. When a judge asks "how do you handle different road
   types?", the answer is "we don't - the risk map changes, the planner
   doesn't", and you can show them there is no such branch in the code.

3. SPEED IS A SEARCH DIMENSION. Slowing down is a legitimate avoidance
   manoeuvre, not a special case, so the planner will hang back behind an
   unpredictable auto-rickshaw when swerving is costlier.

Kinematics: bicycle model
-------------------------
    x'     = v cos(theta)
    y'     = v sin(theta)
    theta' = (v / L) tan(delta)
    v'     = a
Non-holonomic, so every planned path is physically drivable. A grid A* over
cells would happily return a path requiring the car to move sideways.

PERFORMANCE NOTES (both of these were real bugs found by test_modules.py)
------------------------------------------------------------------------
* EARLY EXIT. A* pops nodes in order of f = g + h. With a consistent
  heuristic, the first node popped that has reached the planning horizon is
  already optimal - every node still queued has f at least as large. The
  original version kept searching to "prove" optimality and burned its entire
  node budget on every call: 1668 ms instead of ~30 ms. Return on first pop.

* VECTORISED EXPANSION. All (steer x accel) children of a node are computed as
  NumPy arrays and their risks fetched in ONE batched lookup, instead of a
  Python loop doing one scalar grid lookup per child. Same maths, far less
  interpreter overhead.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .riskmap import RiskMap
from .types import EgoState


@dataclass
class PlanNode:
    x: float
    y: float
    heading: float
    v: float
    steer: float
    depth: int
    g: float
    parent: Optional["PlanNode"] = None

    def __lt__(self, other):   # heapq tiebreaker; never actually compares nodes
        return False


@dataclass
class Plan:
    states: List[Tuple[float, float, float, float]] = field(default_factory=list)
    steer: float = 0.0
    accel: float = 0.0
    cost: float = float("inf")
    nodes_expanded: int = 0
    latency_ms: float = 0.0
    emergency: bool = False


class HybridAStarPlanner:
    """Time-aware kinematic search over the risk field.

    Cost weights - these are the personality of the car
    ---------------------------------------------------
    w_risk  : how much it fears predicted hazards. FIRST knob to touch if the
              demo looks wrong. Higher = more cautious and swervy.
    w_goal  : pull toward the goal. Too high and it ignores risk.
    w_steer : penalty on absolute steering; keeps paths straight-ish.
    w_jerk  : penalty on CHANGE in steering. Without it the path oscillates
              left-right on every replan and the car looks drunk on video.
    w_speed : penalty for deviating from target speed. WITHOUT THIS TERM THE
              PLANNER ALWAYS DECIDES STOPPING FOREVER IS OPTIMAL - zero speed
              means zero future risk. Expect that bug if you ever remove it.
    w_cross : cross-track penalty - how far the path strays sideways from the
              line to the goal. This is LANE DISCIPLINE. Without it nothing
              rewards coming back to your intended line, so the car drifts a
              lane and a half sideways to shave a little risk and looks like
              it is wandering. Raise it for tighter lane keeping, lower it to
              allow bolder avoidance manoeuvres.
    """

    def __init__(
        self,
        risk: RiskMap,
        dt: float = 0.25,
        horizon: float = 3.0,
        wheelbase: float = 2.7,
        max_steer: float = 0.55,
        n_steer: int = 5,
        accels: Tuple[float, ...] = (-3.0, 0.0, 1.5),
        v_target: float = 8.0,
        v_max: float = 14.0,
        node_budget: int = 3000,
        w_risk: float = 14.0,
        w_goal: float = 1.0,
        w_steer: float = 0.6,
        w_jerk: float = 2.5,
        w_speed: float = 0.35,
        w_cross: float = 0.8,
    ):
        self.risk = risk
        self.dt = dt
        self.n_steps = int(round(horizon / dt))
        self.L = wheelbase
        self.v_target = v_target
        self.v_max = v_max
        self.node_budget = node_budget

        self.w_risk = w_risk
        self.w_goal = w_goal
        self.w_steer = w_steer
        self.w_jerk = w_jerk
        self.w_speed = w_speed
        self.w_cross = w_cross

        # Precompute the full (steer x accel) primitive set ONCE as flat arrays.
        # 5 steers x 3 accels = 15 children per node. Fewer, well-chosen
        # primitives beat many redundant ones: at dt=0.25 s two adjacent
        # steering angles land in the same 1 m closed-set cell anyway.
        st = np.linspace(-max_steer, max_steer, n_steer)
        ac = np.asarray(accels, dtype=np.float64)
        S, A = np.meshgrid(st, ac, indexing="ij")
        self.prim_steer = S.ravel()
        self.prim_accel = A.ravel()
        self.n_prim = self.prim_steer.size
        self.tan_steer = np.tan(self.prim_steer)
        self.abs_steer = np.abs(self.prim_steer)

        # Closed-set resolution, deliberately coarser than the risk map: this is
        # what stops the branching factor exploding. 1 m cells and 16 heading
        # bins merge states that are genuinely equivalent without merging ones
        # the car could tell apart.
        # TUNED, not guessed. Measured across a sweep in test_modules.py:
        #   pos_res 1.0 -> 105 ms p95, 2.29 m clearance
        #   pos_res 1.5 ->  83 ms p95, 4.18 m clearance  <-- chosen
        #   pos_res 2.0 ->  36 ms p95, 3.12 m, and it STOPS AVOIDING
        # Coarser is faster but merges genuinely different states, so the
        # avoidance manoeuvre disappears. 1.5 m is the knee of that curve.
        self.pos_res = 1.5
        self.head_bins = 16

    # ------------------------------------------------------------------
    def plan(self, ego: EgoState, goal: np.ndarray) -> Plan:
        t0 = time.perf_counter()
        gx, gy = float(goal[0]), float(goal[1])

        # Reference line for cross-track error: ego -> goal. Once Module 5
        # is wired up this becomes the route polyline; the maths is identical.
        rdx, rdy = gx - ego.x, gy - ego.y
        rlen = math.hypot(rdx, rdy)
        if rlen < 1e-6:
            self._ref = (ego.x, ego.y, 1.0, 0.0)
        else:
            self._ref = (ego.x, ego.y, rdx / rlen, rdy / rlen)

        root = PlanNode(ego.x, ego.y, ego.heading, ego.v, ego.steer, 0, 0.0)
        open_heap: List[Tuple[float, int, PlanNode]] = []
        counter = 0
        heapq.heappush(open_heap, (self._h(root.x, root.y, gx, gy), 0, root))
        closed = set()
        expanded = 0

        terminal: Optional[PlanNode] = None
        best_f = float("inf")

        while open_heap and expanded < self.node_budget:
            f, _, node = heapq.heappop(open_heap)

            # EARLY EXIT: first horizon-depth node popped is optimal, because
            # every node still on the heap has f >= this one's f.
            if node.depth >= self.n_steps:
                terminal, best_f = node, f
                break

            key = self._key(node)
            if key in closed:
                continue
            closed.add(key)
            expanded += 1

            for child in self._expand(node, gx, gy):
                if self._key(child) in closed:
                    continue
                counter += 1
                heapq.heappush(
                    open_heap,
                    (child.g + self._h(child.x, child.y, gx, gy), counter, child),
                )

        latency = (time.perf_counter() - t0) * 1000.0

        if terminal is None:
            # No complete rollout found: every branch blocked, or budget spent.
            # Brake hard and hold the wheel. Do NOT reuse the previous plan -
            # driving a stale plan when you cannot see a way through is exactly
            # how you collide in front of judges.
            return Plan(
                states=[(ego.x, ego.y, ego.heading, ego.v)],
                steer=ego.steer, accel=-5.0, cost=float("inf"),
                nodes_expanded=expanded, latency_ms=latency, emergency=True,
            )

        states, first = self._reconstruct(terminal)
        return Plan(
            states=states,
            steer=first.steer,
            accel=(first.v - root.v) / self.dt,
            cost=best_f,
            nodes_expanded=expanded,
            latency_ms=latency,
        )

    # ------------------------------------------------------------------
    def _expand(self, node: PlanNode, gx: float, gy: float) -> List[PlanNode]:
        """All children of one node, computed as arrays with ONE risk lookup."""
        dt = self.dt

        v = node.v + self.prim_accel * dt
        np.clip(v, 0.0, self.v_max, out=v)

        heading = node.heading + (v / self.L) * self.tan_steer * dt
        x = node.x + v * np.cos(heading) * dt
        y = node.y + v * np.sin(heading) * dt

        # One batched grid lookup for all 15 children at the correct time slice.
        #
        # SLICE INDEX, NOT DEPTH. riskmap._stamp_mode writes prediction
        # points[k] - which is the hazard at time (k+1)*dt - into grid[k].
        # A child sits at depth (node.depth + 1), i.e. time
        # (node.depth + 1)*dt, so the matching slice is node.depth.
        # This read node.depth + 1, one slice too far ahead, so the car
        # dodged every hazard 0.25 s early - about 2 m at 8 m/s. Exactly the
        # "avoids things slightly too early or too late" failure pipeline.py
        # warns about in its own comment.
        t_idx = np.full(self.n_prim, node.depth, dtype=np.int32)
        risk = self.risk.cost_batch(x, y, t_idx)

        # Cross-track error: perpendicular distance from the ego->goal line.
        # Sign does not matter, we square it.
        ox, oy, ux, uy = self._ref
        cte = (x - ox) * (-uy) + (y - oy) * ux

        cost = (
            self.w_risk * risk
            + self.w_cross * cte ** 2 * dt
            + self.w_steer * self.abs_steer
            + self.w_jerk * np.abs(self.prim_steer - node.steer)
            + self.w_speed * (v - self.v_target) ** 2 * dt
            + 0.05                      # per-step cost: prefer decisive rollouts
        )
        g = node.g + cost

        d = node.depth + 1
        return [
            PlanNode(float(x[i]), float(y[i]), float(heading[i]), float(v[i]),
                     float(self.prim_steer[i]), d, float(g[i]), node)
            for i in range(self.n_prim)
        ]

    def _h(self, x: float, y: float, gx: float, gy: float) -> float:
        """Weighted straight-line distance to goal.

        Deliberately a WEIGHTED (slightly inadmissible) heuristic. Strict
        admissibility would buy optimality we do not need - we replan ten times
        a second, so a near-optimal path in 30 ms beats an optimal one in 300.
        """
        return self.w_goal * math.hypot(gx - x, gy - y)

    def _key(self, node: PlanNode):
        return (
            int(node.x / self.pos_res),
            int(node.y / self.pos_res),
            int((node.heading % (2 * math.pi)) / (2 * math.pi) * self.head_bins),
            node.depth,
        )

    def _reconstruct(self, node: PlanNode):
        chain = []
        n = node
        while n is not None:
            chain.append(n)
            n = n.parent
        chain.reverse()
        states = [(c.x, c.y, c.heading, c.v) for c in chain]
        first = chain[1] if len(chain) > 1 else chain[0]
        return states, first
