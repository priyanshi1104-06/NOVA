"""
NOVA - lateral control. Follow the road; let the planner handle hazards.

WHY THIS EXISTS - AND WHY IT IS NOT "GIVING UP ON THE PLANNER"
--------------------------------------------------------------
We were asking the Hybrid A* search to do two different jobs:

  1. FOLLOW THE ROAD - stay in the lane, take the corner, hold a smooth line
  2. AVOID HAZARDS   - route around a stopped truck, brake for a pedestrian

It is genuinely good at (2) and genuinely bad at (1), and that is not a bug
to tune away, it is what the algorithm is. The search quantises position to a
1.5 m closed-set cell and heading to 16 bins, expands 5 discrete steering
angles, and looks 3 s ahead. That is the right tool for "is there a way
through this gap", and the wrong tool for "hold the centre of a 3.5 m lane" -
the lane is barely two cells wide, so a path that hugs the kerb and a path
down the middle score almost identically.

The result on the road was exactly that: the car wandered, clipped kerbs,
left the carriageway, and ended up nose-to-a-guardrail. Meanwhile CARLA's own
autopilot drove the same streets smoothly, because it does what every real
stack does - it TRACKS A REFERENCE PATH with a dedicated controller.

So: pure pursuit steers along the route, the planner and the safety governor
own speed and emergencies. Each part now does the job it is good at. This is
the standard decomposition, not a workaround.

WHEN THE PLANNER STILL STEERS
-----------------------------
If the planner is in emergency, or its chosen path departs hard from the
route, its steering wins - that is it routing around something the route
knows nothing about. Normal driving is pure pursuit; avoidance is the
planner. See `blend()`.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple


class PurePursuit:
    """Classic pure-pursuit tracker over route points in the EGO frame.

    Parameters
    ----------
    wheelbase : metres. Must match the planner's, or the two disagree about
        what a given steering angle does.
    k_speed : lookahead grows with speed - `L = k_speed * v + l_min`. Short
        lookahead at low speed turns tightly into junctions; longer at speed
        keeps the line smooth instead of sawing at the wheel.
    max_steer : radians, the same limit the planner searches within.
    """

    def __init__(self, wheelbase: float = 2.7, k_speed: float = 1.1,
                 l_min: float = 4.5, l_max: float = 16.0,
                 max_steer: float = 0.55):
        self.L = wheelbase
        self.k_speed = k_speed
        self.l_min = l_min
        self.l_max = l_max
        self.max_steer = max_steer
        self.last_target: Optional[Tuple[float, float]] = None

    # ------------------------------------------------------------------
    def _target(self, route_xy: Sequence[Tuple[float, float]], v: float):
        """The first route point at least `L` metres ahead of the car."""
        L = min(max(self.k_speed * v + self.l_min, self.l_min), self.l_max)
        best = None
        for x, y in route_xy:
            if x <= 0.2:                       # behind or beside us
                continue
            if math.hypot(x, y) >= L:
                best = (x, y)
                break
        if best is None:
            # Route ends inside the lookahead: aim at the furthest point we
            # have rather than giving up, so the car still finishes the leg.
            ahead = [(x, y) for x, y in route_xy if x > 0.2]
            best = ahead[-1] if ahead else None
        return best

    def steer(self, route_xy: Sequence[Tuple[float, float]],
              v: float) -> Optional[float]:
        """Steering angle in radians, +ve = LEFT (NOVA convention).

        Returns None when there is no usable target, so the caller can fall
        back to the planner rather than driving on a stale value.
        """
        if not route_xy:
            return None
        tgt = self._target(route_xy, v)
        if tgt is None:
            return None
        self.last_target = tgt

        x, y = tgt
        ld = math.hypot(x, y)
        if ld < 1e-3:
            return None
        # Pure pursuit: curvature = 2*y / L^2, steer = atan(curvature * L).
        # y is the lateral offset of the target in the ego frame, +y LEFT, so
        # a target to the left gives a positive (left) steering angle - which
        # matches the planner's sign convention and step_c_drive's negation.
        curvature = 2.0 * y / (ld * ld)
        angle = math.atan(curvature * self.L)
        return max(-self.max_steer, min(self.max_steer, angle))


def blend(pp_steer: Optional[float], plan_steer: float, emergency: bool,
          hazard_near: bool = False) -> Tuple[float, str]:
    """Choose between the tracker and the planner. Returns (steer, who).

    THE OVERRIDE MUST BE GATED ON A HAZARD, NOT ON DISAGREEMENT.
    The first version handed control to the planner whenever it differed from
    the route by more than 0.25 rad, reasoning that the difference must be an
    avoidance manoeuvre. It is not. On an empty road approaching a junction
    the planner says "straight" - because a 1.5 m search grid cannot express
    the turn - while the tracker says "hard left". That is maximum
    disagreement with NO hazard present, so the planner won and the turn was
    cancelled. Measured: goal 4.5 m to the left, steering output -0.15,
    route progress stuck at 0.5%.

    So the planner only takes the wheel when there is something to avoid:
    it is braking hard, or a tracked hazard is close. Otherwise the tracker
    drives, which on an empty road is simply "follow the lane".
    """
    if pp_steer is None:
        return plan_steer, "planner"
    if emergency:
        return plan_steer, "planner:emergency"
    if hazard_near:
        return plan_steer, "planner:avoid"
    return pp_steer, "tracker"


class Stanley:
    """Cross-track controller. Holds the LINE, where pure pursuit chases a POINT.

    WHY THIS REPLACED PURE PURSUIT AS THE DEFAULT
    ---------------------------------------------
    Pure pursuit steers at a point `L` metres ahead on the route. On a
    straight that is fine. On a tight junction it is not: the lookahead point
    sits across the corner, and the shortest arc to it cuts the corner - which
    in Town01 means mounting the kerb. Photographed doing exactly that, front
    wheels on the pavement at a right-hand bend.

    Pure pursuit has no term for "how far am I from the path". Stanley does:

        steer = heading_error + atan( k * cross_track / (v + eps) )

    The first term aligns the car with the road's direction. The second pulls
    it back onto the centreline, harder the further off it is and more gently
    the faster it is going. So it tracks the path itself rather than a point
    beyond it, and corners stop being cut.

    This is the controller from DARPA Grand Challenge-winning Stanley, and it
    is the standard choice when tracking accuracy matters more than comfort.
    """

    def __init__(self, k: float = 1.6, k_soft: float = 1.0,
                 max_steer: float = 0.55, front_axle: float = 1.35):
        self.k = k                      # cross-track gain
        self.k_soft = k_soft            # stops the term exploding at v -> 0
        self.max_steer = max_steer
        self.front_axle = front_axle    # metres ahead of the ego origin

    def steer(self, route_xy, v: float) -> Optional[float]:
        """Steering in radians, +ve = LEFT. None if the route is unusable."""
        if not route_xy or len(route_xy) < 2:
            return None

        # Stanley works at the FRONT AXLE, not the centre of mass.
        fx, fy = self.front_axle, 0.0

        # Nearest route segment to the front axle.
        best_i, best_d = None, float("inf")
        for i in range(len(route_xy) - 1):
            px, py = route_xy[i]
            d = math.hypot(px - fx, py - fy)
            if d < best_d:
                best_d, best_i = d, i
        if best_i is None:
            return None

        ax, ay = route_xy[best_i]
        bx, by = route_xy[min(best_i + 1, len(route_xy) - 1)]
        seg_x, seg_y = bx - ax, by - ay
        seg_len = math.hypot(seg_x, seg_y)
        if seg_len < 1e-6:
            return None

        # Heading error: the path's direction in the ego frame. The ego points
        # along +x by definition, so the path angle IS the heading error.
        heading_err = math.atan2(seg_y, seg_x)

        # Cross-track error: signed perpendicular distance from the front axle
        # to the segment. +ve when the path is to our LEFT, matching +y LEFT,
        # so a path on the left produces a left correction.
        cross = ((bx - ax) * (fy - ay) - (by - ay) * (fx - ax)) / seg_len
        cross_term = math.atan2(self.k * -cross, self.k_soft + v)

        angle = heading_err + cross_term
        return max(-self.max_steer, min(self.max_steer, angle))
