"""
NOVA - MODULE 5: the global route. A to B, on roads, with junctions resolved.

WHY THIS EXISTS
---------------
Before this, the planner's goal came from RouteFollower:

    wp = carla_map.get_waypoint(ego_location)
    goal = wp.next(25.0)[0]

`next()` returns EVERY branch at a junction and that code took the first one,
arbitrarily. So at a junction the goal could point down a turn the car was not
lined up for; the planner drove at it, crossed the kerb, and wedged. That is
the "why does it collide" bug, and no amount of planner tuning fixes it -
the planner was solving the problem it was given, and the problem was wrong.

A route fixes it because the goal is always a point the car can actually
reach by driving on the road.

WHAT IT GIVES YOU
-----------------
  goal(ego_location)      -> carla.Location, a lookahead point ON the route
  remaining(ego_location) -> the route ahead, for drawing on the HUD
  progress(ego_location)  -> 0..1, how far along; 1.0 means arrived

The heavy lifting is CARLA's own GlobalRoutePlanner, which builds a NetworkX
graph of the road network and runs A* over it. Using it rather than
reimplementing is deliberate: it already handles lane connectivity, junction
topology and turn options correctly, and a hand-rolled version would be a
worse copy of it.

THE ONE SUBTLETY: STICKY LOOKAHEAD.
Recomputing the nearest route point every frame makes the goal jitter, and the
planner's w_jerk term turns that into visible wheel-wobble. So we track an
INDEX into the route and only ever move it forward. The car cannot go
backwards along its own route, which is also what stops it turning around when
it briefly overshoots a corner.
"""

from __future__ import annotations

import math
import os
import sys
from typing import List, Optional, Tuple

# CARLA ships the route planner inside its PythonAPI tree rather than in the
# pip package, so it has to be found on disk. Checked in order: an explicit
# override, then the usual install path.
_CANDIDATES = [
    os.environ.get("CARLA_PYTHONAPI", ""),
    r"C:\CARLA_0.9.16\PythonAPI\carla",
]
for _p in _CANDIDATES:
    if _p and os.path.isdir(os.path.join(_p, "agents")) and _p not in sys.path:
        sys.path.append(_p)


class GlobalRoute:
    """A driveable route from the ego's position to a destination.

    Parameters
    ----------
    world : carla.World
    sampling : metres between route points. 2 m is fine - the planner only
        ever consumes a lookahead point, and denser sampling costs memory in
        the NetworkX graph for no benefit.
    lookahead : how far along the route to place the goal. Must exceed what
        the planner can travel in its horizon (3 s at v_target 8 m/s = 24 m)
        or the car arrives at its goal mid-search and has nothing to aim at.
    """

    def __init__(self, world, carla_map=None, seed: Optional[int] = None,
                 sampling: float = 2.0, lookahead: float = 30.0):
        import random

        import carla
        from agents.navigation.global_route_planner import GlobalRoutePlanner

        self._carla = carla
        self.world = world
        self.map = carla_map if carla_map is not None else world.get_map()
        self.lookahead = lookahead
        self.sampling = sampling
        self.rng = random.Random(seed)
        self._grp = GlobalRoutePlanner(self.map, sampling)
        self.points: List["carla.Location"] = []
        self._i = 0                       # index of the nearest route point
        self.replans = 0                  # how often we had to re-plan

    # ------------------------------------------------------------------
    def ensure(self, ego_location) -> int:
        """Plan a route from here if we do not have one yet.

        Called lazily from goal(), so the caller can construct GlobalRoute
        before the ego has a real transform. A freshly spawned actor reports
        (0, 0, 0) until the world ticks - planning from there would route the
        car from the map origin, which is nowhere near it.
        """
        if not self.points:
            self.build_random(ego_location, self.rng)
        return len(self.points)

    # ------------------------------------------------------------------
    def build(self, start, destination) -> int:
        """Plan start -> destination. Returns the number of route points."""
        trace = self._grp.trace_route(start, destination)
        self.points = [wp.transform.location for wp, _road_option in trace]
        self._i = 0
        return len(self.points)

    def build_random(self, start, rng, min_length: float = 150.0,
                     tries: int = 20) -> int:
        """Plan to a random spawn point that is FAR ENOUGH AWAY to be a demo.

        Rejects short routes: a 30 m route ends before anything interesting
        happens, and the car then sits at its destination looking broken.
        """
        spawns = self.map.get_spawn_points()
        best: List = []
        for _ in range(tries):
            dest = rng.choice(spawns).location
            if dest.distance(start) < min_length:
                continue
            if self.build(start, dest) >= 2:
                if self.length() >= min_length:
                    return len(self.points)
                if len(self.points) > len(best):
                    best = list(self.points)
        if best:
            self.points = best
            self._i = 0
        return len(self.points)

    # ------------------------------------------------------------------
    def length(self) -> float:
        return sum(a.distance(b) for a, b in zip(self.points[:-1],
                                                 self.points[1:]))

    # If the car is further than this from its own route, the route is no
    # longer a description of where it is - re-plan from where it actually is.
    OFF_ROUTE_M = 9.0

    def _advance(self, ego_location) -> None:
        """Move the cursor to the nearest point AHEAD of where it already is.

        Forward-only, so the goal cannot jitter backwards - see the module
        docstring. The window is bounded to keep this O(1) per frame.

        BUT FORWARD-ONLY NEEDS AN ESCAPE HATCH. Without one, a car that leaves
        the route is stuck for good: every remaining point is further away, so
        the cursor freezes, the goal becomes a fixed world position it cannot
        reach, and it drives at that spot forever. Measured: route progress
        pinned at 6.3% for 7000+ frames while the car shuttled into the same
        guardrail ten times.

        A real navigation system re-plans when you miss a turn. So does this.
        """
        if not self.points:
            return
        best_i, best_d = self._i, float("inf")
        hi = min(len(self.points), self._i + 60)
        for i in range(self._i, hi):
            d = self.points[i].distance(ego_location)
            if d < best_d:
                best_d, best_i = d, i

        if best_d > self.OFF_ROUTE_M:
            # Off route. Re-plan from here rather than chase a stale cursor.
            self.points = []
            self._i = 0
            self.replans += 1
            self.build_random(ego_location, self.rng)
            return
        self._i = best_i

    def force_replan(self, ego_location) -> int:
        """Throw the current route away and plan afresh from here.

        Needed when the car ends up FACING THE WRONG WAY along its route -
        after a collision spins it, say. The cursor is still near the route,
        so the off-route distance check does not fire, but every lookahead
        point is behind the car. The planner's primitives are forward-only, so
        a goal behind it is unreachable and it creeps at 0.1 km/h forever.
        Measured: goal=(-0.7 m) with route progress pinned at 6.3%.
        """
        self.points = []
        self._i = 0
        self.replans += 1
        return self.build_random(ego_location, self.rng)

    def goal(self, ego_location, speed: float = None):
        """A point further along the route, INSIDE the planner's horizon.

        THE LOOKAHEAD MUST BE REACHABLE. This is what stopped the car turning.
        The planner searches 3 s ahead - 15 m at 5 m/s - but the goal sat at a
        fixed 30 m, permanently outside the reachable set. Every path the
        search could build therefore pointed straight at a goal it could never
        arrive at, so a corner 12 m ahead got cut instead of followed. The car
        drove forward, stopped, reversed, and never took a turn.

        Pure-pursuit practice is a lookahead of roughly two seconds of travel,
        with a floor so it does not collapse to zero when stopped. Keeping it
        INSIDE the horizon means the search can actually reach the goal, and
        the cheapest way to reach a goal round a corner is to go round the
        corner.
        """
        self.ensure(ego_location)
        if speed is not None:
            self.lookahead = max(7.0, min(2.0 * speed, 18.0))
        if not self.points:
            return ego_location
        # Arrived? Plan a fresh route onward so the demo keeps driving instead
        # of parking at its destination looking broken.
        if self.arrived(ego_location):
            self.points = []
            self.ensure(ego_location)
        self._advance(ego_location)
        travelled = 0.0
        i = self._i
        while i + 1 < len(self.points) and travelled < self.lookahead:
            travelled += self.points[i].distance(self.points[i + 1])
            i += 1
        return self.points[i]

    def remaining(self, ego_location, max_points: int = 400):
        """The route from here on, for drawing. Cheap - it is a slice."""
        if not self.points:
            return []
        self._advance(ego_location)
        return self.points[self._i:self._i + max_points]

    def progress(self, ego_location) -> float:
        if not self.points:
            return 0.0
        self._advance(ego_location)
        return self._i / max(1, len(self.points) - 1)

    def arrived(self, ego_location, tol: float = 8.0) -> bool:
        return bool(self.points) and (
            self.points[-1].distance(ego_location) < tol)


def pick_clear_spawn(world, carla_map, rng, tries: int = 40,
                     clearance: float = 12.0):
    """A spawn point with ROAD AHEAD and nothing parked on top of it.

    spawn_ego() used to take the first point that accepted an actor. That
    only proves the point was empty at that instant - not that the car has
    anywhere to go. Spawning nose-first at a junction or a dead end meant the
    first route goal was already across a kerb, and the car drove into it
    within a couple of seconds.

    Here we additionally require that the road continues `clearance` metres
    ahead, and that no other vehicle is sitting within that distance.
    """
    spawns = list(carla_map.get_spawn_points())
    rng.shuffle(spawns)
    others = [a.get_transform().location
              for a in world.get_actors().filter("vehicle.*")]

    for sp in spawns[:tries]:
        wp = carla_map.get_waypoint(sp.location)
        if wp is None or wp.is_junction:
            continue                       # do not start nose-first in a junction
        if not wp.next(clearance):
            continue                       # no road ahead
        if any(o.distance(sp.location) < clearance for o in others):
            continue                       # someone is already there
        return sp
    return spawns[0] if spawns else None


def corridor_mask(route_xy, xs, ys, half_width: float = 4.0):
    """Rasterise a tube around the route, in the ego grid. 1 = inside.

    WHY A CORRIDOR AND NOT JUST "IS IT ROAD".
    The map-derived drivable mask says "this cell is some road". That still
    lets the planner cut diagonally across a junction, mount the pavement on
    the inside of a turn, or wander onto a side street - all of which it did,
    and which ended with the ego parked on a plaza beside a guardrail.

    The route already says which road we are meant to be on. Confining the
    planner to a tube around it turns "stay on tarmac" into "stay on YOUR
    lane", which is what following a mapped path actually means.

    half_width defaults to 4 m: about one lane plus a margin, so there is
    still room to swerve around an obstacle without leaving the corridor.

    Combine with the drivable mask by multiplying - a cell must be BOTH road
    AND on-route.
    """
    import numpy as np

    grid = np.zeros((len(ys), len(xs)), dtype=np.uint8)
    if not route_xy:
        return grid + 1                   # no route yet: do not constrain

    res_x = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
    res_y = float(ys[1] - ys[0]) if len(ys) > 1 else 1.0
    rx = int(max(1, round(half_width / abs(res_x))))
    ry = int(max(1, round(half_width / abs(res_y))))

    x0, y0 = float(xs[0]), float(ys[0])
    for px, py in route_xy:
        ix = int(round((px - x0) / res_x))
        iy = int(round((py - y0) / res_y))
        xa, xb = max(0, ix - rx), min(len(xs), ix + rx + 1)
        ya, yb = max(0, iy - ry), min(len(ys), iy + ry + 1)
        if xa < xb and ya < yb:
            grid[ya:yb, xa:xb] = 1

    # The ego is always inside its own corridor. Without this, a moment where
    # the route pointer lags behind the car boxes the planner in completely
    # and it brakes to a stop for no visible reason.
    ix0 = int(round((0.0 - x0) / res_x))
    iy0 = int(round((0.0 - y0) / res_y))
    xa, xb = max(0, ix0 - rx), min(len(xs), ix0 + rx + 1)
    ya, yb = max(0, iy0 - ry), min(len(ys), iy0 + ry + 1)
    if xa < xb and ya < yb:
        grid[ya:yb, xa:xb] = 1
    return grid
