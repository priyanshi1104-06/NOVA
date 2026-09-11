"""
NOVA - top-down navigation map. Fills the panel the camera cannot.

WHY THIS EXISTS
---------------
Two problems, one answer.

1. With the cameras disabled (this machine's CARLA hangs when any camera
   sensor is attached) the largest panel on the dashboard was a black
   rectangle. A blank half-screen reads as "broken", however well the rest
   of the stack is working.

2. Nothing on screen showed WHERE THE CAR IS GOING. The risk map is
   ego-relative and 50 m deep, so a 300 m route through a town is invisible
   in it - you see a line leaving the top of the panel and nothing else.

A top-down map answers both: the road network for context, the full planned
route, the ego on it, and every tracked agent drawn in its CLASS COLOUR with
a label. That last part matters for the pitch: NOVA classifies cars, trucks,
buses, two-wheelers, bicycles and pedestrians, and until now that
classification was only ever drawn on camera detections - so with the camera
off it looked like the system did not classify anything at all. It does; it
was simply never shown.

Everything here is drawing. No decision in the driving stack reads it.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .types import AgentClass

BG = (16, 15, 20)
ROAD = (54, 52, 60)
ROUTE = (150, 120, 60)
ROUTE_DONE = (70, 62, 55)
EGO = (255, 255, 255)
INK = (238, 234, 228)
DIM = (150, 148, 155)

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

SHORT = {
    AgentClass.CAR: "CAR",
    AgentClass.TRUCK: "TRUCK",
    AgentClass.BUS: "BUS",
    AgentClass.TWO_WHEELER: "2WH",
    AgentClass.BICYCLE: "CYCLE",
    AgentClass.PEDESTRIAN: "PED",
    AgentClass.AUTO_RICKSHAW: "AUTO",
    AgentClass.ANIMAL: "ANIMAL",
    AgentClass.PUSHCART: "CART",
    AgentClass.UNKNOWN: "?",
}


class NavigationMap:
    """Renders a north-up town map with the route, the ego and the traffic.

    The road network is rasterised ONCE - it never changes - and only the
    moving parts are drawn per frame. Rebuilding it every frame would cost
    more than the planner.
    """

    def __init__(self, width: int = 640, height: int = 480,
                 span: float = 140.0):
        self.w = width
        self.h = height
        self.span = span              # metres shown across the shorter axis
        self._roads: Optional[np.ndarray] = None      # (N,2) world x, -y

    # ------------------------------------------------------------------
    def set_roads(self, points: Sequence[Tuple[float, float]]) -> None:
        """World-space road samples, already mirrored to NOVA's +y LEFT."""
        self._roads = (np.asarray(points, dtype=np.float32)
                       if len(points) else None)

    def _to_px(self, X, Y, cx, cy, scale):
        """World (NOVA frame) -> pixels, north-up, ego centred."""
        u = self.w * 0.5 + (Y - cy) * -scale      # +y LEFT -> left on screen
        v = self.h * 0.5 - (X - cx) * scale       # +x forward -> up
        return u, v

    # ------------------------------------------------------------------
    def render(self, ego_xy, ego_yaw, route_world, tracks_world,
               progress: float = 0.0, replans: int = 0) -> np.ndarray:
        """One frame. All inputs are in the NOVA world frame (+y LEFT)."""
        img = np.full((self.h, self.w, 3), BG, np.uint8)
        cx, cy = ego_xy
        scale = min(self.w, self.h) / self.span

        # --- road network -------------------------------------------------
        if self._roads is not None and len(self._roads):
            near = self._roads[
                (np.abs(self._roads[:, 0] - cx) < self.span) &
                (np.abs(self._roads[:, 1] - cy) < self.span)]
            for X, Y in near:
                u, v = self._to_px(X, Y, cx, cy, scale)
                if 0 <= u < self.w and 0 <= v < self.h:
                    cv2.circle(img, (int(u), int(v)), 3, ROAD, -1)

        # --- the route ----------------------------------------------------
        if route_world:
            pts = [self._to_px(X, Y, cx, cy, scale) for X, Y in route_world]
            for i in range(len(pts) - 1):
                a = (int(pts[i][0]), int(pts[i][1]))
                b = (int(pts[i + 1][0]), int(pts[i + 1][1]))
                cv2.line(img, a, b, ROUTE, 3, cv2.LINE_AA)

        # --- other agents, in class colour, labelled ----------------------
        for X, Y, cls in tracks_world:
            u, v = self._to_px(X, Y, cx, cy, scale)
            if not (0 <= u < self.w and 0 <= v < self.h):
                continue
            col = CLASS_COLOUR.get(cls, CLASS_COLOUR[AgentClass.UNKNOWN])
            cv2.circle(img, (int(u), int(v)), 6, col, -1)
            cv2.circle(img, (int(u), int(v)), 6, (30, 30, 34), 1)
            cv2.putText(img, SHORT.get(cls, "?"), (int(u) + 9, int(v) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, col, 1, cv2.LINE_AA)

        # --- the ego, as an arrow so heading is visible --------------------
        eu, ev = self.w * 0.5, self.h * 0.5
        nose = (int(eu + math.sin(ego_yaw) * 14 * -1),
                int(ev - math.cos(ego_yaw) * 14))
        cv2.circle(img, (int(eu), int(ev)), 7, EGO, -1)
        cv2.line(img, (int(eu), int(ev)), nose, EGO, 2, cv2.LINE_AA)

        # --- labels --------------------------------------------------------
        cv2.putText(img, "NAVIGATION MAP   north up", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, DIM, 1, cv2.LINE_AA)
        cv2.putText(img, f"route {progress * 100:.0f}%   re-plans {replans}",
                    (10, self.h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, INK,
                    1, cv2.LINE_AA)
        bar_w = int((self.w - 20) * max(0.0, min(progress, 1.0)))
        cv2.rectangle(img, (10, self.h - 34), (self.w - 10, self.h - 28),
                      (40, 38, 45), -1)
        if bar_w > 0:
            cv2.rectangle(img, (10, self.h - 34), (10 + bar_w, self.h - 28),
                          ROUTE, -1)
        return img
