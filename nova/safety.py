"""
NOVA - the safety governor. Slow down, and stop, when a collision is coming.

WHY THIS IS A SEPARATE LAYER
----------------------------
The problem statement asks for "path navigation AND collision avoidance", and
specifically that the vehicle SLOWS OR STOPS when it is about to collide. The
planner alone did not deliver that, for a structural reason worth knowing:

  * It expresses caution through a COST. w_risk makes hazardous paths
    expensive, so the search prefers a way around. That is avoidance by
    steering, not by braking, and it only works if a way around exists.
  * Its only braking behaviour was a last-resort emergency stop, taken when
    the search found NO feasible path at all (planner.py, emergency=True).
    Between "comfortable" and "nothing is possible" there was nothing.

So on the road the car ran at `accel=+1.50, thr=0.50` almost continuously and
drove into a Volkswagen at 25 km/h. The planner was not malfunctioning; there
was simply no rule that said "something is close ahead, ease off".

This module is that rule. It is deliberately a SEPARATE, LAST stage, after
the planner, and it can only ever slow the car down - never speed it up or
steer it. That ordering matters:

  * it is simple enough to explain to a judge in one sentence, and to trust
  * it cannot be defeated by cost-weight tuning
  * it still applies if the planner is swapped for a learned policy

This is the same argument for a safety monitor sitting outside the planner
that you would make for a real vehicle.

TIME TO COLLISION
-----------------
For each tracked agent we compute the time until the ego reaches it ALONG THE
EGO'S OWN PATH, using closing speed rather than raw distance. A car 10 m ahead
travelling the same speed is not a hazard; a car 10 m ahead that has stopped
is. Distance alone cannot tell those apart, which is why `min_gap` in the HUD
is a reporting number and not a control input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .types import AgentClass, EgoState, Track

# Lateral half-width of the corridor we care about, in metres. Wider than the
# car (1.0 m half-width) because a vehicle drifting toward our lane matters
# before it is actually in it.
CORRIDOR_HALF_W = 1.9

# Vulnerable road users get a wider berth and an earlier reaction. On Indian
# roads this is the difference that matters, and it is cheap to state.
VRU = (AgentClass.PEDESTRIAN, AgentClass.BICYCLE, AgentClass.TWO_WHEELER)
VRU_EXTRA_M = 0.7
VRU_EXTRA_TTC = 0.6


@dataclass
class SafetyDecision:
    """What the governor did, so the HUD can show it and metrics can log it."""
    accel: float                       # m/s^2, the command actually issued
    ttc: float = float("inf")          # seconds to the closest threat
    gap: float = float("inf")          # metres to it, along the path
    reason: str = ""                   # "", "SLOW", or "BRAKE"
    threat: Optional[int] = None       # track id, for the HUD


class SafetyGovernor:
    """Caps the planner's acceleration by time-to-collision.

    Parameters
    ----------
    t_brake : below this TTC, brake as hard as the tyres allow.
    t_slow  : below this TTC, ease off proportionally. Between t_brake and
        t_slow the response ramps, so the car lifts off early and gently
        rather than doing nothing and then panicking - which is both safer
        and what reads as competent on video.
    a_brake : maximum deceleration, m/s^2. 5.0 matches the planner's own
        emergency value and CARLA's Model 3 on dry asphalt.
    stop_gap : hold a full stop until there is at least this much room. Stops
        the car creeping into whatever it just stopped for.
    """

    def __init__(self, t_brake: float = 1.6, t_slow: float = 3.2,
                 a_brake: float = 5.0, stop_gap: float = 4.5):
        self.t_brake = t_brake
        self.t_slow = t_slow
        self.a_brake = a_brake
        self.stop_gap = stop_gap

    # ------------------------------------------------------------------
    @staticmethod
    def _threat(ego: EgoState, tr: Track) -> Optional[Tuple[float, float]]:
        """(ttc, gap) for one track, or None if it is not in our way.

        Everything is already in the ego frame: +x forward, +y left, ego at
        the origin. So "ahead" is x > 0 and "in our corridor" is |y| small.
        """
        if tr.x <= 0.0:
            return None                      # behind us

        half_w = CORRIDOR_HALF_W + (VRU_EXTRA_M if tr.cls in VRU else 0.0)
        # Project the agent's own lateral motion forward a little: a bike
        # angling into the lane is a threat before it has arrived.
        y_soon = tr.y + tr.vy * 1.0
        if abs(tr.y) > half_w and abs(y_soon) > half_w:
            return None                      # stays out of our path

        gap = max(tr.x - 2.5, 0.0)           # bumper, not centre
        closing = ego.v - tr.vx              # +ve = we are catching up
        if closing <= 0.05:
            return (float("inf"), gap)       # not closing: no TTC
        return (gap / closing, gap)

    # ------------------------------------------------------------------
    def govern(self, ego: EgoState, tracks: List[Track],
               planned_accel: float) -> SafetyDecision:
        worst_ttc, worst_gap, worst_id = float("inf"), float("inf"), None
        for tr in tracks:
            hit = self._threat(ego, tr)
            if hit is None:
                continue
            ttc, gap = hit
            # Rank by TTC, but let a very close agent win on gap alone - a
            # stationary car 1 m ahead has infinite TTC once we have stopped,
            # and we must not then drive into it.
            if ttc < worst_ttc or (gap < worst_gap and gap < self.stop_gap):
                worst_ttc, worst_gap, worst_id = ttc, gap, tr.id

        if worst_id is None:
            return SafetyDecision(accel=planned_accel)

        t_brake = self.t_brake
        t_slow = self.t_slow
        threat_cls = next((t.cls for t in tracks if t.id == worst_id), None)
        if threat_cls in VRU:
            t_brake += VRU_EXTRA_TTC
            t_slow += VRU_EXTRA_TTC

        # Too close, full stop, and stay stopped until there is room.
        if worst_gap < self.stop_gap and ego.v < 1.0:
            return SafetyDecision(accel=-self.a_brake, ttc=worst_ttc,
                                  gap=worst_gap, reason="BRAKE",
                                  threat=worst_id)

        if worst_ttc <= t_brake:
            return SafetyDecision(accel=-self.a_brake, ttc=worst_ttc,
                                  gap=worst_gap, reason="BRAKE",
                                  threat=worst_id)

        if worst_ttc <= t_slow:
            # Ramp linearly from "no change" at t_slow to full braking at
            # t_brake. NEVER let this raise the planner's acceleration - the
            # governor's whole contract is that it can only slow the car.
            f = (t_slow - worst_ttc) / max(t_slow - t_brake, 1e-3)
            capped = planned_accel * (1.0 - f) + (-self.a_brake) * f
            return SafetyDecision(accel=min(planned_accel, capped),
                                  ttc=worst_ttc, gap=worst_gap,
                                  reason="SLOW", threat=worst_id)

        return SafetyDecision(accel=planned_accel, ttc=worst_ttc,
                              gap=worst_gap, threat=worst_id)


class StuckRecovery:
    """Reverse out when the car has driven into something and cannot proceed.

    WHY THIS IS NOT OPTIONAL FOR A LIVE DEMO.
    Measured, repeatedly: the ego clipped a pole, then sat at 0.1 km/h with
    throttle 0.50 for over 3000 frames - pushing against it forever. Avoidance
    can be excellent and one unlucky contact still ends the run, because
    nothing in the stack notices that commanded motion is not producing
    actual motion.

    The rule: if we are asking for forward acceleration and the car is not
    moving for `patience` seconds, reverse briefly, then hand control back.
    Deliberately dumb and time-boxed - a recovery behaviour that thinks too
    hard is another thing that can fail on stage.
    """

    def __init__(self, patience: float = 2.0, reverse_time: float = 1.5,
                 v_stuck: float = 0.4, blocked_patience: float = 8.0):
        self.patience = patience
        self.reverse_time = reverse_time
        self.v_stuck = v_stuck            # m/s below which we count as still
        # How long to keep yielding to a hazard before calling it a deadlock.
        # Long enough to let real traffic clear, short enough that a demo does
        # not die waiting for a parked cyclist.
        self.blocked_patience = blocked_patience
        self._stuck_since: Optional[float] = None
        self._blocked_since: Optional[float] = None
        self._reversing_until: Optional[float] = None
        self.events = 0

    def update(self, now: float, speed: float, wants_forward: bool,
               blocked: bool = False, at_light: bool = False) -> bool:
        """True if the caller should drive in REVERSE this frame.

        `blocked` means standing still is CORRECT right now - a red light, or
        something close in front. While it is set we never reverse, however
        long the car has been stationary.

        Without the `not blocked` branch this only caught one of the two ways
        to be stuck. The other, measured on seed 7, is a planner that settles
        on accel = 0 and stays there: 2,230 frames at 0 km/h with the throttle
        shut, route progress 0.0%, and nothing asking to go forward for
        `wants_forward` to notice. Being stopped for no reason is being stuck.
        """
        if self._reversing_until is not None:
            if now < self._reversing_until:
                return True
            self._reversing_until = None
            self._stuck_since = None
            self._blocked_since = None
            return False

        if speed >= self.v_stuck:                 # moving: nothing to recover
            self._stuck_since = None
            self._blocked_since = None
            return False

        # A RED LIGHT IS NOT A DEADLOCK. Waiting is the correct behaviour and
        # there is no time limit on it; reversing away from a signal would be
        # both wrong and alarming to watch.
        if at_light:
            self._stuck_since = None
            self._blocked_since = None
            return False

        # A HAZARD IN FRONT IS ONLY A REASON TO WAIT FOR SO LONG.
        #
        # This was the bug: `blocked` cleared the stuck timer on every frame,
        # so a hazard that never moved held the car at 0 km/h indefinitely.
        # Seen on demo day - a stationary cyclist 4.2 m ahead, planner in
        # EMERGENCY, route progress 0.0%, and nothing in the stack able to
        # notice that the wait had become permanent. Yielding is correct;
        # yielding forever is a deadlock, and a human driver would reverse out
        # and go round.
        if blocked:
            if self._blocked_since is None:
                self._blocked_since = now
            if now - self._blocked_since < self.blocked_patience:
                self._stuck_since = None
                return False
        else:
            self._blocked_since = None

        if self._stuck_since is None:
            self._stuck_since = now
        elif now - self._stuck_since >= self.patience:
            self._reversing_until = now + self.reverse_time
            self._blocked_since = None
            self.events += 1
            return True
        return False
