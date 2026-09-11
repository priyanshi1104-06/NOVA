"""
NOVA Module 2 - Multimodal Intent Prediction.

THIS IS YOUR NOVELTY MODULE. Know it cold; it is what you will be questioned on.

The one-line pitch
------------------
"We don't predict where a vehicle WILL go. We predict every place it COULD
plausibly go in the next 3 seconds, with a probability on each, and we plan
against the whole distribution."

Why that matters for the problem statement
------------------------------------------
MathWorks' background paragraph complains about vehicles that change direction
suddenly, merge without signalling, and cross at unmarked spots. Every one of
those is a case where a single-hypothesis predictor is CONFIDENTLY WRONG.
Constant-velocity says the auto-rickshaw continues straight; it cuts across
your bonnet; you brake late. Multimodal prediction keeps the "cuts across"
branch alive at 30% probability the whole time, so the planner has already
been leaving room for it.

The manoeuvre-hypothesis approach used here
-------------------------------------------
For each tracked object we instantiate a small set of candidate manoeuvres
(straight / cut-left / cut-right / brake, plus cross for VRUs), roll each one
forward with a kinematic model, and assign probabilities from three signals:

  1. Class prior      - a motorcycle is inherently more erratic than a bus
  2. Observed yaw rate - already leaning left => boost the left-cut branch
  3. Interaction      - an agent near the ego's path is likelier to react

This is a CLASSICAL predictor, and that is a deliberate choice, not a
shortcut. It runs at 1000+ Hz, needs no training data, and never silently
fails on an out-of-distribution scene. The learned predictor
(LSTMPredictor, below) implements the IDENTICAL interface, so upgrading is a
one-line swap and you always have a working fallback.

Be honest about this if asked. "Heuristic priors with a learned model behind
the same interface" is a strong engineering answer. Claiming the LSTM is
trained when it isn't is how you lose the round.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .types import AgentClass, Prediction, Track, TrajectoryMode


class ManeuverPredictor:
    """Generates a probability-weighted fan of futures for each tracked object.

    Parameters
    ----------
    horizon : float
        How many seconds into the future to predict. 3.0s is the sweet spot:
        long enough to react to a cut-in at urban speeds, short enough that the
        uncertainty cone has not swallowed the whole road.
    dt : float
        Timestep of the predicted trajectory. Must match the risk map's time
        slicing so the planner can index directly without interpolating.
    """

    def __init__(self, horizon: float = 3.0, dt: float = 0.25):
        self.horizon = horizon
        self.dt = dt
        self.n_steps = int(round(horizon / dt))
        # Precompute the time vector once; this runs every frame for every agent.
        self.tvec = np.arange(1, self.n_steps + 1, dtype=np.float64) * dt

    # ------------------------------------------------------------------
    # public API - the CARLA adapter and sim2d both call exactly this
    # ------------------------------------------------------------------
    def predict(self, tracks: Sequence[Track]) -> List[Prediction]:
        return [self._predict_one(tr) for tr in tracks]

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _predict_one(self, tr: Track) -> Prediction:
        prof = tr.profile
        speed = tr.speed
        yaw_rate = tr.yaw_rate()

        # Heading to propagate along. For a near-stationary object the velocity
        # vector is noise, so fall back to the tracked heading.
        if speed > 0.3:
            heading = float(np.arctan2(tr.vy, tr.vx))
        else:
            heading = tr.heading

        modes = self._build_modes(tr, prof, speed, yaw_rate, heading)

        # Normalise so probabilities sum to exactly 1.0. The planner treats
        # these as weights on an expectation; if they don't sum to 1 the risk
        # field silently changes scale between frames and your cost tuning
        # stops meaning anything.
        total = sum(m.prob for m in modes)
        if total > 0:
            for m in modes:
                m.prob /= total

        return Prediction(track_id=tr.id, cls=tr.cls, modes=modes)

    def _build_modes(
        self, tr: Track, prof, speed: float, yaw_rate: float, heading: float
    ) -> List[TrajectoryMode]:
        e = prof.erraticness

        # --- probability mass allocation -------------------------------
        # Straight gets (1 - erraticness); the rest is split between the two
        # lateral manoeuvres and braking.
        p_straight = 1.0 - e
        p_lateral_total = e * 0.7
        p_brake = e * 0.3

        # Yaw-rate evidence: an object already rotating is far likelier to
        # continue that way. tanh keeps the bias bounded in [-1, 1] and
        # saturates around 0.5 rad/s, roughly a decisive lane change.
        bias = float(np.tanh(yaw_rate / 0.5))
        p_left = p_lateral_total * (0.5 + 0.5 * bias)
        p_right = p_lateral_total * (0.5 - 0.5 * bias)

        # A strongly turning object should not keep a big "straight" branch.
        p_straight *= (1.0 - 0.6 * abs(bias))

        modes: List[TrajectoryMode] = []

        # --- 1. CONTINUE (constant turn rate, constant velocity) --------
        # Named 'continue', not 'straight': if the object is mid-turn this
        # mode follows that turn. Calling it 'straight' on the debug overlay
        # confused the very first test run - the label said straight while
        # the path curved 30 degrees, which looks like a bug to a judge.
        modes.append(
            self._roll(
                tr, heading, speed, lat_offset=0.0, accel=0.0,
                yaw_rate=yaw_rate * 0.5, prof=prof,
                prob=p_straight, label="continue",
            )
        )

        # --- 2/3. LATERAL CUT-IN, both directions ----------------------
        # Target lateral displacement is capped by what the object could
        # physically achieve: d = 0.5 * a_lat * T^2, limited to ~one lane.
        max_lat = min(3.5, 0.5 * prof.max_lat_accel * self.horizon ** 2)
        if speed > 0.5:
            modes.append(
                self._roll(
                    tr, heading, speed, lat_offset=+max_lat, accel=0.0,
                    yaw_rate=0.0, prof=prof,
                    prob=p_left, label="cut-left",
                )
            )
            modes.append(
                self._roll(
                    tr, heading, speed, lat_offset=-max_lat, accel=0.0,
                    yaw_rate=0.0, prof=prof,
                    prob=p_right, label="cut-right",
                )
            )

        # --- 4. BRAKE / stop -------------------------------------------
        # Sudden unsignalled stops are endemic in Indian traffic (autos halting
        # for a fare mid-lane). Deceleration clamped so the object never
        # reverses through the trajectory.
        if speed > 1.0:
            decel = -min(6.0, speed / self.horizon * 2.0)
            modes.append(
                self._roll(
                    tr, heading, speed, lat_offset=0.0, accel=decel,
                    yaw_rate=0.0, prof=prof,
                    prob=p_brake, label="brake",
                )
            )

        # --- 5. CROSS (vulnerable road users only) ---------------------
        # A pedestrian, cyclist or animal at an unmarked crossing point is the
        # single highest-consequence event in the problem statement. We give
        # them a dedicated perpendicular branch that does NOT depend on their
        # current velocity, because a stationary pedestrian at the kerb is
        # exactly the one who steps out.
        if tr.cls in (AgentClass.PEDESTRIAN, AgentClass.ANIMAL, AgentClass.BICYCLE):
            # Cross towards the ego centreline - i.e. into our path.
            direction = -np.sign(tr.y) if abs(tr.y) > 0.1 else 1.0
            cross_speed = min(prof.max_speed, 1.5)
            modes.append(
                self._roll(
                    tr, heading=float(np.arctan2(direction, 0.0)),
                    speed=cross_speed, lat_offset=0.0, accel=0.0,
                    yaw_rate=0.0, prof=prof,
                    prob=0.35 if tr.cls != AgentClass.ANIMAL else 0.5,
                    label="cross",
                )
            )

        return modes

    def _roll(
        self, tr: Track, heading: float, speed: float, lat_offset: float,
        accel: float, yaw_rate: float, prof, prob: float, label: str,
    ) -> TrajectoryMode:
        """Roll one manoeuvre hypothesis forward into an (T,2) trajectory.

        Longitudinal: s(t) = v*t + 0.5*a*t^2, clamped so braking stops at zero
        rather than reversing.
        Lateral: a smoothstep profile from 0 to `lat_offset`. Smoothstep
        (3u^2 - 2u^3) has zero derivative at both ends, so the manoeuvre starts
        and finishes gently - real vehicles do not teleport sideways, and a
        hard ramp produces a visibly fake-looking prediction fan.
        """
        t = self.tvec

        # longitudinal distance travelled, monotonic non-decreasing
        s = speed * t + 0.5 * accel * t ** 2
        if accel < 0:
            t_stop = speed / abs(accel)
            s_max = speed * t_stop + 0.5 * accel * t_stop ** 2
            s = np.minimum(s, s_max)
        s = np.maximum.accumulate(np.maximum(s, 0.0))

        # lateral offset via smoothstep
        u = t / self.horizon
        d = lat_offset * (3 * u ** 2 - 2 * u ** 3)

        # Curvature from residual yaw rate (the "continues its turn" case).
        #
        # TWO SAFEGUARDS, both needed. A naive theta = heading + yaw_rate * t
        # is unstable: an estimated 1.0 rad/s held for 3 s rotates the object
        # 172 degrees and flings the predicted path tens of metres sideways.
        # test_modules.py caught exactly that - a "straight" mode ending 30 m
        # off the road.
        #
        #   (a) PHYSICAL CLAMP. A turn rate implies lateral acceleration
        #       a_lat = v * yaw_rate. No real vehicle exceeds its tyre limit,
        #       so clamp |yaw_rate| <= max_lat_accel / v.
        #   (b) EXPONENTIAL DECAY. Vehicles straighten out; they do not hold a
        #       turn rate forever. Integrating a rate that decays with time
        #       constant tau gives a TOTAL heading change bounded by
        #       yaw_rate * tau, instead of growing without limit.
        if abs(yaw_rate) > 1e-4:
            if speed > 0.5:
                yr_max = prof.max_lat_accel / max(speed, 1.0)
                yaw_rate = float(np.clip(yaw_rate, -yr_max, yr_max))
            tau = 1.0                      # seconds; turn decays over ~1 s
            theta = heading + yaw_rate * tau * (1.0 - np.exp(-t / tau))
        else:
            theta = np.full_like(t, heading)

        # Compose: advance along theta, then displace perpendicular to it.
        # Perpendicular to (cos, sin) in a +y-is-left frame is (-sin, cos).
        px = tr.x + s * np.cos(theta) - d * np.sin(theta)
        py = tr.y + s * np.sin(theta) + d * np.cos(theta)
        points = np.stack([px, py], axis=1)

        # Uncertainty growth. Superlinear (t^1.5) because error compounds:
        # a small heading error early becomes a large position error late.
        # Scaled by class erraticness, so the cone around a motorcycle is
        # visibly fatter than the cone around a bus. This is the term that
        # makes the risk map look intelligent.
        sigma0 = 0.3 + 0.5 * prof.width
        sigma = sigma0 + 0.8 * prof.erraticness * np.power(t, 1.5)

        return TrajectoryMode(
            points=points,
            sigma=sigma,
            prob=float(max(prob, 1e-6)),
            label=label,
        )


class LSTMPredictor:
    """Drop-in learned replacement for ManeuverPredictor.

    DELIBERATELY A STUB. Wire this up ONLY if you are ahead of schedule on
    Tuesday. It implements the same `.predict(tracks) -> List[Prediction]`
    signature, so switching is one line in pipeline.py and switching BACK
    during the demo is also one line.

    Training recipe, if you get there:
      1. Run the CARLA scenario suite with logging on; dump every agent's
         (x, y, heading, v) at 10 Hz into a parquet file.
      2. Window it: 2s of history (20 frames) -> 3s of future (30 frames).
      3. Model: 2-layer LSTM, hidden 128, plus a class embedding. Output
         K=5 modes, each (30, 2) plus a logit. Train with the standard
         "winner-takes-all" multimodal loss: backprop the L2 error of the
         BEST mode only, and cross-entropy on the mode logits. WTA is what
         stops all K modes collapsing onto the same average trajectory,
         which is the classic failure and would silently destroy your novelty.
      4. Export to TorchScript, load here, keep ManeuverPredictor as fallback.
    """

    def __init__(self, checkpoint: str | None = None, horizon: float = 3.0, dt: float = 0.25):
        self.fallback = ManeuverPredictor(horizon=horizon, dt=dt)
        self.model = None
        if checkpoint:
            raise NotImplementedError(
                "LSTM checkpoint loading not implemented. This is intentional - "
                "use ManeuverPredictor until you have trained weights."
            )

    def predict(self, tracks: Sequence[Track]) -> List[Prediction]:
        return self.fallback.predict(tracks)
