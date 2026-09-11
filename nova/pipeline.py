"""
NOVA pipeline - the glue that ties Modules 2/3/4 into one call.

This is the ONLY object CARLA or sim2d needs to talk to:

    pipeline = NovaPipeline()
    plan = pipeline.step(observation)     # observation -> steering + throttle

Keeping the glue in its own file (rather than inside the CARLA script) is what
lets you demo the identical brain in two front-ends. When you are asked "how
much of this is CARLA-specific?", the answer is: one adapter file, zero lines
in here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .planner import HybridAStarPlanner, Plan
from .prediction import ManeuverPredictor
from .riskmap import RiskMap
from .safety import SafetyDecision, SafetyGovernor
from .types import Observation, Prediction


@dataclass
class StepResult:
    plan: Plan
    predictions: List[Prediction] = field(default_factory=list)
    t_predict_ms: float = 0.0
    t_risk_ms: float = 0.0
    t_plan_ms: float = 0.0
    safety: Optional[SafetyDecision] = None

    @property
    def t_total_ms(self) -> float:
        return self.t_predict_ms + self.t_risk_ms + self.t_plan_ms


class NovaPipeline:
    """Perception-agnostic autonomy core: predict -> risk -> plan."""

    def __init__(
        self,
        horizon: float = 3.0,
        dt: float = 0.25,
        v_target: float = 8.0,
        predictor=None,
        **planner_kwargs,
    ):
        # horizon and dt MUST match across all three modules or the planner
        # will index the wrong risk slice - a silent, vicious bug that shows up
        # as "the car avoids things slightly too early or too late".
        self.predictor = predictor or ManeuverPredictor(horizon=horizon, dt=dt)
        self.risk = RiskMap(horizon=horizon, dt=dt)
        self.planner = HybridAStarPlanner(
            self.risk, dt=dt, horizon=horizon, v_target=v_target, **planner_kwargs
        )
        # LAST stage, and it can only ever slow the car. See safety.py for
        # why this is not folded into the planner's cost function.
        self.safety = SafetyGovernor()
        self.history: List[StepResult] = []

    def step(self, obs: Observation) -> StepResult:
        t0 = time.perf_counter()
        predictions = self.predictor.predict(obs.tracks)
        t1 = time.perf_counter()
        self.risk.build(obs, predictions)
        self.risk.add_static_obstacles(
            getattr(obs, 'static_obstacles', None))
        t2 = time.perf_counter()

        goal = obs.goal if obs.goal is not None else np.array([40.0, 0.0])
        plan = self.planner.plan(obs.ego, goal)

        # Collision avoidance by BRAKING, not just by steering. The planner
        # avoids hazards by routing around them; if there is no way around, it
        # used to keep its comfortable acceleration right up until the search
        # failed outright. This caps it on time-to-collision.
        decision = self.safety.govern(obs.ego, obs.tracks, plan.accel)
        plan.accel = decision.accel
        if decision.reason == "BRAKE":
            plan.emergency = True
        t3 = time.perf_counter()

        res = StepResult(
            plan=plan,
            predictions=predictions,
            t_predict_ms=(t1 - t0) * 1000,
            t_risk_ms=(t2 - t1) * 1000,
            t_plan_ms=(t3 - t2) * 1000,
            safety=decision,
        )
        self.history.append(res)
        return res
