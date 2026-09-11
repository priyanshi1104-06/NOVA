"""
NOVA core data types.

THIS FILE IS THE MOST IMPORTANT ARCHITECTURAL DECISION IN THE PROJECT.

Everything downstream (prediction, risk map, planner) consumes ONLY the types
defined here. Nothing downstream knows whether the data came from CARLA, from
the 2D simulator, or from YOLOv8 running on a dashcam video.

That means:
  - You can build and prove the whole "brain" before CARLA is even installed.
  - If CARLA fails on demo day, you swap one adapter file and the demo still runs.
  - When you pitch, this is the slide: "our planner is sensor-agnostic."

COORDINATE FRAME (fix this in your head, it causes 90% of bugs):
  Ego-centric, right-handed, metres.
    +x = forward (direction the ego car is pointing)
    +y = LEFT
    heading = radians, 0 = straight ahead, positive = turning left (CCW)

  CARLA uses a LEFT-handed frame with +y = RIGHT. The CARLA adapter is the ONLY
  place allowed to know that; it negates y and yaw on the way in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np


class AgentClass(str, Enum):
    """Object classes NOVA reasons about.

    These are deliberately the Indian Driving Dataset (IDD) classes, not the
    COCO/nuScenes ones. AUTO_RICKSHAW, PUSHCART and ANIMAL are the three that
    Western AV stacks simply do not have, and they are the ones you point at
    when a judge asks "why not just use an existing planner?"
    """

    CAR = "car"
    BUS = "bus"
    TRUCK = "truck"
    AUTO_RICKSHAW = "autorickshaw"
    TWO_WHEELER = "motorcycle"
    BICYCLE = "bicycle"
    PEDESTRIAN = "person"
    PUSHCART = "pushcart"
    ANIMAL = "animal"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ClassProfile:
    """Per-class behavioural priors.

    This is where "Indian road conditions" stops being a slogan and becomes a
    number. A motorcycle in Ahmedabad traffic can change its lateral position
    far faster and far less predictably than a bus can. `erraticness` is the
    knob that encodes it, and it flows straight through into how wide the
    predicted trajectory fan opens in prediction.py.

    length, width  : metres, used for footprint inflation in the risk map
    max_speed      : m/s, caps how far a prediction can reach
    max_lat_accel  : m/s^2, how hard it can swerve -> how wide the fan opens
    erraticness    : 0..1, probability mass assigned to NON-straight manoeuvres
    """

    length: float
    width: float
    max_speed: float
    max_lat_accel: float
    erraticness: float


# Tuned from observation of Indian urban traffic, not from a paper.
# If a judge asks where these came from, the honest answer is "hand-tuned
# priors; the learned predictor replaces them" - that is a normal and
# respectable answer for a hackathon prototype.
CLASS_PROFILES: dict[AgentClass, ClassProfile] = {
    AgentClass.CAR:           ClassProfile(4.2, 1.8, 22.0, 3.0, 0.25),
    AgentClass.BUS:           ClassProfile(11.0, 2.6, 16.0, 1.5, 0.15),
    AgentClass.TRUCK:         ClassProfile(8.0, 2.5, 16.0, 1.5, 0.18),
    # The auto-rickshaw is the signature Indian agent: small, tuk-tuk turning
    # circle, and near-zero commitment to lane discipline.
    AgentClass.AUTO_RICKSHAW: ClassProfile(2.8, 1.4, 13.0, 4.0, 0.55),
    # Two-wheelers filter through gaps. Highest erraticness of any vehicle.
    AgentClass.TWO_WHEELER:   ClassProfile(1.9, 0.7, 20.0, 5.0, 0.65),
    AgentClass.BICYCLE:       ClassProfile(1.7, 0.6, 7.0, 3.0, 0.50),
    AgentClass.PEDESTRIAN:    ClassProfile(0.5, 0.5, 2.0, 4.0, 0.80),
    AgentClass.PUSHCART:      ClassProfile(2.0, 1.0, 2.0, 1.0, 0.40),
    AgentClass.ANIMAL:        ClassProfile(1.5, 0.6, 5.0, 5.0, 0.90),
    AgentClass.UNKNOWN:       ClassProfile(3.0, 1.5, 15.0, 3.0, 0.40),
}


@dataclass
class Track:
    """One tracked object at one instant, in the ego frame.

    A CARLA adapter fills this from ground-truth actor state.
    A vision adapter fills this from YOLOv8 + ByteTrack + a depth/IPM estimate.
    Neither one is visible to anything downstream.

    `history` is the last N observed (x, y) positions, oldest first. The
    predictor uses it to estimate yaw rate, which is what lets it tell
    "this bike is already leaning left" apart from "this bike is going straight".
    Without history you can only ever do constant-velocity, which is useless in
    exactly the merge/cut-in scenarios the problem statement is about.
    """

    id: int
    cls: AgentClass
    x: float
    y: float
    heading: float          # rad, ego frame
    vx: float               # m/s, ego frame
    vy: float               # m/s, ego frame
    history: List[tuple] = field(default_factory=list)
    confidence: float = 1.0
    # Seconds between consecutive `history` samples. The producer knows this
    # and nothing else does, so it travels with the track. yaw_rate() divides
    # by it; hand it the wrong value and every turn rate is scaled, which
    # silently rebalances the cut-left / cut-right split in prediction.py.
    dt: float = 0.1

    @property
    def profile(self) -> ClassProfile:
        return CLASS_PROFILES[self.cls]

    @property
    def speed(self) -> float:
        return float(np.hypot(self.vx, self.vy))

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)

    def yaw_rate(self, dt: Optional[float] = None) -> float:
        """Estimate turn rate (rad/s) from the position history.

        Uses the heading of the last two motion segments and differences them.
        Returns 0.0 when there is not enough history or the object is basically
        stationary (heading of a near-zero motion vector is pure noise).

        `dt` defaults to this track's own sampling interval. It used to default
        to a hardcoded 0.1 s, which was right only when the caller happened to
        observe at 10 Hz - step_b_observe.py observes every tick (0.05 s) and
        was therefore printing half the true yaw rate.
        """
        step = self.dt if dt is None else dt
        if len(self.history) < 3:
            return 0.0
        p = np.asarray(self.history[-3:], dtype=np.float64)
        d1 = p[1] - p[0]
        d2 = p[2] - p[1]
        if np.linalg.norm(d1) < 1e-3 or np.linalg.norm(d2) < 1e-3:
            return 0.0
        h1 = np.arctan2(d1[1], d1[0])
        h2 = np.arctan2(d2[1], d2[0])
        # wrap into [-pi, pi] so a heading crossing +/-pi does not explode
        return float(np.arctan2(np.sin(h2 - h1), np.cos(h2 - h1)) / step)


@dataclass
class TrajectoryMode:
    """ONE possible future for an object, with a probability attached.

    This is the heart of Module 2 and the thing that differentiates NOVA from a
    "CNN + A*" submission. A single-hypothesis predictor says "the bike will be
    HERE". A multimodal predictor says "60% here, 30% cut into my lane, 10%
    braking" - and the planner can then avoid the 30% branch without having to
    believe it will definitely happen.

    points : (T, 2) float array of predicted (x, y) in the ego frame
    sigma  : (T,) float array, positional 1-sigma uncertainty in metres.
             GROWS with t. This is what makes the risk map smear out into the
             future instead of drawing hard little dots.
    prob   : scalar in [0, 1]; modes for one track sum to 1.0
    label  : human-readable, printed on the debug overlay so you can point at
             the screen during the pitch and say what the system is thinking
    """

    points: np.ndarray
    sigma: np.ndarray
    prob: float
    label: str


@dataclass
class Prediction:
    """All hypothesised futures for one tracked object."""

    track_id: int
    cls: AgentClass
    modes: List[TrajectoryMode]

    def most_likely(self) -> TrajectoryMode:
        return max(self.modes, key=lambda m: m.prob)


@dataclass
class EgoState:
    """The ego vehicle. Always at the origin in its own frame, but we keep the
    fields explicit because the planner searches in this same frame and the
    metrics logger wants world coordinates too."""

    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    v: float = 0.0            # m/s, forward
    steer: float = 0.0        # rad, current front-wheel angle
    wheelbase: float = 2.7    # m
    length: float = 4.5
    width: float = 1.9


@dataclass
class Observation:
    """Everything NOVA knows at one timestep. The single input to the pipeline.

    Both sim2d.py and the CARLA adapter emit exactly this object, which is
    precisely why the pipeline does not need to change between them.

    drivable : optional (H, W) uint8 mask, 1 = drivable. From CARLA's semantic
               segmentation camera, or from DeepLabV3+ on real video. When it
               is None the risk map falls back to a geometric road model, so
               the pipeline still runs before segmentation is wired up.
    """

    t: float
    ego: EgoState
    tracks: List[Track]
    drivable: Optional[np.ndarray] = None
    goal: Optional[np.ndarray] = None   # (2,) target point in ego frame
