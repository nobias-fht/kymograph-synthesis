from math import isclose

from pydantic import (
    Field,
    BaseModel,
    ConfigDict,
    field_validator,
    PositiveInt,
    PositiveFloat,
    NonNegativeFloat
)
import numpy as np

from ..dynamics.particle_simulator.motion_state_collection import MotionStateCollection

TransitionMatrixType = dict[
    MotionStateCollection, dict[MotionStateCollection, NonNegativeFloat]
]


class DynamicsParams(BaseModel):

    model_config = ConfigDict(validate_assignment=True)

    n_steps: PositiveInt = Field(description="The number of simulation steps in the dynamics simulation.")
    """The number of simulation steps in the dynamics simulation."""

    particle_density: PositiveFloat
    """Average particle density. Units are number of particles per path."""

    retro_mode: PositiveFloat
    """
    Mode of particle speed in the retrograde direction, modeled by a log-normal 
    distribution. Units are percentage of path per simulation time step.
    """

    retro_var: PositiveFloat
    """
    Variance of particle speed in the retrograde direction, modeled by a log-normal 
    distribution. Units are percentage of path per simulation time step.
    """

    antero_mode: PositiveFloat
    """
    Mode of particle speed in the anterograde direction, modeled by a log-normal 
    distribution. Units are percentage of path per simulation time step.
    """

    antero_var: PositiveFloat
    """
    Variance of particle speed in the anterograde direction, modeled by a log-normal 
    distribution.
    """

    noise_var: PositiveFloat
    """
    Variance of the velocity noise, modeled by a Gaussian distribution. Units are 
    percentage of path per simulation time step.
    """

    intensity_mode: PositiveFloat
    """
    Mode of the particle intensities, modeled by a log-normal distribution.
    """

    intensity_var: PositiveFloat
    """Variance of the particle intensities, modeled by a log-normal distribution."""

    intensity_halflife_mode: PositiveFloat
    """Mode of particle intensity half-life, modeled by a log-normal distribution."""

    intensity_halflife_var: PositiveFloat
    """
    Variance of the paticle intensity half-life, modeled by a log-normal distribution.
    """

    transition_matrix: TransitionMatrixType
    """
    Matrix containing the probabilities for a particle to transition from one state to 
    another. First index denotes the particles current state and the second denotes the 
    state it can transition to.
    """

    seed: int = Field(default_factory=lambda: np.random.randint(2**63))

    @field_validator("transition_matrix")
    @classmethod
    def validate_transition_probabilities(
        cls, transition_matrix: TransitionMatrixType
    ) -> TransitionMatrixType:
        for state, transition_probs in transition_matrix.items():
            probability_sum = sum(transition_probs.values())
            if not isclose(probability_sum, 1):
                raise ValueError(
                    f"Transition probabilities from state '{state.value}' sum to "
                    f"{probability_sum}, probabilities must sum to 1."
                )
        return transition_matrix
