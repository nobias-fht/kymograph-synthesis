from typing import Any, Literal
from typing_extensions import Self
from math import isclose

from pydantic import (
    Field,
    BaseModel,
    ConfigDict,
    field_validator,
    model_validator,
    PositiveInt,
    PositiveFloat,
    NonNegativeFloat,
)
import numpy as np

from ..dynamics.particle_simulator.motion_state_collection import MotionStateCollection

TransitionMatrixType = dict[
    MotionStateCollection, dict[MotionStateCollection, NonNegativeFloat]
]


def speed_variance_from_normal_std(
    speed_mode: PositiveFloat, normal_std: PositiveFloat
) -> PositiveFloat:
    normal_mean = np.log(speed_mode) + normal_std**2
    return (np.exp(normal_std**2) - 1) * np.exp(2 * normal_mean + normal_std**2)


def retro_var_default_factory(normal_std: PositiveFloat):
    def inner(data: dict[str, Any]) -> PositiveFloat:
        return speed_variance_from_normal_std(
            speed_mode=data["retro_speed_mode"], normal_std=normal_std
        )

    return inner


def antero_var_default_factory(normal_std: PositiveFloat):
    def inner(data: dict[str, Any]) -> PositiveFloat:
        return speed_variance_from_normal_std(
            speed_mode=data["antero_speed_mode"], normal_std=normal_std
        )

    return inner


def transition_matrix_default_factory(data: dict[str, Any]) -> TransitionMatrixType:
    values = np.random.rand(3, 3)
    # unidirectional particles do not transition
    if data["particle_behaviour"] == "unidirectional":
        values[(0, 0, 1, 2, 2), (1, 2, 1, 0, 1)] = 0
    # normalise to sum to 1
    values_normed = np.divide(
        values,
        values.sum(axis=1).reshape(-1, 1),
        out=np.zeros_like(values),
        where=values != 0,
    )
    states = [
        MotionStateCollection.ANTEROGRADE,
        MotionStateCollection.STATIONARY,
        MotionStateCollection.RETROGRADE,
    ]
    transition_matrix = {}
    for i, from_state in enumerate(states):
        transition_matrix[from_state] = {}
        for j, to_state in enumerate(states):
            transition_matrix[from_state][to_state] = values_normed[i, j]
    return transition_matrix


class DynamicsParams(BaseModel):

    model_config = ConfigDict(validate_assignment=True, validate_default=True)

    particle_density: PositiveFloat = Field(
        default_factory=lambda: np.random.uniform(1, 12)
    )
    """Average particle density. Units are number of particles per path."""

    retro_speed_mode: PositiveFloat = Field(
        default_factory=lambda: np.random.uniform(0.5, 6)
    )
    """
    Mode of particle speed in the retrograde direction, modeled by a log-normal 
    distribution. Units are percentage of path per simulation time step.
    """

    retro_speed_var: PositiveFloat = Field(
        default_factory=retro_var_default_factory(normal_std=0.1)
    )
    """
    Variance of particle speed in the retrograde direction, modeled by a log-normal 
    distribution. Units are percentage of path per simulation time step.
    """

    antero_speed_mode: PositiveFloat = Field(
        default_factory=lambda: np.random.uniform(0.5, 6)
    )
    """
    Mode of particle speed in the anterograde direction, modeled by a log-normal 
    distribution. Units are percentage of path per simulation time step.
    """

    antero_speed_var: PositiveFloat = Field(
        default_factory=antero_var_default_factory(normal_std=0.1)
    )
    """
    Variance of particle speed in the anterograde direction, modeled by a log-normal 
    distribution.
    """

    retro_resample_prob: float = 0
    """
    Probability per time step that the speed of the particle will be resampled while
    moving in the retrograde direction.
    """

    antero_resample_prob: float = 0
    """
    Probability per time step that the speed of the particle will be resampled while
    moving in the anterograde direction.
    """    

    velocity_noise_var: PositiveFloat = Field(
        default_factory=lambda: np.random.uniform(0, 0.4)
    )
    """
    Variance of the velocity noise, modeled by a Gaussian distribution. Units are 
    percentage of path per simulation time step.
    """

    fluorophore_count_mode: PositiveFloat = Field(
        default_factory=lambda: np.random.uniform(200, 400)
    )
    """
    Mode of the particle intensities, modeled by a log-normal distribution.
    """

    fluorophore_count_var: PositiveFloat = Field(
        default_factory=lambda: np.random.uniform(0, 30**2)
    )
    """Variance of the particle intensities, modeled by a log-normal distribution."""

    fluorophore_halflife_mode: PositiveFloat
    """Mode of particle intensity half-life, modeled by a log-normal distribution."""

    fluorophore_halflife_var: PositiveFloat
    """
    Variance of the paticle intensity half-life, modeled by a log-normal distribution.
    """

    particle_behaviour: Literal["unidirectional", "bidirectional"] = Field(
        default_factory=lambda: np.random.choice(["unidirectional", "bidirectional"])
    )

    # TODO: generate transition matrix from unidirectional label
    transition_matrix: TransitionMatrixType = Field(
        default_factory=transition_matrix_default_factory
    )
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

    @model_validator(mode="after")
    def validate_particle_behaviour(self) -> Self:
        states = [
            MotionStateCollection.ANTEROGRADE,
            MotionStateCollection.STATIONARY,
            MotionStateCollection.RETROGRADE,
        ]
        transition_matrix_array = np.array(
            [
                [self.transition_matrix[from_state][to_state] for to_state in states]
                for from_state in states
            ]
        )
        if (self.particle_behaviour == "unidirectional") and (
            transition_matrix_array[(0, 0, 1, 2, 2), (1, 2, 1, 0, 1)] != 0
        ).all():
            raise ValueError(
                "Particle behviour label 'unidirectional' does not match transition "
                f"matrix: \n{transition_matrix_array}"
            )
        return self
