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


# TODO: this is so confusing ... just define original distribution in terms of
#   normal_std
def speed_variance_from_normal_std(
    speed_mode: PositiveFloat, normal_std: PositiveFloat
) -> PositiveFloat:
    normal_mean = np.log(speed_mode) + normal_std**2
    return (np.exp(normal_std**2) - 1) * np.exp(2 * normal_mean + normal_std**2)


def retro_var_default_factory(normal_std: PositiveFloat):
    def inner(data: dict[str, Any]) -> PositiveFloat:
        return speed_variance_from_normal_std(
            speed_mode=data["retro_mode"], normal_std=normal_std
        )

    return inner


def antero_var_default_factory(normal_std: PositiveFloat):
    def inner(data: dict[str, Any]) -> PositiveFloat:
        return speed_variance_from_normal_std(
            speed_mode=data["antero_mode"], normal_std=normal_std
        )

    return inner


def transition_matrix_default_factory(data: dict[str:Any]) -> TransitionMatrixType:
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
    """
    Parameters for the dynamics simulation of particles.

    Parameters
    ----------
    n_steps: int, optional
        The number of steps in the simulation. If unspecified, a random integer between
        64 and 128 will be chosen.
    time_delta: float, default=1.0
        The time difference between simulation steps, the units are seconds [s].
    path_length: float, optional
        The length of the path the particles will be observed moving along, the units
        are micrometers [um]. If unspecified, a random number between 10 and 20 will be
        chosen.
    particle_density: float, optional
        Average particle density, the units are particles per micrometer [#/um]. If
        unspecified, a random number between 0.1 and 1 will be chosen.
    retro_mode: float, optional
        Mode of particle speed in the retrograde direction, modeled by a log-normal
        distribution. Units are micrometers per second [um/s]. If unspecified, a
        random number between 0.2 and 1 is chosen.
    retro_var: float, optional
        Variance of particle speed in the retrograde direction, modeled by a log-normal
        distribution. Units are micrometers per second [um/s]. If not speciefied, the
        value will be calculated such that the underlying normal distribution has
        a standard deviation of 0.1.
    antero_mode: float, optional
        Mode of particle speed in the anterograde direction, modeled by a log-normal
        distribution. Units are micrometers per second [um/s]. If unspecified, a
        random number between 0.2 and 1 is chosen.
    antero_var: float, optional
        Variance of particle speed in the anterograde direction, modeled by a
        log-normal distribution. Units are micrometers per second [um/s]. If not
        speciefied, the value will be calculated such that the underlying normal
        distribution has a standard deviation of 0.1.
    noise_var: float, optional
        Variance of the velocity noise, modeled by a Gaussian distribution. Units are
        micrometers per second [um/s]. If unspecified a random number between 0 and
        0.4 will be chosen.
    fluorophores_per_particle_mode: float, optional
        Mode of the number of fluorophores per particle, modeled by a log-normal
        distribution. If unspecified, a random number between 200 and 400 will be
        chosen.
    fluorophores_per_particle_var: float, optional
        Variance of the number of fluorophores per particle, modeled by a log-normal
        distribution. If unspecified, a random number between 0 and 900 will be chosen.
    fluorophore_halflife_mode: float, optional
        Mode of the flurophore_halflife of each particle, modeled by a log-normal
        distribution.
    fluorophore_halflife_var: float, optional
        Variance of the flurophore_halflife of each particle, modeled by a log-normal
        distribution.
    particle_behaviour: {"unidirectional", "bidirectional"}, optional
        Label that can be used to specify the particle behaviour. This will determine 
        the construction of the transition matrix.
    transition_matrix: dict[MotionStateCollection, dict[MotionStateCollection, float]]
        Matrix containing the probabilities for a particle to transition from one state 
        to another. The first index denotes the particle's current state, and the 
        second denotes the state it can transition to.
    seed: int
        A seed for the dynamics simulation for reproducibility.
    """
    # TODO: 
    #   - fluorophore_halflife_mode default values docs
    #   - fluorophore_halflife_var default values docs

    model_config = ConfigDict(validate_assignment=True, validate_default=True)

    n_steps: PositiveInt = Field(default_factory=lambda: np.random.randint(64, 128))

    time_delta: PositiveFloat = 1.0

    path_length: PositiveFloat = Field(
        default_factory=lambda: np.random.uniform(10, 20)
    )

    particle_density: PositiveFloat = Field(
        default_factory=lambda: np.random.uniform(0.1, 1)
    )

    retro_mode: PositiveFloat = Field(default_factory=lambda: np.random.uniform(0.2, 1))

    retro_var: PositiveFloat = Field(
        default_factory=retro_var_default_factory(normal_std=0.1)
    )

    antero_mode: PositiveFloat = Field(
        default_factory=lambda: np.random.uniform(0.2, 1)
    )

    antero_var: PositiveFloat = Field(
        default_factory=antero_var_default_factory(normal_std=0.1)
    )

    noise_var: PositiveFloat = Field(default_factory=lambda: np.random.uniform(0, 0.4))

    fluorophores_per_particle_mode: PositiveFloat = Field(
        default_factory=lambda: np.random.uniform(300, 600)
    )

    fluorophores_per_particle_var: PositiveFloat = Field(
        default_factory=lambda: np.random.uniform(0, 45**2)
    )

    # TODO: probably not very realistic, particles should have the same bleaching rate
    fluorophore_halflife_mode: PositiveFloat = Field(
        default_factory=lambda data: np.random.uniform(
            data["n_steps"] * data["time_delta"] * 0.5,
            data["n_steps"] * data["time_delta"] * 1.5,
        )
    )

    fluorophore_halflife_var: PositiveFloat = Field(
        default_factory=lambda data: np.random.uniform(
            data["n_steps"] * data["time_delta"] * 0.5 / 10,
            data["n_steps"] * data["time_delta"] * 1.5 / 10,
        )
        ** 2
    )

    particle_behaviour: Literal["unidirectional", "bidirectional"] = Field(
        default_factory=lambda: np.random.choice(["unidirectional", "bidirectional"])
    )

    transition_matrix: TransitionMatrixType = Field(
        default_factory=transition_matrix_default_factory
    )


    seed: PositiveInt = Field(default_factory=lambda: np.random.randint(2**63))

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
