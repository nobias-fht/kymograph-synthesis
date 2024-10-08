from typing import Callable, Optional

import numpy as np

from .motion_state_collection import MotionStateCollection

TransitionMatrixType = dict[MotionStateCollection, dict[MotionStateCollection, float]]


class ParticleSimulator:

    def __init__(
        self,
        initial_position: float,
        initial_state: MotionStateCollection,
        initial_intensity: float,
        intensity_decay_rate: float,
        antero_speed_distr: Callable[[], float],
        retro_speed_distr: Callable[[], float],
        velocity_noise_distr: Callable[[], float],
        state_switch_prob: dict[MotionStateCollection, float],
        transition_prob_matrix: TransitionMatrixType,
    ):
        self._antero_speed_distr = antero_speed_distr
        self._retro_speed_distr = retro_speed_distr
        self._velocity_noise_distr = velocity_noise_distr
        self._state_switch_prob = state_switch_prob
        self._transition_prob_matrix = transition_prob_matrix

        self.state: MotionStateCollection = initial_state
        self.position: float = initial_position
        self.velocity: float = self.sample_velocity()

        self.initial_intensity = initial_intensity
        self.intensity = initial_intensity
        self.intensity_decay_rate = intensity_decay_rate

        self.step_count = 0

    def step(self):
        # update position
        self.position += self.velocity + self._velocity_noise_distr()
        # decay intensity
        self.intensity = self.initial_intensity * np.exp(
            -self.intensity_decay_rate * self.step_count
        )

        self.step_count += 1

        # decide if switching state
        decision_prob = np.random.random()
        if decision_prob > self._state_switch_prob[self.state]:
            # if not early return
            return

        # in case of switching state
        # update state
        self.state = self.next_state_transition()
        # resample velocity
        self.velocity = self.sample_velocity()

    def next_state_transition(self) -> MotionStateCollection:
        transition_probs = self._transition_prob_matrix[self.state]
        decision_prob = np.random.random()
        cumulative_prob = 0
        for transition_state, prob in transition_probs.items():
            cumulative_prob += prob
            if decision_prob <= cumulative_prob:
                return transition_state
        print("Should be unreachable")

    def sample_velocity(self) -> float:
        match self.state:
            case MotionStateCollection.STATIONARY:
                return 0
            case MotionStateCollection.ANTEROGRADE:
                return self._antero_speed_distr()  # TODO: make sure always pos
            case MotionStateCollection.RETROGRADE:
                return -self._retro_speed_distr()  # TODO: make sure always neg
            case _:
                raise ValueError(f"Unknown state '{self.state}'.")
