import copy
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
        intensity_half_life: float,
        antero_speed_distr: Callable[[], float],
        retro_speed_distr: Callable[[], float],
        antero_resample_prob: float,
        retro_resample_prob: float,
        velocity_noise_distr: Callable[[], float],
        transition_matrix: TransitionMatrixType,
        rng: np.random.Generator,
    ):
        self._antero_speed_distr = antero_speed_distr
        self._retro_speed_distr = retro_speed_distr
        self._antero_resample_prob = antero_resample_prob
        self._retro_resample_prob = retro_resample_prob
        self._velocity_noise_distr = velocity_noise_distr
        self._transition_matrix = transition_matrix

        self.state: MotionStateCollection = initial_state
        self.position: float = initial_position
        self.velocity: float = self._sample_velocity()

        self.initial_intensity = initial_intensity
        self.intensity = initial_intensity
        self.intensity_decay_rate = np.log(2) / intensity_half_life

        self.step_count = 0

        self.rng = rng

    def step(self):
        self.step_count += 1

        # update position
        self.position += self.velocity + self._velocity_noise_distr()
        # decay intensity
        self.intensity = self.initial_intensity * np.exp(
            -self.intensity_decay_rate * self.step_count
        )

        self._update_state()

        # probability of resampling velocity
        if self.state == MotionStateCollection.ANTEROGRADE:
            resampling_prob = self._antero_resample_prob
        elif self.state == MotionStateCollection.RETROGRADE:
            resampling_prob = self._retro_resample_prob
        else:
            resampling_prob = None

        if resampling_prob is not None:
            decision_prob = self.rng.random()
            if decision_prob <= resampling_prob:
                self.velocity = self._sample_velocity()

    def _update_state(self):
        new_state = self._decide_next_state()
        if new_state != self.state:
            self.state = new_state
            # resample velocity
            self.velocity = self._sample_velocity()

    def _decide_next_state(self) -> MotionStateCollection:
        transition_probs = self._transition_matrix[self.state]

        decision_prob = self.rng.random()
        cumulative_prob = 0
        for transition_state, prob in transition_probs.items():
            cumulative_prob += prob
            if decision_prob <= cumulative_prob:
                return transition_state
        raise RuntimeError(
            "Something went wrong in the probability calculation, this code should be "
            "unreachable."
        )

    def _sample_velocity(self) -> float:
        match self.state:
            case MotionStateCollection.STATIONARY:
                return 0
            case MotionStateCollection.ANTEROGRADE:
                return self._antero_speed_distr()  # TODO: make sure always pos
            case MotionStateCollection.RETROGRADE:
                return -self._retro_speed_distr()  # TODO: make sure always neg
            case _:
                raise ValueError(f"Unknown state '{self.state}'.")
