from ..params import Params

from .simulate_dynamics import simulate_dynamics, DynamicsSimOutput
from .simulate_imaging import simulate_imaging, ImagingSimOutput
from .sample_kymograph import sample_kymograph, SampleKymographOutput
from .generate_ground_truth import generate_ground_truth, GenerateGroundTruthOutput


class Pipeline:

    def __init__(self, params: Params):
        self.params = params
        self.simulate_dynamics_output: DynamicsSimOutput
        self.simulate_imaging_output: ImagingSimOutput
        self.sample_kymograph_output: SampleKymographOutput
        self.generate_ground_truth_output: GenerateGroundTruthOutput

    def run(self):
        self.simulate_dynamics_output = simulate_dynamics(self.params.dynamics)
        self.simulate_imaging_output = simulate_imaging(
            self.params.rendering,
            n_steps=self.simulate_dynamics_output.n_steps,
            particle_positions=self.simulate_dynamics_output.particle_positions,
            particle_fluorophore_count=self.simulate_dynamics_output.particle_fluorophore_count,
        )
        self.sample_kymograph_output = sample_kymograph(
            self.params.kymograph, frames=self.simulate_imaging_output.frames
        )
        self.generate_ground_truth_output = generate_ground_truth(
            particle_positions=self.simulate_dynamics_output.particle_positions,
            particle_states=self.simulate_dynamics_output.particle_states,
            n_spatial_values=self.sample_kymograph_output.n_spatial_values
        )

    def save(self):
        # TODO: save params and outputs
        ...
