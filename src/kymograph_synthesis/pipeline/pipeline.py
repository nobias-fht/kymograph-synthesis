from typing import Optional, cast, Mapping
from pathlib import Path
import yaml
from PIL import Image
from glob import glob

import numpy as np
from numpy.typing import NDArray, ArrayLike
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

from ..params import Params
from .steps.simulate_dynamics import simulate_dynamics, DynamicsSimOutput
from .steps.simulate_imaging import simulate_imaging, ImagingSimOutput
from .steps.sample_kymograph import sample_kymograph, SampleKymographOutput
from .steps.generate_ground_truth import (
    generate_ground_truth,
    GenerateGroundTruthOutput,
)
from .write_log import WriteLogManager, PipelineFilenames


class Pipeline:

    def __init__(
        self,
        params: Params,
        out_dir: Path,
        output_id: Optional[str] = None,
        output_filenames: Optional[PipelineFilenames] = None,
    ):
        self.params = params
        self.out_dir = out_dir
        self.write_log_manager = WriteLogManager(
            out_dir=out_dir, pipeline_filenames=output_filenames
        )
        if output_id is None:
            self.output_id = self.write_log_manager.create_new_id()
        else:
            self.output_id = self.output_id

        # These will be created when the pipeline is run
        self.dynamics_sim_output: Optional[DynamicsSimOutput] = None
        self.imaging_sim_output: Optional[ImagingSimOutput] = None
        self.sample_kymograph_output: Optional[SampleKymographOutput] = None
        self.generate_ground_truth_output: Optional[GenerateGroundTruthOutput] = None

    def run(self):
        self.dynamics_sim_output = simulate_dynamics(self.params.dynamics)
        # (alias to avoid too long line)
        particle_fluorophore_count = self.dynamics_sim_output[
            "particle_fluorophore_count"
        ]
        self.imaging_sim_output = simulate_imaging(
            self.params.rendering,
            n_steps=self.dynamics_sim_output["n_steps"],
            particle_positions=self.dynamics_sim_output["particle_positions"],
            particle_fluorophore_count=particle_fluorophore_count,
        )
        self.sample_kymograph_output = sample_kymograph(
            self.params.kymograph, frames=self.imaging_sim_output["frames"]
        )
        self.generate_ground_truth_output = generate_ground_truth(
            particle_positions=self.dynamics_sim_output["particle_positions"],
            particle_states=self.dynamics_sim_output["particle_states"],
            n_spatial_values=self.sample_kymograph_output["n_spatial_values"],
        )

    def save(self, save_visualization: bool = True):
        self._save_params()
        self._save_outputs()
        if save_visualization:
            self._save_visualization()
        self.write_log_manager.add_output_id(self.output_id)
        self.write_log_manager.log()

    def _save_params(self):
        fname = self.write_log_manager.write_log.pipeline_filenames.params.file_name(
            self.output_id
        )
        with open(self.out_dir / fname, "w") as f:
            yaml.dump(self.params.model_dump(mode="json"), f, sort_keys=False)

    def _save_outputs(self):
        if (
            (self.dynamics_sim_output is None)
            or (self.imaging_sim_output is None)
            or (self.sample_kymograph_output is None)
            or (self.generate_ground_truth_output is None)
        ):
            raise ValueError(
                "Outputs are None. Pipeline needs to be run before it can be saved."
            )
        pipeline_filenames = self.write_log_manager.write_log.pipeline_filenames
        dynamics_sim_output_fname = pipeline_filenames.dynamics_sim_output.file_name(
            self.output_id
        )
        imaging_sim_output_fname = pipeline_filenames.imaging_sim_output.file_name(
            output_id=self.output_id
        )
        sample_kymograph_output_fname = (
            pipeline_filenames.sample_kymograph_output
        ).file_name(output_id=self.output_id)

        generate_ground_truth_output_fname = (
            pipeline_filenames.generate_ground_truth_output
        ).file_name(output_id=self.output_id)

        np.savez(
            self.out_dir / dynamics_sim_output_fname,
            **cast(Mapping[str, ArrayLike], self.dynamics_sim_output),
        )
        np.savez(
            self.out_dir / imaging_sim_output_fname,
            **cast(Mapping[str, ArrayLike], self.imaging_sim_output),
        )
        np.savez(
            self.out_dir / sample_kymograph_output_fname,
            **cast(Mapping[str, ArrayLike], self.sample_kymograph_output),
        )
        np.savez(
            self.out_dir / generate_ground_truth_output_fname,
            **cast(Mapping[str, ArrayLike], self.generate_ground_truth_output),
        )

    def _save_visualization(self):
        self._save_kymograph_png()
        self._save_kymograph_gt_png()
        self._save_animation_gif()

    def _save_animation_gif(self):
        if self.imaging_sim_output is None:
            raise RuntimeError(
                "Imaging sim output has to be generated before animation gif can be "
                "saved."
            )
        pipeline_filenames = self.write_log_manager.write_log.pipeline_filenames
        animation_2d_visual_fname = pipeline_filenames.animation_2d_visual.file_name(
            output_id=self.output_id
        )
        file_path = self.out_dir / animation_2d_visual_fname

        z_index = self.params.rendering.imaging.output_space.shape[0] // 2
        raw_frames = self.imaging_sim_output["frames"][:, z_index]
        norm = Normalize(vmin=raw_frames.min(), vmax=raw_frames.max())
        cmap = cm.get_cmap("gray")
        visual_frames: NDArray[np.float_] = cmap(norm(raw_frames))
        images = [
            Image.fromarray(
                (_resize_image(frame, factor=4) * 255).astype(np.uint8), mode="RGBA"
            )
            for frame in visual_frames
        ]
        images[0].save(
            file_path,
            save_all=True,
            append_images=images[1:],
            optimize=False,
            duration=0.2,
            loop=0,
        )

    def _save_kymograph_png(self):
        if self.sample_kymograph_output is None:
            raise RuntimeError(
                "Cannot save kymograph output before kymograph sampling step has "
                "happend."
            )

        pipeline_filenames = self.write_log_manager.write_log.pipeline_filenames
        kymograph_visual_fname = pipeline_filenames.kymograph_visual.file_name(
            output_id=self.output_id
        )
        file_path = self.out_dir / kymograph_visual_fname

        kymograph_raw = self.sample_kymograph_output["kymograph"]
        kymograph_resized = _resize_image(kymograph_raw, factor=4)
        plt.imsave(
            file_path,
            np.squeeze(kymograph_resized),
            cmap="gray",
        )

    def _save_kymograph_gt_png(self):
        if self.generate_ground_truth_output is None:
            raise RuntimeError(
                "Cannot save kymograph ground truth before the ground truth has been "
                "generated."
            )
        pipeline_filenames = self.write_log_manager.write_log.pipeline_filenames
        kymograph_gt_visual_fname = pipeline_filenames.kymograph_gt_visual.file_name(
            output_id=self.output_id
        )
        file_path = self.out_dir / kymograph_gt_visual_fname
        kymograph_gt_raw = self.generate_ground_truth_output["ground_truth"]
        kymograph_gt_resized = _resize_image(kymograph_gt_raw, factor=4)
        plt.imsave(
            file_path,
            np.squeeze(kymograph_gt_resized),
            cmap="gray",
        )


def _resize_image(image: NDArray, factor: int) -> NDArray:
    return np.repeat(
        np.repeat(
            image,
            repeats=factor,
            axis=0,
        ),
        repeats=factor,
        axis=1,
    )
