from typing import Optional
from pathlib import Path
import yaml
from PIL import Image
from glob import glob

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
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

    def __init__(self, params: Params, output_id: Optional[str] = None):
        self.output_id = output_id
        self.params = params
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

    def save(
        self,
        out_dir: Path,
        output_filenames: Optional[PipelineFilenames] = None,
        save_visualization: bool = True,
    ):
        write_log_manager = WriteLogManager(
            out_dir=out_dir, pipeline_filenames=output_filenames
        )

        if self.output_id is None:
            output_id = write_log_manager.create_new_id()
        else:
            output_id = self.output_id
        self._save_params(out_dir=out_dir, output_id=output_id)
        self._save_outputs(out_dir=out_dir, output_id=output_id)
        if save_visualization:
            self._save_visualization(out_dir=out_dir, output_id=output_id)

    def _save_params(self, out_dir: Path, output_id: str):
        with open(out_dir / f"params_{output_id}.yaml", "w") as f:
            yaml.dump(self.params.model_dump(mode="json"), f, sort_keys=False)

    def _save_outputs(self, out_dir: Path, output_id: str):
        if (
            (self.dynamics_sim_output is None)
            or (self.imaging_sim_output is None)
            or (self.sample_kymograph_output is None)
            or (self.generate_ground_truth_output is None)
        ):
            raise ValueError(
                "Outputs are None. Pipeline needs to be run before it can be saved."
            )
        np.savez(
            out_dir / f"dynamics_sim_output_{output_id}",
            **self.dynamics_sim_output,
        )
        np.savez(
            out_dir / f"imaging_sim_output_{output_id}",
            **self.imaging_sim_output,
        )
        np.savez(
            out_dir / f"sample_kymograph_output_{output_id}",
            **self.sample_kymograph_output,
        )
        np.savez(
            out_dir / f"generate_ground_truth_output_{output_id}",
            **self.generate_ground_truth_output,
        )

    def _save_visualization(self, out_dir: Path, output_id: str):
        self._save_kymograph_png(out_dir=out_dir, output_id=output_id)
        self._save_kymograph_gt_png(out_dir=out_dir, output_id=output_id)
        self._save_animation_gif(out_dir=out_dir, output_id=output_id)

    def _save_animation_gif(self, out_dir: Path, output_id: str):
        file_path = out_dir / f"simulation_animation_{output_id}.gif"

        z_index = self.params.rendering.imaging.output_space.shape[0] // 2
        raw_frames = self.imaging_sim_output["frames"][:, z_index]
        norm = plt.Normalize(vmin=raw_frames.min(), vmax=raw_frames.max())
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

    def _save_kymograph_png(self, out_dir: Path, output_id: str):
        file_path = out_dir / f"kymograph_{output_id}.png"
        kymograph_raw = self.sample_kymograph_output["kymograph"]
        kymograph_resized = _resize_image(kymograph_raw, factor=4)
        plt.imsave(
            file_path,
            np.squeeze(kymograph_resized),
            cmap="gray",
        )

    def _save_kymograph_gt_png(self, out_dir: Path, output_id: str):
        file_path = out_dir / f"kymograph_gt_{output_id}.png"
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
