from typing import Annotated, Optional, overload, Literal
from pathlib import Path
import json
import shutil
import logging

from pydantic import (
    BaseModel,
    StringConstraints,
    model_validator,
    ConfigDict,
    Field,
    TypeAdapter,
)

logger = logging.getLogger(__name__)

# alphanumeric + dash and underscore, has to include "{output id}""
# TODO: now also allowed are curly brackets for {type} in visual ground truth
#   This should be validated better
FILE_NAME_REGEX = "^(?:[a-zA-Z0-9_-}{]*\{output_id\}[a-zA-Z0-9_-}{]*)$"
# alphanumeric . at the start and one other . allowed (for example .tar.gz)
FILE_EXTENSION_REGEX = "^\.[a-zA-Z0-9]+(\.[a-zA-Z0-9]+)?$"

AlphaNumericStr = Annotated[str, StringConstraints(pattern="^[a-zA-Z0-9]+$")]


class OutputFileName(BaseModel):

    name: Annotated[str, StringConstraints(pattern=FILE_NAME_REGEX)]
    extension: Annotated[str, StringConstraints(pattern=FILE_EXTENSION_REGEX)]

    def file_name(self, output_id: str) -> str:
        return self.name.format(output_id=output_id) + self.extension


class PipelineFilenames(BaseModel):

    model_config = ConfigDict(validate_assignment=True, validate_default=True)

    params: OutputFileName = OutputFileName(
        name="params_{output_id}",
        extension=".yaml",
    )

    dynamics_sim: OutputFileName = OutputFileName(
        name="dynamics_sim_output_{output_id}", extension=".npz"
    )

    imaging_sim: OutputFileName = OutputFileName(
        name="imaging_sim_output_{output_id}", extension=".npz"
    )

    sample_kymograph: OutputFileName = OutputFileName(
        name="sample_kymograph_output_{output_id}", extension=".npz"
    )

    generate_ground_truth: OutputFileName = OutputFileName(
        name="generate_ground_truth_output_{output_id}", extension=".npz"
    )

    kymograph_visual: OutputFileName = OutputFileName(
        name="kymograph_{output_id}", extension=".png"
    )

    kymograph_gt_visual: OutputFileName = OutputFileName(
        name="kymograph_gt_{{type}}_{output_id}", extension=".png"
    )

    animation_2d_visual: OutputFileName = OutputFileName(
        name="simulation_animation_{output_id}", extension=".gif"
    )

    @model_validator(mode="after")
    def validate_names_unique(self):
        fname_dict = self.model_dump()
        # TODO: can improve this
        fnames = [file_name["name"] for file_name in fname_dict.values()]
        if len(fnames) != len(set(fnames)):
            raise ValueError(f"Pipeline write filenames not unique: {self}")
        return self


class WriteLog(BaseModel):

    model_config = ConfigDict(validate_assignment=True, validate_default=True)

    pipeline_filenames: PipelineFilenames = Field(default=PipelineFilenames())
    # alpha numeric 1 or more characters
    output_ids: set[AlphaNumericStr] = Field(default=set())

    def add_output_id(self, output_id: str):
        """Make sure output ids are validated"""
        self.output_ids.add(TypeAdapter(AlphaNumericStr).validate_python(output_id))


class WriteLogManager:

    def __init__(
        self, out_dir: Path, pipeline_filenames: Optional[PipelineFilenames] = None
    ):
        self.fname = "write_log"
        self.path: Path = (out_dir / self.fname).with_suffix(".json")
        self.backup_path = (out_dir / self.fname).with_suffix(".json.backup")
        self.write_log: WriteLog

        load_path = self._get_writelog_path()

        if load_path is not None:
            if pipeline_filenames is not None:
                logger.warning(
                    "Loading exisiting pipeline write log but pipeline filenames have "
                    "been provided. Only existing filenames in the write log will be "
                    "used."
                )
            self.write_log = _load_existing_write_log(load_path)
        else:
            self.write_log = (
                WriteLog()
                if pipeline_filenames is None
                else WriteLog(pipeline_filenames=pipeline_filenames)
            )

    def add_output_id(self, output_id: str):
        self.write_log.add_output_id(output_id)

    def log(self):
        self.save_back_up()
        with open(self.path, "w") as f:
            json.dump(self.write_log.model_dump(mode="json"), f, indent=4)

    def save_back_up(self):
        if self.path.is_file():
            shutil.copy2(self.path, self.backup_path)

    def create_new_id(self, n_digits: int = 4) -> str:
        # TODO: allow different ID creation strategies - probs overkill
        existing_ids = [
            int(file_id) for file_id in self.write_log.output_ids if file_id.isnumeric()
        ]
        if len(existing_ids) == 0:
            new_id = 0
        else:
            new_id = max(existing_ids) + 1
        return "{1:0{0}d}".format(n_digits, new_id)

    def _get_writelog_path(self) -> Optional[Path]:

        if self.path.is_file():
            return self.path

        if self.backup_path.is_file():
            return self.backup_path

        return None


def _load_existing_write_log(path: Path) -> WriteLog:
    with open(path, "r") as f:
        data = json.load(f)
    return WriteLog(**data)
