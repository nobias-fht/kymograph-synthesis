from typing import Annotated, Optional
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
FNAME_REGEX = "^(?:[a-zA-Z0-9_-]*\{output_id\}[a-zA-Z0-9_-]*)$"

AlphaNumericStr = Annotated[str, StringConstraints(pattern="^[a-zA-Z0-9]+$")]


class PipelineFilenames(BaseModel):

    model_config = ConfigDict(validate_assignment=True, validate_default=True)

    params: Annotated[str, StringConstraints(pattern=FNAME_REGEX)] = Field(
        default="params_{output_id}"
    )

    dynamics_sim_output: Annotated[str, StringConstraints(pattern=FNAME_REGEX)] = Field(
        default="dynamics_sim_output_{output_id}"
    )

    imaging_sim_output: Annotated[str, StringConstraints(pattern=FNAME_REGEX)] = Field(
        default="imaging_sim_output_{output_id}"
    )

    sample_kymograph_output: Annotated[str, StringConstraints(pattern=FNAME_REGEX)] = (
        Field(default="sample_kymograph_output_{output_id}")
    )

    generate_ground_truth_output: Annotated[
        str, StringConstraints(pattern=FNAME_REGEX)
    ] = Field(default="generate_ground_truth_output_{output_id}")

    kymograph_visual: Annotated[str, StringConstraints(pattern=FNAME_REGEX)] = Field(
        default="kymograph_{output_id}"
    )

    kymograph_gt_visual: Annotated[str, StringConstraints(pattern=FNAME_REGEX)] = Field(
        default="kymograph_gt_{output_id}"
    )

    animation_2d_visual: Annotated[str, StringConstraints(pattern=FNAME_REGEX)] = Field(
        default="simulation_animation_{output_id}"
    )

    @model_validator(mode="after")
    def validate_names_unique(self):
        fname_dict = self.model_dump()
        fnames = list(fname_dict.values())
        if len(fnames) != len(set(fnames)):
            raise ValueError(f"Pipeline write filenames not unique: {self}")
        return self


class WriteLog(BaseModel):

    model_config = ConfigDict(validate_assignment=True, validate_default=True)

    pipeline_filenames:PipelineFilenames = Field(default=PipelineFilenames())
    # alpha numeric 1 or more characters
    output_ids: set[AlphaNumericStr] = Field(default=set())

    def add_output_id(self, output_id: str):
        """Make sure output ids are validated"""
        self.output_ids.add(TypeAdapter(AlphaNumericStr).validate_python(output_id))


class WriteLogManager:

    def __init__(
        self, out_dir: Path, pipeline_filename: Optional[PipelineFilenames] = None
    ):
        self.fname = "write_log"
        self.path: Path = (out_dir / self.fname).with_suffix(".json")
        self.backup_path = (out_dir / self.fname).with_suffix(".json.backup")
        self.write_log: WriteLog

        load_existing, load_path = self._writelog_file_exists()

        if load_existing:
            if pipeline_filename is not None:
                logger.warning(
                    "Loading exisiting pipeline write log but pipeline filenames have "
                    "been provided. Only existing filenames in the write log will be "
                    "used."
                )
            self.write_log = _load_existing_write_log(load_path)
        else:
            self.write_log = (
                WriteLog()
                if pipeline_filename is None
                else WriteLog(pipeline_filename=pipeline_filename)
            )

    def add_output_id(self, output_id: str):
        self.write_log.add_output_id(output_id)

    def save(self):
        self.save_back_up()
        with open(self.path, "w") as f:
            json.dump(self.write_log.model_dump(mode="json"), f, indent=4)

    def save_back_up(self):
        if self.path.is_file():
            shutil.copy2(self.path, self.backup_path)

    def _writelog_file_exists(self) -> tuple[bool, Optional[Path]]:

        if self.path.is_file():
            return True, self.path

        if self.backup_path.is_file():
            return True, self.backup_path

        return False, None


def _load_existing_write_log(path: Path) -> WriteLog:
    with open(path, "r") as f:
        data = json.load(f)
    return WriteLog(**data)
