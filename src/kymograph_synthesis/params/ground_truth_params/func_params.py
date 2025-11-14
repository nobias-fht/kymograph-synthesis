from typing import Literal, Union, Annotated

from pydantic import BaseModel, Field

from .collection import GroundTruthFuncCollection


class StateGroundTruth(BaseModel):

    name: Literal[GroundTruthFuncCollection.STATE] = GroundTruthFuncCollection.STATE
    line_thickness: float = 1
    project_to_xy: bool


class InstanceGroundTruth(BaseModel):

    name: Literal[GroundTruthFuncCollection.INSTANCE] = (
        GroundTruthFuncCollection.INSTANCE
    )
    line_thickness: float = 1
    project_to_xy: bool

GroundTruthFuncParams = Annotated[
    Union[StateGroundTruth, InstanceGroundTruth], Field(discriminator="name")
]
