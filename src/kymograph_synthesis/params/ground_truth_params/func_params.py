from typing import Literal, Union, Annotated

from pydantic import BaseModel, Field

from .collection import GroundTruthFuncCollection


class StateGroundTruth(BaseModel):

    name: Literal[GroundTruthFuncCollection.STATE] = GroundTruthFuncCollection.STATE


class InstanceGroundTruth(BaseModel):

    name: Literal[GroundTruthFuncCollection.INSTANCE] = (
        GroundTruthFuncCollection.INSTANCE
    )


GroundTruthFuncParams = Annotated[
    Union[StateGroundTruth, InstanceGroundTruth], Field(discriminator="name")
]
