from typing import Annotated, Union

from pydantic import Field

__all__ = [
    "SimplexNoiseParams",
    "FluoroDistrName",
    "FluoroDistrParams"
]

from .simplex_noise import SimplexNoiseParams
from .collection import FluoroDistrName

# possibility to expand union
FluoroDistrParams = Annotated[
    Union[SimplexNoiseParams], Field(discriminator="name")
]
