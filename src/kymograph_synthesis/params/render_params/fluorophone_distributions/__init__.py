from typing import Annotated, Union

from pydantic import Field

from .simplex_noise import SimplexNoiseParams

# possibility to expand union
FluorphoneDistributions = Annotated[
    Union[SimplexNoiseParams], Field(discriminator="name")
]