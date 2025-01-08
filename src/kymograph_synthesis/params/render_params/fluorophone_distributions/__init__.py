from typing import Annotated, Union

from pydantic import Field

from .simplex_noise import SimplexNoiseParams

# possibility to expand union
FluorphoneDistributionParams = Annotated[
    Union[SimplexNoiseParams], Field(discriminator="name")
]
