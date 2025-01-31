from typing import Literal
from typing_extensions import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator, Field, PositiveInt

from .collection import FluoroDistrName


class SimplexNoiseParams(BaseModel):

    model_config = ConfigDict(validate_assignment=True, validate_default=True)

    name: Literal[FluoroDistrName.SIMPLEX_NOISE] = FluoroDistrName.SIMPLEX_NOISE

    noise_scales: tuple[float, ...] = Field(default=(1, 0.25))
    scale_weights: tuple[float, ...] = Field(
        default_factory=lambda data: tuple(1 for _ in range(len(data["noise_scales"]))),
        validate_default=True,
    )
    max_fluorophore_count_per_nm3: float
    # random int64 as specified in the opensimplex docs
    seed: int = Field(
        default_factory=lambda: np.random.randint(-(2**63), 2**63 - 1),
        validate_default=True,
    )

    @model_validator(mode="after")
    def validate_scales_and_weights_length(self) -> Self:
        if len(self.noise_scales) != len(self.scale_weights):
            raise ValueError(
                f"Length of `scales` and `scale_weights`, {len(self.scales)} and "
                f"{len(self.scale_weights)}, are not the same."
            )
        return self
