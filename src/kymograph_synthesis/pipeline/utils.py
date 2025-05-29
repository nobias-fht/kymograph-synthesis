import numpy as np
from numpy.typing import NDArray


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
