from typing import Protocol

from numpy.typing import NDArray

class SpaceTimeObject(Protocol):

    def render(self, t: int, space: NDArray) -> NDArray:
        ...