from typing import Protocol

from numpy.typing import NDArray

class SpaceTimeObject(Protocol):

    def render_time_point(self, t: int, space: NDArray) -> NDArray:
        ...

    def render(self, space_time: NDArray) -> NDArray:
        ...