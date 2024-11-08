from typing import Protocol, Callable

import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy.interpolate import interp1d

class StaticPath(Protocol):

    def __call__(self, ratio: NDArray) -> NDArray: ...


class LinearPath:

    def __init__(self, start: NDArray, end: NDArray):
        self.dims = len(start)
        self.start = start
        self.end = end
        self.direction = end - start

    def __call__(self, ratio: NDArray) -> NDArray:
        shape = ratio.shape
        result = self.start.reshape(1, -1) + np.outer(ratio.flatten(), self.direction)
        return result.reshape(*shape, self.dims)


class PiecewiseLinearPath:

    def __init__(self, vertices: list[NDArray]):
        self.n_segments = len(vertices) - 1
        self.vertices = vertices
        self.dims = len(self.vertices[0])  # TODO: validate all vertices have same dims
        self.linear_path_segments: list[LinearPath] = [
            LinearPath(start=self.vertices[i], end=self.vertices[i + 1])
            for i in range(self.n_segments)
        ]
        self.segment_magnitudes: NDArray[np.float_] = np.array(
            [
                np.linalg.norm(path_segment.end - path_segment.start, ord=2)
                for path_segment in self.linear_path_segments
            ]
        )
        self.total_length = np.sum(self.segment_magnitudes)
        self.segment_ratio_bins = np.concatenate(
            [np.array([0]), np.cumsum(self.segment_magnitudes) / self.total_length]
        )

    def __call__(self, ratio: NDArray) -> NDArray:
        segment_mask = np.digitize(ratio, bins=self.segment_ratio_bins[1:], right=True)
        result = np.zeros((*ratio.shape, self.dims))  # initialize place holder
        for n in range(self.n_segments):
            linear_path_segment = self.linear_path_segments[n]
            segment_ratios = ratio[segment_mask == n]
            # scale correctly
            segment_ratios = (segment_ratios - self.segment_ratio_bins[n]) * (
                self.total_length / self.segment_magnitudes[n]
            )
            segment_result = linear_path_segment(segment_ratios)
            result[segment_mask == n] = segment_result

        return result

class QuadraticBezierPath:

    def __init__(self, points: tuple[NDArray, NDArray, NDArray]):
        self.points = points
        self.dims = len(points[0])

    def __call__(self, ratio: NDArray) -> NDArray:
        ratio[ratio < 0] = np.nan
        ratio[ratio > 1] = np.nan
        ratio_mapping = self.ratio_mapping(n=128)
        t = ratio_mapping(ratio)
        return self._bezier_func(t)
    
    def _bezier_func(self, t: NDArray) -> NDArray:
        shape = t.shape
        p0, p1, p2 = self.points
        result = (
            p1 
            + np.outer((1 - t.flatten()) ** 2, (p0 - p1))
            + np.outer(t.flatten() ** 2, (p2 - p1))
        )
        result.reshape(*shape, self.dims)
        return result
    
    def ratio_mapping(self, n=128) -> Callable[[NDArray], NDArray]:
        lengths = self._calc_lengths(n=n)
        x = np.concatenate([np.array([0]), np.cumsum(lengths)/np.cumsum(lengths)[-1]])
        return interp1d(x, np.linspace(0, 1.0, n))
    
    def _calc_lengths(self, n=128) -> NDArray:
        samples = np.linspace(0, 1, n)
        coords = self._bezier_func(samples)
        vecs = coords[1:] - coords[:-1]
        lengths = np.linalg.norm(vecs, ord=2, axis=1)
        return lengths