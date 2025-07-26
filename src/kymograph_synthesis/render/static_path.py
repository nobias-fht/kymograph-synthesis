from typing import Protocol, Callable, Any

import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy.interpolate import interp1d


class StaticPath(Protocol):

    def __call__(self, ratio: NDArray[np.floating]) -> NDArray[np.floating]: ...

    def length(self) -> float: ...

    def tangents(self, ratios: NDArray[np.floating]) -> NDArray[np.floating]: ...


class LinearPath:

    def __init__(self, start: NDArray[np.floating], end: NDArray[np.floating]):
        self.dims = len(start)
        self.start = start
        self.end = end
        self.direction = end - start

    def __call__(self, ratio: NDArray[np.floating]) -> NDArray[np.floating]:
        shape = ratio.shape
        result = self.start.reshape(1, -1) + np.outer(ratio.flatten(), self.direction)
        return result.reshape(*shape, self.dims)

    def length(self) -> float:
        return np.linalg.norm(self.direction, ord=2).item()

    def tangents(self, ratios: NDArray[np.floating]) -> NDArray[np.floating]:
        tangent_vec = self.direction / np.sum(self.direction ** 2) ** 0.5
        return np.tile(tangent_vec, (*ratios.shape, 1))


class PiecewiseLinearPath:

    def __init__(self, vertices: list[NDArray[np.floating]]):
        # TODO: rename variable to be more cohesive with piecewise bezier path
        self.n_segments = len(vertices) - 1
        self.vertices = vertices
        self.dims = len(self.vertices[0])  # TODO: validate all vertices have same dims
        self.linear_path_segments: list[LinearPath] = [
            LinearPath(start=self.vertices[i], end=self.vertices[i + 1])
            for i in range(self.n_segments)
        ]
        self.segment_magnitudes: NDArray[np.floating] = np.array(
            [
                np.linalg.norm(path_segment.end - path_segment.start, ord=2)
                for path_segment in self.linear_path_segments
            ]
        )
        self.total_length = np.sum(self.segment_magnitudes)
        self.segment_ratio_bins = np.concatenate(
            [np.array([0]), np.cumsum(self.segment_magnitudes) / self.total_length]
        )

    def __call__(self, ratio: NDArray[np.floating]) -> NDArray[np.floating]:
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

    def length(self) -> float:
        return sum([segment.length() for segment in self.linear_path_segments])


class QuadraticBezierPath:

    def __init__(self, points: tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]):
        self.points = points
        self.dims = len(points[0])

    def __call__(self, ratio: NDArray[np.floating]) -> NDArray[np.floating]:
        ratio[ratio < 0] = np.nan
        ratio[ratio > 1] = np.nan
        ratio_mapping = self.ratio_mapping(n=128)
        t = ratio_mapping(ratio)
        return self._bezier_func(t)

    def tangents(self, ratios: NDArray[np.floating]) -> NDArray[np.floating]:
        shape = ratios.shape
        p0, p1, p2 = self.points
        result = (
            p1
            + np.outer(2*(1 - ratios.flatten()), (p0 - p1))
            + np.outer(2*ratios.flatten(), (p2 - p1))
        )
        result.reshape(*shape, self.dims)
        return result / np.sum(result**2, axis=-1, keepdims=True) ** 0.5

    def _bezier_func(self, t: NDArray[np.floating]) -> NDArray[np.floating]:
        shape = t.shape
        p0, p1, p2 = self.points
        result = (
            p1
            + np.outer((1 - t.flatten()) ** 2, (p0 - p1))
            + np.outer(t.flatten() ** 2, (p2 - p1))
        )
        result.reshape(*shape, self.dims)
        return result

    def ratio_mapping(self, n=128) -> Callable[[NDArray[np.floating]], NDArray[np.floating]]:
        # TODO: allow change of n?
        lengths = self._calc_lengths(n=n)
        x = np.concatenate([np.array([0]), np.cumsum(lengths) / np.cumsum(lengths)[-1]])
        return interp1d(x, np.linspace(0, 1.0, n))

    def _calc_lengths(self, n=128) -> NDArray[np.floating]:
        samples = np.linspace(0, 1, n)
        coords = self._bezier_func(samples)
        vecs = coords[1:] - coords[:-1]
        lengths = np.linalg.norm(vecs, ord=2, axis=1)
        return lengths

    def length(self) -> float:
        return self._calc_lengths().sum()


class PiecewiseQuadraticBezierPath:

    def __init__(self, points: list[NDArray[np.floating]]):

        self.dims = len(points[0])
        n_points = len(points)
        midpoints = [(points[i] + points[i + 1]) / 2 for i in range(n_points - 1)]
        self.path_segments: list[StaticPath] = [
            LinearPath(start=points[0], end=midpoints[0])
        ]
        for i in range(0, n_points - 2):
            self.path_segments.append(
                QuadraticBezierPath(
                    points=(midpoints[i], points[i + 1], midpoints[i + 1])
                )
            )
        self.path_segments.append(LinearPath(start=midpoints[-1], end=points[-1]))
        self.n_segments = len(self.path_segments)

        self.segment_lengths = self._segment_lengths()
        self.total_length = sum(self.segment_lengths)

        self.segment_ratio_bins = np.concatenate(
            [np.array([0]), np.cumsum(self.segment_lengths) / self.total_length]
        )

    def __call__(self, ratio: NDArray[np.floating], thickness=1) -> NDArray[np.floating]:
        if (thickness > 1) and (self.dims != 2):
            raise NotImplementedError("Cannot give a thickness for dims != 2.")

        segment_labels = np.digitize(ratio, bins=self.segment_ratio_bins[1:], right=True)
        result = np.zeros((*ratio.shape, thickness, self.dims))  # initialize place holder
        for n in range(self.n_segments):
            path_segment = self.path_segments[n]
            segment_ratios = ratio[segment_labels == n]
            # scale correctly
            segment_ratios = (segment_ratios - self.segment_ratio_bins[n]) * (
                self.total_length / self.segment_lengths[n]
            )

            segment_coords = path_segment(segment_ratios)
            if thickness > 1:
                segment_tangents = path_segment.tangents(segment_ratios)
                segment_normals = np.stack([segment_tangents[:,1], -segment_tangents[:,0]], axis=-1)
                segment_result = np.linspace(
                    segment_coords - 0.5*thickness*segment_normals, 
                    segment_coords + 0.5*thickness*segment_normals,
                    thickness,
                    endpoint = True
                )
                segment_result = np.moveaxis(segment_result, 0, 1)
            else:
                segment_result = segment_coords[:, np.newaxis]

            result[segment_labels == n] = segment_result
        return result

    def _segment_lengths(self):
        return [
            segment.length() for segment in self.path_segments
        ]
    
    def length(self):
        return sum(self._segment_lengths())
