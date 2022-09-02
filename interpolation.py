import numpy as np

from functools import cache

from typing import Sequence


def dists(x: np.array, y: np.array, /) -> np.array:
    """
    Compute distances between each element of x and each element of y.
    x: (n, k) array
    y: (m, k) array
    return: (n, m) array of distances
    """
    return np.linalg.norm(y - x.reshape(-1, 1, 2), axis=2)


def weights(x: np.array, y: np.array, /) -> np.array:
    """
    Compute weights (inverse distances squared) between each element of x and each element of y.
    """
    return np.power(dists(x, y), -2)


def interpolate(to: np.array, fro: np.array, val: np.array) -> np.array:
    """
    Interpolate values from positions in fro to positions in to accoring to the average weighted by
    the inverse distances squared.
    """
    return np.average(
        np.broadcast_to(val, (len(to), len(val))),
        axis=1,
        weights=weights(to, fro),
    )


class Interpolator:
    """
    Interpolates measurements from a variable set of coordinates to a static set of coordinates. The
    Intepolator class utilises caching to greatly improve the runtime where the set of coordinates,
    from which the measurements are interpolated, does not change often.
    """

    def __init__(self, to: np.array):
        """
        to: the static set of coordinates onto which the measurements will be interpolated.
        """
        self.to = to

    @cache
    def dist(self, fro: tuple) -> np.array:
        """
        Cached euclidean distance function. The fro argument must be a immutable in order to be cachable.
        """
        return np.linalg.norm(self.to - fro, axis=1)

    def dists(self, fros: np.array) -> np.array:
        """
        Euclidean distance for entire array of coordinates. These are internally converted into tuples,
        so the argument need not be immutable, as opposed to the dist method.
        """
        return np.fromiter((
            self.dist(tuple(fro)) for fro in fros
        ), dtype=np.dtype((float, len(self.to)))).T

    def weights(self, fros: np.array) -> np.array:
        """
        Compute weights (inverse distances squared) between each element of fros and each element of
        self.to. Whenever a zero-distance position is enountered, it is given a weight of 1 and all
        non-zero distance positions are given a weight of 0.
        """
        dist = self.dists(fros=fros)
        return np.fromiter((
            zeros if np.any(zeros := row == 0) else np.power(row, -2)
            for row in dist
        ), dtype=np.dtype((float, (len(fros), ))))

    def interpolate(self, fros: np.array, vals: np.array) -> np.array:
        """
        Interpolate a snapshot. TODO perhaps change the arguments to a single snapshot object.
        """
        return np.average(
            np.broadcast_to(vals, (len(self.to), len(vals))),
            axis=1,
            weights=self.weights(fros=fros)
        )

    def interpolate_batch(self, pm_batch: Sequence[np.array], gps_batch: Sequence[np.array]) -> np.array:
        """
        Convenience function for interpolating an entire batch.
        """
        return np.fromiter((
            self.interpolate(fros=gps, vals=pm) for pm, gps in zip(pm_batch, gps_batch)
        ), dtype=np.dtype((float, len(self.to))))

    def clear_cache(self) -> None:
        """
        When the positions from which the values are interpolated vary frequently, the cache might get
        prohibitively large and one might want to clear it. Another solutions for cache management would
        be to change the cache decorator to lru_cache.
        """
        self.dist.cache_clear()

    def __call__(self, *args, **kwargs) -> np.array:
        return self.interpolate(*args, **kwargs)
