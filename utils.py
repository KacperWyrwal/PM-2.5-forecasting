import numpy as np
import pandas as pd

from functools import wraps
from time import perf_counter
from sklearn.preprocessing import StandardScaler

from typing import Callable, List, Iterable, Set, Tuple
from numpy.typing import ArrayLike


def timeit(func: Callable) -> Callable:
    """
    Timing wrapper. Useful for evaluating performance when writing code operating on large graphs.
    """

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} took {total_time:.6f} seconds')
        return result

    return timeit_wrapper


def find_nearest(arr: ArrayLike, vals: ArrayLike) -> np.array:
    """
    Find the indices of the nearest values in arr from vals.
    """
    idx = np.searchsorted(arr, vals)
    return idx - (abs(arr[idx - 1] - vals) < abs(arr[idx % (len(arr))] - vals))


def delete_indices(arr: List, indices: ArrayLike) -> None:
    """
    Delete indices in arr from indices.
    """
    for idx in sorted(indices, reverse=True):
        del arr[idx]


class MissingSequence:
    """
    Dummy class representing a missing sequence. Always returns None when indexed and when checking length.
    """

    def __getitem__(self, key):
        return None

    def __len__(self, key):
        return None


def standardize(df: pd.DataFrame, cols: Iterable[str]) -> StandardScaler:
    """
    Standardize given columns of the given dataframe and return a fitted StandardScaler instance.
    """
    scaler = StandardScaler().fit(df[cols])
    df[cols] = scaler.transform(df[cols])
    return scaler


def persistent_gps(batch_gps: List[np.array]) -> Set[Tuple[float, float]]:
    """
    Return the gps coordinates present in all snapshots of a given batch.
    """
    return set.intersection(*(set(map(tuple, snapshot)) for snapshot in batch_gps))
