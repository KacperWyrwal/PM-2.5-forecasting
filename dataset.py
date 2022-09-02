import numpy as np
import pandas as pd

from dataclasses import dataclass, field
from utils import find_nearest, delete_indices, MissingSequence

from typing import Sequence, Iterable, Callable, Any, List, Union
from datetime import datetime


def comp_snapshot_indices(index: np.array, snapshot_minutes: int) -> np.array:
    """
    Returns a (number of snapshots)-by-2 array whose each row represents indices [i_start, i_end],
    where i_start is the beginning index of a snapshot (inclusive), and i_end is the ending index of a
    snapshot (exclusive).
    """
    indices = np.nonzero(np.diff(index.to_numpy()) >= np.timedelta64(snapshot_minutes, 'm'))
    # add a zero in front, and length at the back
    return np.column_stack((np.insert(indices, 0, 0), np.append(indices, len(index))))


def comp_snapshot_timestamps(snapshot_indices: np.array, timestamps: np.array) -> np.array:
    """
    For the snapshot_indices array of the type returned by the comp_snapshot_indices compute the
    average timestamp corresponding to each snapshot.
    """
    return np.fromiter(
        (timestamps[start:end].view('i8').mean()
         for start, end in snapshot_indices),
        dtype='datetime64[ns]',
    )


def average_duplicates(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """
    Average duplicates according to the index and columns i.e. for rows to be averaged
    """
    index_name = df.index.name  # arguably, perhaps index should be explicitly stated in the arguments
    temp_col_name = ', '.join(columns)
    df[temp_col_name] = list(map(tuple, df[columns].values))
    return df.groupby([df.index, temp_col_name]).mean().reset_index().drop(columns=temp_col_name). \
        set_index(index_name)


def split_snapshot_indices_to_batches(timestamps: np.array, snapshot_indices: np.array,
                                      snapshot_minutes: int) -> np.array:
    """
        Split snapshot indices into batches based on snapshot duration. If the end of one snapshot
        if farther away in time from the beginning of the next snapshot than snapshot_minutes, then
        a new batch begins.
    """
    # start of snapshots (except first) minus end of snapshots (except last)
    return np.nonzero(
        timestamps[snapshot_indices[1:, 0]] - timestamps[snapshot_indices[:-1, 1] - 1]
        > np.timedelta64(snapshot_minutes, 'm')
    )[0]


def find_indices_of_unit_batches(batch_indices: Sequence[Sequence]) -> List[int]:
    return [i for i, n in enumerate(map(len, batch_indices)) if n <= 1]


def map_timestamps_to_features(timestamps: np.array, f: Callable[[datetime], Any]) -> np.array:
    return np.array(list(map(f, pd.to_datetime(timestamps))))


def split_by_double_index(index: np.array, arr: np.array) -> List[np.array]:
    """
    Double index here stands for a 2-by-n array where each row contains two indices representing a single
    snapshot. This kind of index contains redundant information, but is easier to work with in the case
    of the Dataset class.
    """
    return [arr[start:end] for start, end in index]


@dataclass
class Snapshot:
    """
    Snapshot dataclass used by the Batch dataclass.
    """
    timestamp: datetime = field(default=None)
    gps: np.array = field(default=None)
    pm: np.array = field(default=None)
    weather: np.array = field(default=None)
    time: np.array = field(default=None)


@dataclass
class Batch:
    """
    Batch dataclass used by the Dataset class.
    """
    timestamps: np.array = field(default=None)
    time: Sequence[np.array] = field(default=MissingSequence())
    weather: Sequence[np.array] = field(default=MissingSequence())
    pm: Sequence[np.array] = field(default=MissingSequence())
    gps: Sequence[np.array] = field(default=MissingSequence())

    def __getitem__(self, key: Union[int, slice]) -> Union[Snapshot, List[Snapshot]]:
        if isinstance(key, int):
            return Snapshot(
                timestamp=self.timestamps[key],
                time=self.time[key],
                weather=self.weather[key],
                pm=self.pm[key],
                gps=self.gps[key],
            )
        if isinstance(key, slice):
            return [self[k] for k in range(*key.indices(len(self)))]

    def __len__(self) -> int:
        return len(self.timestamps)


class Dataset:
    """
    Dataset class makes sure that the PM 2.5, Weather, and Time features are aligned within
    batches and snapshots - that is, in a single snapshot the PM 2.5 values, weather features, and time
    features all correspond to the same time window.
    """

    def __init__(
            self,
            pm_gps: pd.DataFrame,
            snapshot_minutes: int,
            weather: pd.DataFrame = None,
            minutes_to_weather: int = None,
            datetime_to_vec: Callable[[datetime], Any] = None,
    ):
        self.prep_pm_gps(pm_gps=pm_gps, snapshot_minutes=snapshot_minutes)

        # Save values arrays for getitem functionality, later, if weather dataframe is passed and/or
        # datetime_to_vec function is passed, their corresponding values attributes are set.
        self.pm_values = pm_gps['pm2_5'].values
        self.gps_values = pm_gps[['gpsLongitude', 'gpsLatitude']].values
        self.weather_values = None
        self.time_values = None

        # Compute snapshot indices and timestamps corresponding to these snapshots
        self.pm_gps_idx = comp_snapshot_indices(index=pm_gps.index,
                                                snapshot_minutes=snapshot_minutes)
        self.timestamps = comp_snapshot_timestamps(snapshot_indices=self.pm_gps_idx,
                                                   timestamps=pm_gps.index.values)

        if weather is not None:
            self.weather_values = weather.values
            self.weather_idx = find_nearest(weather.index.values, self.timestamps)
            # Remove indices and timestamps with weather not close enough
            self.remove_snapshots_without_weather(weather_index_values=weather.index.values,
                                                  minutes_to_weather=minutes_to_weather)

        if datetime_to_vec is not None:
            # Converting timestamps to features here means that no work is repeated when loading the
            # same batch multiple times
            self.time_values = map_timestamps_to_features(timestamps=self.timestamps, f=datetime_to_vec)

        # Split the indices of snapshots and timestamps into batches removing unit batches
        self.split_to_batches(pm_gps_index_values=pm_gps.index.values, snapshot_minutes=snapshot_minutes)

    def prep_pm_gps(self, pm_gps: pd.DataFrame, snapshot_minutes: int) -> None:
        """
        Prepare the pm_gps dataframe for computing snapshot indices:
        1. Set up index floored over snapshot duration
        2. Average out multiple measurements at same locations in one snapshot

        Note that this function modifies pm_gps in place.
        """
        pm_gps.set_index(pm_gps.index.floor(f'{snapshot_minutes}min'), inplace=True)
        average_duplicates(df=pm_gps, columns=['gpsLatitude', 'gpsLongitude'])

    def weather_available_mask(self, weather_index_values: np.array,
                                   minutes_to_weather: int) -> np.array:
        """
        Returns a mask where True indicates that a weather reading is within minutes_to_weather of
        the corresponding snapshot.
        """
        return np.abs(weather_index_values[self.weather_idx] - self.timestamps) <= np.timedelta64(
            minutes_to_weather, 'm')

    def remove_snapshots_without_weather(self, weather_index_values: np.array,
                                         minutes_to_weather: int) -> None:
        """
        Removes snapshots where no weather reading in available within minutes_to_weather distance
        of the snapshot's timestamp.
        """
        mask = self.weather_available_mask(weather_index_values=weather_index_values,
                                           minutes_to_weather=minutes_to_weather)
        self.pm_gps_idx = self.pm_gps_idx[mask]
        self.timestamps = self.timestamps[mask]
        self.weather_idx = self.weather_idx[mask]

    def remove_unit_batches(self) -> None:
        """
        Unit batches are removed as they do not provide enough time steps for time-series prediction.
        """
        idx = find_indices_of_unit_batches(self.pm_gps_idx)
        delete_indices(self.pm_gps_idx, idx)
        delete_indices(self.timestamps, idx)
        if self.time_values is not None:
            delete_indices(self.time_values, idx)
        if self.weather_values is not None:
            delete_indices(self.weather_idx, idx)

    def split_to_batches(self, pm_gps_index_values: np.array, snapshot_minutes: int) -> None:
        """
        Split snapshot indices further into batches for all available data.
        """
        batch_splits = split_snapshot_indices_to_batches(
            timestamps=pm_gps_index_values,
            snapshot_indices=self.pm_gps_idx,
            snapshot_minutes=snapshot_minutes,
        )
        self.pm_gps_idx = np.split(self.pm_gps_idx, batch_splits)
        self.timestamps = np.split(self.timestamps, batch_splits)
        if self.time_values is not None:
            self.time_values = np.split(self.time_values, batch_splits)
        if self.weather_values is not None:
            self.weather_idx = np.split(self.weather_idx, batch_splits)
        self.remove_unit_batches()

    def __getitem__(self, key: Union[int, slice]) -> Union[Batch, List[Batch]]:
        """
        Returns a batch or a list of batches for the given index/indices with all data available. PM 2.5
        values and GPS values will always be returned as these are mandary, and, if weather dataframe
        and/or time-to-features function is provided, weather values and/or time values will be returned
        as well.
        """
        if isinstance(key, int):
            batch_kwargs = {
                'timestamps': self.timestamps[key],
                'pm': split_by_double_index(arr=self.pm_values, index=self.pm_gps_idx[key]),
                'gps': split_by_double_index(arr=self.gps_values, index=self.pm_gps_idx[key]),
            }
            if self.weather_values is not None:
                batch_kwargs['weather'] = self.weather_values[self.weather_idx[key]]
            if self.time_values is not None:
                batch_kwargs['time'] = self.time_values[key]
            return Batch(**batch_kwargs)
        if isinstance(key, slice):
            return [self[k] for k in range(*key.indices(len(self)))]

    def __len__(self):
        return len(self.timestamps)
