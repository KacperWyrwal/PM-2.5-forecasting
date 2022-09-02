import datetime
import numpy as np


def month_to_vec(datetime: datetime.datetime) -> np.array:
    """
    Convert datetime object into one-hot encoded representation of its month.
    """
    vec = np.zeros(12)
    vec[datetime.month - 1] = 1
    return vec


def weekday_to_vec(datetime: datetime.datetime) -> np.array:
    """
    Convert datetime object into one-hot encoded representation of its weekday.
    """
    vec = np.zeros(7)
    vec[datetime.weekday()] = 1
    return vec


def hour_to_vec(datetime: datetime.datetime) -> np.array:
    """
    Convert datetime object into one-hot encoded representation of its hour.
    """
    vec = np.zeros(24)
    vec[datetime.hour] = 1
    return vec


def datetime_to_vec(datetime: datetime.datetime) -> np.array:
    """
    Convert datetime object into a vector encoding its month, weekday, and hour.
    """
    month_vec = month_to_vec(datetime=datetime)
    weekday_vec = weekday_to_vec(datetime=datetime)
    hour_vec = hour_to_vec(datetime=datetime)
    return np.concatenate((month_vec, weekday_vec, hour_vec))
