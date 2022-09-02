import numpy as np
import networkx as nx

from typing import Iterator, Tuple
from numpy.typing import ArrayLike


def extract_node_and_position(G: nx.Graph) -> Iterator[Tuple[int, Tuple[float, float]]]:
    """
    Extracts nodes and their corresponding gps coordinates from a networkx Graph. Note that the nodes in
    graph G must have 'x' and 'y' attributes, which is the case for graphs obtained via osmnx.
    """
    for node, attr_dict in G.nodes.data():
        yield node, (attr_dict['x'], attr_dict['y'])


def extract_positions(G: nx.Graph) -> np.array:
    """
    Extracts gps coordinates of nodes from a networkx Graph. Note that the nodes in
    graph G must have 'x' and 'y' attributes, which is the case for graphs obtained via osmnx.
    """
    return np.fromiter((pos for _, pos in extract_node_and_position(G=G)), dtype=np.dtype((float, 2)))


def lat_to_meters(lat: float) -> float:
    """
    Converts distance in degrees latitude to meters.
    """
    return lat * 110540


def lon_to_meters(lat: float, lon: float) -> float:
    """
    Converts distance in degrees longitude to meters. Note that conversion between longitude and meters
    depends on the latitude.
    """
    return lon * (111320 * abs(np.cos(np.radians(lat))))


def gps_to_meters(lat: float) -> Tuple[float, float]:
    """
    Returns a tuple of scaling factors between 2d gps coordinates and meters. This can be used to convert
    an entire numpy array of longitude and latitude values to meters via multiplication by the returned
    tuple.
    """
    return lon_to_meters(lat, 1), lat_to_meters(1)


def meters_to_lon(lat: float, m: float) -> float:
    """
    Converts distance in meters to degrees longitude. Note that conversion between meters and longitude
    depends on the latitude.
    """
    return m/(111320 * abs(np.cos(np.radians(lat))))


def meters_to_lat(m: float):
    """
    Converts distance in meters to degrees latitude.
    """
    return m/110540


def meters_to_gps(lat: float) -> Tuple[float, float]:
    """
    Returns a tuple of scaling factors between meters and 2d gps coordinates. This can be used to convert
    an entire numpy array of meter values to longitude and latitude via multiplication by the returned
    tuple.
    """
    return meters_to_lon(lat, 1), meters_to_lat(1)


def gps_dist(xy1: np.array, xy2: np.array) -> np.array:
    """
    Get the pairwise distance between the arrays of 2d-gps coordinates (longitude, latitude) in meters.
    """
    return np.linalg.norm((xy1 - xy2) * gps_to_meters(lat=xy1[1]), axis=-1)

