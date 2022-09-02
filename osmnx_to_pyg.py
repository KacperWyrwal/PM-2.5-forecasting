import numpy as np
import torch_geometric as pyg
import networkx as nx

from typing import Tuple


def clear_node_features(G: nx.Graph, ignore: Tuple[str, ...] = ('x', 'y')) -> None:
    """
    Remove node features from the graph G except those given in ignore.
    """
    for _, attr_dict in G.nodes(data=True):
        temp = {f: attr_dict[f] for f in ignore}
        attr_dict.clear()
        attr_dict.update(temp)


def clear_edge_features(G: nx.Graph, ignore: Tuple[str, ...] = ('length',)) -> None:
    """
    Remove node features from the graph G except those given in ignore.
    """
    for u, v, attr_dict in G.edges(data=True):
        temp = {f: attr_dict[f] for f in ignore}
        attr_dict.clear()
        attr_dict.update(temp)


def extract_pos(data: pyg.data.Data, x_idx: int, y_idx: int) -> None:
    """
    Move the x, y coordinates from the x dictionary of the data object into the new pos dictionary.
    """
    data.pos = data.x[:, [x_idx, y_idx]]
    data.x = data.x[:, np.delete(np.arange(data.x.shape[1]), [x_idx, y_idx])]


def osmnx_to_pyg(G: nx.Graph, node_features: Tuple[str, ...] = ('x', 'y'),
                 edge_features: Tuple[str, ...] = ('length',), pos: bool = True, inplace: bool = False) \
        -> pyg.data.Data:
    """
    Convert a networkx graph G into a Pytorch Geometric data object keeping the desired node features
    and edge features. Additionally, if the nodes features contain the 'x' and 'y' keys and pos is given
    as True, the x and y features are extracted to a separate 'pos' dictionary of the returned data object.
    """

    if inplace is False:
        G = G.copy()

    clear_node_features(G, ignore=node_features)
    clear_edge_features(G, ignore=edge_features)

    data = pyg.utils.from_networkx(G, group_node_attrs=node_features, group_edge_attrs=edge_features)

    if pos is True:
        extract_pos(data, x_idx=node_features.index('x'), y_idx=node_features.index('y'))

    return data
