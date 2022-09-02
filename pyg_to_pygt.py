import numpy as np
import torch
import torch_geometric as pyg

from torch_geometric_temporal.signal import StaticGraphTemporalSignal


def merge_dynamic_with_static_features(static_features: np.array, dynamic_features: np.array) -> np.array:
    """
    Merges values of dynamic features with values of static features across a batch. The static features
    are broadcasted, instead of being copied, (due to staying unchanged throughout the batch) to save
    memory.

    static_feature: array of values of features that do not change throughout the spatial domain of the
    batch.
    batch: array of values of features that do change throughout the spatial domain of the batch.
    """
    return np.concatenate(
        (np.broadcast_to(static_features, (len(dynamic_features), *static_features.shape)),
         np.expand_dims(dynamic_features, axis=-1)),
        axis=-1)


def make_static_graph_temporal_signal(data: pyg.data.Data, dynamic_features: np.array) -> \
        StaticGraphTemporalSignal:
    """
    Creates a Pytorch Geometric Temporal StaticGraphTemporalSignal object from a Pytoroch Geometric data
    object and an array of values of features that vary across time.
    The 'targets' attribute for a given snapshot of the returned object are exactly the features of the
    subsequent snapshot.

    data: Pytorch Geometric data instance representing a graph and its static features.
    dynamic_features: array of values of features of the nodes of the graph in data that change through
    time.
    """

    edge_index = data.edge_index.type(torch.long)
    edge_weight = data.edge_attr.type(torch.float)

    features = merge_dynamic_with_static_features(
        static_features=data.x.numpy(), dynamic_features=dynamic_features)

    # Set targets to be the features of the subsequent snapshots
    targets = features[1:, :, -1]
    features = features[:-1]

    # The pos array is assumed not to change, so it may be broadcasted instead of being copied
    pos = data.pos.numpy()
    pos = np.broadcast_to(pos, (len(dynamic_features) - 1, *pos.shape))

    # additional features
    kwargs = {
        'pos': pos
    }

    return StaticGraphTemporalSignal(
        edge_index=edge_index,
        edge_weight=edge_weight,
        features=features,
        targets=targets,
        **kwargs,
    )