import torch
import functools

from osmnx_utils import gps_to_meters


def dists(x: torch.tensor, y: torch.tensor, /) -> torch.tensor:
    """
    Compute pair-wise distances between two arrays of gps coordinates (longitude, latitude)
    in meters. 
    """
    # Choose the scale at the average latitude 
    scale = gps_to_meters(x[:, 1].mean())
    return torch.linalg.norm(
        (x - y.view(-1, 1, 2)) * torch.tensor(scale), 
        axis=-1, 
    )


def dists_min(fro: torch.tensor, to: torch.tensor) -> torch.tensor:
    """
    Compute minimum distances from all elements of fro to any element of to.
    """
    return dists(fro, to).min(axis=0)[0]


class RadMSELoss(torch.nn.MSELoss):
    """
    Compute the mean squared error of the predictions at coordinates within radius distance to the
    coordinates of the labels. Optionally, the mean squared error is weighted by the squared inverse of the
    minimum distance to any label.
    """

    def __init__(self, preds_gps: torch.Tensor, labels_gps: torch.Tensor, radius: float,
                 weighted: bool = True, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        """
        preds_gps: coordinates of the predictions.
        labels_gps: coordinates of the labels.
        radius: maximum distance from any label for a prediction to be included in the loss calculation.
        weighted: If true, the mean squared error will be weighted by the squared inverse of the minimum
        distance to any label.
        """
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)
        self.radius = radius
        self.dists = dists_min(fro=preds_gps, to=labels_gps)
        self.weight = torch.pow(self.dists, -2)[self.indices_in_radius] if weighted is True else 1

    @functools.cached_property
    def indices_in_radius(self) -> torch.Tensor:
        """
        Returns indices of elements in self.dists which are no greater than self.radius.
        """
        return torch.ones(self.dists.shape, dtype=torch.long) if self.radius is None \
            else (self.dists <= self.radius).nonzero().type(torch.long).squeeze()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss.
        """
        return torch.mean(
            (input[self.indices_in_radius] - target[self.indices_in_radius]).pow(2) * self.weight
        )