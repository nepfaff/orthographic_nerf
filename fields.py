from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def init_weights_normal(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity="relu", mode="fan_in")

        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class VoxelGrid(nn.Module):
    def __init__(self, resolution_per_dim, out_dim, mode="bilinear"):
        super().__init__()

        self.mode = mode
        self.grid = nn.Parameter(torch.zeros(1, out_dim, *resolution_per_dim))

    def forward(self, coordinate):
        """
        Args:
            coordinate: (batch_size, num_points, 3)

        Returns:
            values: (batch_size, num_points, out_dim)
        """
        coord = coordinate.unsqueeze(2).unsqueeze(3)  # Shape (batch_size, num_points, 1, 1, 3)

        values = F.grid_sample(self.grid, coord, self.mode)

        # Reshape & permute values to shape (batch_size, num_points, out_dim)
        values = values.squeeze(-1).squeeze(-1).permute(0, 2, 1)
        return values


class NeuralField(nn.Module):
    def __init__(self, feature_dim: int, out_dim: int, device: torch.device):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(3, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, out_dim),
        ).to(device)

        self.mlp.apply(init_weights_normal)

    def forward(self, coordinate):
        values = self.mlp(coordinate)

        return values


class HybridVoxelNeuralField(nn.Module):
    def __init__(self, resolution_per_dim: int, feature_dim: int, out_dim: int, device: torch.device, mode="bilinear"):
        super().__init__()

        self.mode = mode
        self.grid = nn.Parameter(torch.randn((1, feature_dim, *resolution_per_dim)) * 0.1)

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, out_dim),
        ).to(device)

        self.mlp.apply(init_weights_normal)

    def forward(self, coordinate):
        coord = coordinate.unsqueeze(2).unsqueeze(3)  # Shape (batch_size, num_points, 1, 1, 3)
        values = F.grid_sample(self.grid, coord, self.mode)

        # Permute the features from the grid_sample such that the feature dimension is the innermost dimension.
        values = values.squeeze(-1).squeeze(-1).permute((0, 2, 1))

        # Evaluate the mlp on the input features.
        values = self.mlp(values)

        return values


class HybridGroundPlanNeuralField(nn.Module):
    def __init__(self, resolution_per_dim: int, feature_dim: int, out_dim: int, device: torch.device, mode="bilinear"):
        super().__init__()

        self.mode = mode
        self.grid = nn.Parameter(torch.randn((1, feature_dim, *resolution_per_dim)) * 0.1)

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim + 1, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, out_dim),
        ).to(device)

        self.mlp.apply(init_weights_normal)

    def forward(self, coordinate: np.ndarray):
        """
        Args:
            coordinate: The sample coordinates of shape (batch_size, num_points, 3).

        Returns:
            sample_values: The field sample values at the coordinates.
        """
        # Coordinates (batch_size, num_points, 3)

        # Project coordinate onto xy-plane (dropping z-coordinate)
        xy = coordinate[..., :2]

        # Add dim to get (batch_size, num_points, 1, 2)
        xy = xy.unsqueeze(2)

        # Extract the z-coordinate
        z = coordinate[..., 2].expand(*xy.shape[:2]).unsqueeze(2)

        # Sample ground plan using the grid_sample function using the specified mode and query xy coordinates.
        features = F.grid_sample(self.grid, xy, self.mode)

        # Reshape and permute such that features have a shape of (batch_size, num_points, latent_dimension)
        features = features.squeeze(-1).permute(0, 2, 1)

        # Concatenate with z coordinate and infer the mlp on features and store it in variable values.
        concatenated_features = torch.cat((features, z), dim=-1)
        values = self.mlp(concatenated_features)

        return values


class RadianceField(nn.Module):
    def __init__(self, scene_rep_name: str, device: torch.device):
        super().__init__()

        if scene_rep_name == "VoxelGrid":
            self.scene_rep = VoxelGrid(resolution_per_dim=(200, 200, 200), out_dim=64).to(
                device
            )  # Results in about 16GB GPU memory
        elif scene_rep_name == "NeuralField":
            self.scene_rep = NeuralField(feature_dim=64, out_dim=64, device=device)
        elif scene_rep_name == "HybridVoxelNeuralField":
            self.scene_rep = HybridVoxelNeuralField(
                resolution_per_dim=(64, 64, 64), feature_dim=64, out_dim=64, device=device
            ).to(device)
        elif scene_rep_name == "HybridGroundPlanNeuralField":
            self.scene_rep = HybridGroundPlanNeuralField(
                resolution_per_dim=(64, 64), feature_dim=64, out_dim=64, device=device
            ).to(device)

        self.sigma = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU(),
        ).to(device)
        self.sigma.apply(init_weights_normal)

        self.radiance = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.ReLU(),
        ).to(device)
        self.radiance.apply(init_weights_normal)

    def forward(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Queries the representation for the density and color values.

        Args:
            xyz: The sample coordinates of shape (batch_size, num_points, 3).

        Returns:
            sigma: The sigma values of shape (batch_size, num_points, 1).
            rad: The radiance values of shape (batch_size, num_points, 3).
        """
        features = self.scene_rep(xyz)

        sigma = self.sigma(features)
        rad = self.radiance(features)

        return sigma, rad
