from typing import Tuple, Dict

import torch
from torch import nn

from geometry import get_unnormalized_cam_ray_directions, homogenize_vecs, transform_cam2world


def get_world_rays(
    xy_pix: torch.Tensor,
    intrinsics: torch.Tensor,
    cam2world: torch.Tensor,
) -> torch.Tensor:
    """Generates camera rays in the world frame.

    Args:
        xy_pix: 2D pixel coordinates of shape (..., 2)
        intrinsics: Camera intrinscics of shape (..., 3, 3)
        cam2world: Camera pose of shape (..., 4, 4)

    Returns:
        cam_origins: The camera origins in the world frame of shape (..., 3)
        ray_world_directions: The ray directions in the world frame of shape (..., 3)
    """
    cam_origin_world = cam2world[..., :3, 3]

    # Get ray directions in cam coordinates
    ray_dirs_cam = get_unnormalized_cam_ray_directions(xy_pix, intrinsics)

    # Homogenize ray directions
    rd_cam_hom = homogenize_vecs(ray_dirs_cam)

    # Transform ray directions to world coordinates
    rd_world_hom = transform_cam2world(rd_cam_hom, cam2world)

    # Tile the ray origins to have the same shape as the ray directions.
    # Currently, ray origins have shape (batch, 3), while ray directions have shape (batch, num_rays, 3)
    cam_origin_world = cam_origin_world.unsqueeze(1).tile((1, rd_world_hom.shape[1], 1))

    # Return tuple of cam_origins, ray_world_directions
    return cam_origin_world, rd_world_hom[..., :3]


def sample_points_along_rays(
    near_depth: float,
    far_depth: float,
    num_samples: int,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
) -> torch.Tensor:
    """Returns 3D coordinates of points along the camera rays defined by ray_origin
    and ray_directions. Dept values are uniformly sampled between the near_depth and
    the far_depth.

    Parameters:
        near_depth: The depth at which we start sampling points.
        far_depth: The depth at which we stop sampling points.
        num_samples: The number of depth samples between near_depth and far_depth.
        ray_origins: Tensor of shape (batch_size, num_rays, 3). The origins of camera rays.
        ray_directions: Tensor of shape (batch_size, num_rays, 3). The directions of camera rays.

    Returns:
        pts: Tensor of shape (batch_size, num_rays, num_samples, 3). 3D points uniformly sampled between near_depth
            and far_depth
        z_vals: Tensor of shape (num_samples) of depths linearly spaced between near and far plane.
    """
    z_vals = torch.linspace(near_depth, far_depth, num_samples, device=ray_origins.device)

    ray_origins = ray_origins.unsqueeze(-2).tile((1, 1, num_samples, 1))
    pts = ray_origins + torch.einsum("...rv,s->...rsv", ray_directions, z_vals)

    return pts, z_vals


def volume_integral(
    z_vals: torch.Tensor, sigmas: torch.Tensor, radiances: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes the volume rendering integral.

    Parameters:
        z_vals: Tensor of shape (num_samples) of depths linearly spaced between near and far plane.
        sigmas: Tensor of shape (batch_size, num_rays, num_samples, 1). Densities of points along rays.
        radiances: tensor of shape (batch_size, num_rays, num_samples, 3). Emitted radiance of points along rays.

    Returns:
        rgb: Tensor of shape (batch_size, num_rays, 3). Total radiance observed by rays. Computed of weighted sum of
            radiances along rays.
        depth_map: Tensor of shape (batch_size, num_rays, 1). Expected depth of each ray. Computed as weighted sum of
            z_vals along rays.
        weights: The volume integral weights of shape (batch_size, num_rays, num_samples, 1).
    """
    # Compute the deltas in depth between the points.
    # The appended large value reflects a point at depth of infinity (Not using np.inf due to numerical issues)
    dists = torch.diff(z_vals, append=torch.tensor([1.0e3], device=z_vals.device))

    alpha = 1.0 - torch.exp(-torch.einsum("...rso,s->...rso", sigmas, dists))

    Ts = torch.cumprod(1 - alpha, dim=2)

    weights = alpha * Ts

    # Compute the pixel color as the weighted sum of the radiance values.
    rgb = torch.sum(weights * radiances, dim=2)

    # Compute the depths as the weighted sum of z_vals.
    depth_map = torch.einsum("...sv,s->...v", weights, z_vals)

    return rgb, depth_map, weights


class VolumeRenderer(nn.Module):
    def __init__(self, near, far, n_samples=32, white_back=True, rand=False):
        super().__init__()
        self.near = near
        self.far = far
        self.n_samples = n_samples
        self.white_back = white_back

    def unpack_input_dict(self, input_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c2w = input_dict["cam2world"]
        intrinsics = input_dict["intrinsics"]
        x_pix = input_dict["x_pix"]
        return c2w, intrinsics, x_pix

    def forward(
        self,
        input_dict: Dict[str, torch.Tensor],
        radiance_field: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Takes as inputs ray origins and directions - samples points along the
        rays and then calculates the volume rendering integral.

        Params:
            input_dict: Dictionary with keys 'cam2world', 'intrinsics', and 'x_pix'
            radiance_field: nn.Module instance of the radiance field we want to render.

        Returns:
            rgb: for each pixel coordinate x_pix, the color of the respective ray.
            depth_map: for each pixel coordinate x_pix, the depth of the respective ray.

        """
        cam2world, intrinsics, x_pix = self.unpack_input_dict(input_dict)
        batch_size, num_rays = x_pix.shape[0], x_pix.shape[1]

        # Compute the ray directions in world coordinates.
        ros, rds = get_world_rays(x_pix, intrinsics, cam2world)

        # Generate the points along rays and their depth values.
        pts, z_vals = sample_points_along_rays(
            self.near,
            self.far,
            self.n_samples,
            ros,
            rds,
        )

        pts = pts.reshape((batch_size, -1, 3))

        # Sample the radiance field with the points along the rays.
        sigma, rad = radiance_field(pts)

        sigma = sigma.reshape((batch_size, num_rays, self.n_samples, -1))
        rad = rad.reshape((batch_size, num_rays, self.n_samples, -1))

        # Compute pixel colors, depths, and weights via the volume integral.
        rgb, depth_map, weights = volume_integral(z_vals, sigma, rad)

        if self.white_back:
            accum = weights.sum(dim=-2)
            rgb = rgb + (1.0 - accum)

        return rgb, depth_map
