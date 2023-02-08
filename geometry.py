import torch


def homogenize_points(points: torch.Tensor) -> torch.Tensor:
    """Appends a "1" to the coordinates of a (batch of) points of dimension DIM.
    Args:
        points: points of shape (..., DIM)
    Returns:
        points_hom: points with appended "1" dimension.
    """
    points_hom = torch.cat([points, torch.tensor([1]).expand([*points.shape[:-1], 1]).to(points.device)], dim=-1)
    return points_hom


def homogenize_vecs(vectors: torch.Tensor):
    """Appends a "0" to the coordinates of a (batch of) vectors of dimension DIM.
    Args:
        vectors: vectors of shape (..., DIM)
    Returns:
        vectors_vec: vectors with appended "0" dimension.
    """
    vectors_vec = torch.cat([vectors, torch.tensor([0]).expand([*vectors.shape[:-1], 1]).to(vectors.device)], dim=-1)
    return vectors_vec


def transform_rigid(xyz_hom: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Apply a rigid-body transform to a (batch of) points / vectors.
    Args:
        xyz_hom: homogenized 3D points of shape (..., 4)
        T: rigid-body transform matrix of shape (..., 4, 4)
    Returns:
        xyz_trans: transformed points *in homogeneous coordinates*.
    """
    xyz_trans = torch.einsum("...ij,...j -> ...i", T, xyz_hom)
    return xyz_trans


def transform_cam2world(xyz_cam_hom: torch.Tensor, cam2world: torch.Tensor) -> torch.Tensor:
    """Transforms points from 3D world coordinates to 3D camera coordinates.
    Args:
        xyz_cam_hom: homogenized 3D points of shape (..., 4)
        cam2world: camera pose of shape (..., 4, 4)
    Returns:
        xyz_world: Homogenous points in world coordinates of shape (..., 4).
    """
    xyz_world = torch.einsum("...ij,...j -> ...i", cam2world, xyz_cam_hom)
    return xyz_world


def unproject(xy_pix: torch.Tensor, z: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Unproject (lift) 2D pixel coordinates x_pix and per-pixel z coordinate
    to 3D points in camera coordinates.
    Args:
        xy_pix: 2D pixel coordinates of shape (..., 2)
        z: per-pixel depth, defined as z coordinate of shape (..., 1)
        intrinscis: camera intrinsics of shape (..., 3, 3)
    Returns:
        xyz_cam: points in 3D camera coordinates.
    """
    xy_pix_hom = homogenize_points(xy_pix)
    xyz_cam = z * transform_rigid(xy_pix_hom, intrinsics.inverse())
    return xyz_cam


def get_unnormalized_cam_ray_directions(xy_pix: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Compute the unnormalized camera ray directions.
    Args:
        xy_pix: 2D pixel coordinates of shape (..., 2)
        intrinsics: Camera intrinscics of shape (..., 3, 3)
    Returns:
        torch.Tensor: Camera ray directions of shape (..., 2)
    """
    return unproject(xy_pix, torch.ones_like(xy_pix[..., :1], device=xy_pix.device), intrinsics=intrinsics)


def get_camera_ray_directions(xy_pix: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Compute the camera ray directions.
    Args:
        xy_pix: 2D pixel coordinates of shape (..., 2)
        intrinsics: Camera intrinscics of shape (..., 3, 3)
    Returns:
        torch.Tensor: Camera ray directions of shape (..., 2)
    """
    # The ray direction in camera space is the normalized unprojected point
    unnormalized_ray_directions = get_unnormalized_cam_ray_directions(xy_pix, intrinsics)
    return torch.nn.functional.normalize(unnormalized_ray_directions, dim=-1)
