import torch
import numpy as np


def get_opencv_pixel_coordinates(y_resolution: int, x_resolution: int, device: torch.device):
    """For an image with y_resolution and x_resolution, return a tensor of pixel coordinates
    normalized to lie in [0, 1], with the origin (0, 0) in the top left corner,
    the x-axis pointing right, the y-axis pointing down, and the bottom right corner
    being at (1, 1).

    Returns:
        xy_pix: a meshgrid of values from [0, 1] of shape (y_resolution, x_resolution, 2)
    """
    i, j = torch.meshgrid(
        torch.linspace(0, 1, steps=x_resolution, device=device), torch.linspace(0, 1, steps=y_resolution, device=device)
    )

    xy_pix = torch.stack([i.float(), j.float()], dim=-1).permute(1, 0, 2)
    return xy_pix


def diff_rendering_dataset(images: np.ndarray, cam2world: np.ndarray, device: torch.device):
    """Generates an iterator from a tensor of images and a tensor of cam2world matrices.
    Yields one random image per iteration.
    """
    image_resolution = images.shape[1:3]
    intrinsics = torch.tensor([[0.7, 0.0, 0.5], [0.0, 0.7, 0.5], [0.0, 0.0, 1.0]]).to(images.device)

    x_pix = get_opencv_pixel_coordinates(*image_resolution, device=device)
    x_pix = x_pix.reshape(1, -1, 2).to(images.device)

    while True:
        idx = np.random.randint(low=0, high=len(cam2world))
        c2w = cam2world[idx : idx + 1]
        ground_truth = images[idx : idx + 1]
        model_input = {"cam2world": c2w, "intrinsics": intrinsics, "x_pix": x_pix}
        yield model_input, ground_truth[..., :3].view(-1, 3)
