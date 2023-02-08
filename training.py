from typing import Iterator, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import to_gpu


def fit_inverse_graphics_representation(
    representation,
    renderer,
    data_iter: Iterator,
    img_resolution: Tuple[int, int, int],
    total_steps: int = 2001,
    lr: float = 1e-4,
    steps_til_summary: int = 100,
):
    optim = torch.optim.Adam(lr=lr, params=representation.parameters())

    losses = []
    for step in range(total_steps):
        # Get the next batch of data and move it to the GPU
        cam_params, ground_truth = next(data_iter)
        cam_params = to_gpu(cam_params)
        ground_truth = to_gpu(ground_truth)

        # Compute the MLP output for the given input data and compute the loss
        rgb, depth = renderer(cam_params, representation)

        loss = ((rgb - ground_truth) ** 2).mean()

        # Accumulate the losses so that we can plot them later
        losses.append(loss.detach().cpu().numpy())

        # Evaluation
        if not step % steps_til_summary:
            print(f"Step {step}: loss = {float(loss.detach().cpu()):.6f}")
            fig, axes = plt.subplots(1, 3, figsize=(18, 6), squeeze=False)
            axes[0, 0].imshow(rgb.cpu().view(*img_resolution).detach().numpy())
            axes[0, 0].set_title("Trained MLP")
            axes[0, 1].imshow(ground_truth.cpu().view(*img_resolution).detach().numpy())
            axes[0, 1].set_title("Ground Truth")

            depth = depth.cpu().view(*img_resolution[:2]).detach().numpy()
            axes[0, 2].imshow(depth, cmap="Greys")
            axes[0, 2].set_title("Depth")

            for i in range(3):
                axes[0, i].set_axis_off()

            plt.show()

        optim.zero_grad()
        loss.backward()
        optim.step()

    # Plot the loss
    fig, axes = plt.subplots(1, 1, figsize=(8, 8), squeeze=False)
    axes[0, 0].plot(np.array(losses))
    plt.show()
