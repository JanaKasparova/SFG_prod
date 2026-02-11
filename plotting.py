# def make_plots(calibrated_data, results, output_dir):
#     """Generate all figures."""
#     pass


def make_plots(calibrated_data, results, output_dir):
    print(f"  Saving plots to {output_dir}")


import numpy as np
import matplotlib.pyplot as plt
from math import ceil, sqrt
from typing import Optional, Sequence


def plot_image_grid(
    images: np.ndarray,
    titles: Optional[Sequence[str]] = None,
    n_images: int = 6,
    suptitle: Optional[str] = None,
    cmap: str = "gray",
    figsize_scale: float = 5.5,
) -> None:
    """
        Plot a uniform grid of images with optional titles.

        The function automatically determines a near-square grid layout
        and ensures visually balanced spacing between images.

        Parameters
        ----------
        images : np.ndarray
            Image array of shape:
            - (N, H, W)      for grayscale images, or
            - (N, H, W, C)   for RGB/BGR images

        titles : sequence of str, optional
            Titles corresponding to each image. Length must be >= n_images.

        n_images : int, default=6
            Number of images to plot from the input array.

        suptitle : str, optional
            Figure-level title.

        cmap : str, default="gray"
            Colormap used for grayscale images.

        figsize_scale : float, default=5.5
            Controls overall figure size scaling.

        Raises
        ------
        ValueError
            If insufficient images are provided or array shape is invalid.
        """

    if images.ndim not in (3, 4):
        raise ValueError("Images must have shape (N,H,W) or (N,H,W,C)")

    if images.shape[0] < n_images:
        raise ValueError("Not enough images to plot")

    cols = ceil(sqrt(n_images))
    rows = ceil(n_images / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * figsize_scale, rows * figsize_scale),
        constrained_layout=True,
    )

    axes = np.atleast_1d(axes).ravel()

    for i, ax in enumerate(axes):
        if i >= n_images:
            ax.axis("off")
            continue

        img = images[i]

        if img.ndim == 2:
            ax.imshow(img, cmap=cmap, origin="lower")
        else:
            ax.imshow(img, origin="lower")

        # Default title: image index
        if titles is None:
            ax.set_title(f"Index {i}", fontsize=11)
        else:
            ax.set_title(titles[i], fontsize=11)

        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=18, y=1.02)

    plt.show()

