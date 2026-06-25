import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from math import ceil, sqrt
from typing import Optional, Sequence
from astropy.io import fits
from processing import crop_images
import os
import textwrap
from analysis import *


def plot_single_series(
        data: np.ndarray,
        x_series: np.ndarray = None,
        plot_graph: bool = True,
        title: str = "Data Profile",
        x_label_custom: str = None,
        y_label: str = "Value",
        line_color: str = "#1f77b4",
        save_name: str = None,
        logger=None,
        **fig_kwargs
):
    """
    Plots a single 1D data array with an option to supply a custom X-axis or time series.

    Parameters:
    -----------
    data : np.ndarray
        The 1D data array containing values to plot on the Y-axis.
    x_series : np.ndarray, optional
        A 1D array containing timestamps, frame indexes, or locations for the X-axis.
    plot_graph : bool, default True
        If False, runs silently in the background (useful when you only want to save).
    title : str, default "Data Profile"
        The main title of the plot.
    x_label_custom : str, optional
        Custom label for the X-axis. Defaults to "Time" if x_series is provided, else "Index".
    y_label : str, default "Value"
        Label text for the Y-axis.
    line_color : str, default "#1f77b4" (Classic Blue)
        Hex string or named color for the plot line.
    save_name : str, optional
        Filename to save the plot directly inside the 'Plots' directory.
    """
    if logger:
        logger.info(f"Preparing single series plot for array of size: {len(data)}")

    # 1. Resolve X-Axis Data and Labeling
    if x_series is not None:
        x_data = x_series
        x_label = x_label_custom if x_label_custom else "Time"
        if len(x_data) != len(data):
            raise ValueError(f"X-series length ({len(x_data)}) must match the data array length ({len(data)}).")
    else:
        x_data = np.arange(len(data))
        x_label = x_label_custom if x_label_custom else "Index"

    # 2. Setup Figure Layout
    if plot_graph or save_name is not None:
        figsize = fig_kwargs.get("figsize", (10, 5))
        fig, ax = plt.subplots(figsize=figsize)

        # Plot line with a subtle marker for data points
        ax.plot(x_data, data, color=line_color, linewidth=2, linestyle="-", marker="o", markersize=3, alpha=0.8)

        # Style layout
        ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)

        # ---- Process Saving Block ----
        if save_name is not None:
            output_dir = "Plots"
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, save_name)

            plt.savefig(save_path, bbox_inches='tight', dpi=300)

            if logger is not None:
                logger.info(f"Plot successfully saved to: {save_path}")
            else:
                print(f"Plot successfully saved to: {save_path}")

        # ---- Process Display Block ----
        if plot_graph:
            plt.show()
        else:
            plt.close(fig)  # Free system memory cleanly if running silently

def plot_image_grid(
    images: np.ndarray,
    titles: Optional[Sequence[str]] = None,
    n_images: int = 6,
    suptitle: Optional[str] = None,
    cmap: str = "gray",
    figsize_scale: float = 5.5,
    save_name: str = None,
    logger=None
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

    # ---- Save Figure Setup (Paste right before plt.show()) ----
    # Expects `save_name` (str or None) passed into the function's arguments
    if 'save_name' in locals() or 'save_name' in globals():
        if save_name is not None:
            import os
            output_dir = "Plots"

            # Create the directory safely if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Combine directory and filename
            save_path = os.path.join(output_dir, save_name)

            # Save the figure (bbox_inches='tight' prevents labels from cutting off)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

            if 'logger' in locals() and logger is not None:
                logger.info(f"Plot successfully saved to: {save_path}")
            else:
                print(f"Plot successfully saved to: {save_path}")
    # -----------------------------------------------------------

    plt.show()


def plot_single_image(
        img: np.ndarray,
        plot_graph: bool = True,
        title: str = "Image View",
        x_label: str = None,  # Defaults to None so it doesn't clutter the image
        y_label: str = None,  # Defaults to None so it doesn't clutter the image
        cmap: str = "gray",
        origin: str = "lower",
        vmin: float = None,
        vmax: float = None,
        colorbar: bool = True,
        colorbar_label: str = "Intensity (ADU)",
        logger=None,
        save_name=None,
        **fig_kwargs
):
    """
    Plots a single 2D image matrix with clean, clutter-free axes layout.
    """
    if logger:
        logger.info(f"Rendering single image plot. Matrix shape: {img.shape}")

    # Data shape validation safeguard
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]  # Squeeze out leading single-frame dim if passed incorrectly
    elif img.ndim != 2 and not (img.ndim == 3 and img.shape[-1] in [3, 4]):
        raise ValueError(f"Expected a 2D array or 3D RGB array. Got matrix shape: {img.shape}")

    # Build and map the visualization figure layout
    if plot_graph or save_name is not None:
        figsize = fig_kwargs.get("figsize", (8, 6))
        fig, ax = plt.subplots(figsize=figsize)

        # Draw the target frame data
        im = ax.imshow(img, cmap=cmap, origin=origin, vmin=vmin, vmax=vmax)

        # Append colorbar scales for single-channel datasets (skip for RGB images)
        if colorbar and not (img.ndim == 3 and img.shape[-1] in [3, 4]):
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)

        # Apply the title
        if title:
            ax.set_title(title, fontsize=13, fontweight='bold', pad=12)

        # ONLY add text labels if explicitly passed by the user
        if x_label:
            ax.set_xlabel(x_label, fontsize=11)
        if y_label:
            ax.set_ylabel(y_label, fontsize=11)

        # ---- Process Saving Block ----
        if save_name is not None:
            output_dir = "Plots"
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, save_name)

            plt.savefig(save_path, bbox_inches='tight', dpi=300)

            if logger is not None:
                logger.info(f"Image saved successfully to: {save_path}")
            else:
                print(f"Image saved successfully to: {save_path}")

        # ---- Process Display Block ----
        if plot_graph:
            plt.show()
        else:
            plt.close(fig)  # Prevent background engine memory leaks

def save_fits_image(img: np.ndarray,
                    filename: str,
                    output_dir: str|None = None,
                    logger=None,
                    use_absolute_path: bool = True) -> None:
    '''Saves the fits file into output_dir.

    Parameters
    ----------
    img : np.ndarray
        Image array to save
    output_dir : str
        Output directory path (relative or absolute)
    filename : str
        Filename without extension
    logger : logging.Logger, optional
        Logger instance for logging the save operation
    use_absolute_path : bool, default=True
        If True, converts path to absolute. If False, keeps as-is (relative).
    '''
    # Convert to absolute path if requested
    if output_dir is None:
        output_dir = os.getcwd()
    if use_absolute_path:
        output_dir = os.path.abspath(output_dir)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create a FITS HDU from the image
    hdu = fits.PrimaryHDU(data=img)

    # Create a HDU list and write to file
    hdul = fits.HDUList([hdu])

    # Generate filepath with filename
    filepath = os.path.join(output_dir, f'{filename}')

    # Write to file, overwriting if it exists
    hdul.writeto(filepath, overwrite=True)

    if logger:
        logger.info(f"Saved FITS image to {filepath}")


def plot_image_with_crop(
        image,
        xmin,
        xmax,
        ymin,
        ymax,
        title_full="Full Image (with Crop Region)",
        title_crop="Cropped Region Zoom",
        save_name: str = None,
        logger=None,
        **imshow_kwargs
):
    """
    Plots the full image with a highlighted bounding box and the
    corresponding cropped region side-by-side in a single figure.

    Parameters:
    -----------
    image : ndarray
        The original input image.
    xmin, xmax, ymin, ymax : int
        Bounding box cropping coordinates.
    title_full : str
        Title for the left subplot.
    title_crop : str
        Title for the right subplot.
    logger : logging.Logger, optional
        Logger instance to output execution traces.
    **imshow_kwargs : dict
        Additional arguments passed directly to plt.imshow() for both plots
        (e.g., cmap='gray', vmin=0, vmax=255, origin='lower').
    """
    # Log the operation details if a logger is provided
    if logger is not None:
        logger.info(
            "Plotting image with crop box (xmin=%d, xmax=%d, ymin=%d, ymax=%d)",
            xmin, xmax, ymin, ymax
        )

    # 1. Get the cropped image using your premade function
    cropped_image = crop_images(image, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, logger=logger)

    # Fallback default configuration if not provided in kwargs
    if 'origin' not in imshow_kwargs:
        imshow_kwargs['origin'] = 'lower'
    if 'cmap' not in imshow_kwargs:
        imshow_kwargs['cmap'] = 'gray'

    # 2. Setup a single figure with 2 subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(16, 7),
        gridspec_kw={'width_ratios': [1.2, 1]}
    )

    # --- Left Plot: Full Image with Bounding Box ---
    ax1.imshow(image, **imshow_kwargs)

    rect_width = xmax - xmin
    rect_height = ymax - ymin

    rect = Rectangle(
        (xmin, ymin),
        rect_width,
        rect_height,
        edgecolor='#ff3333',  # Vibrant red
        facecolor='none',
        linewidth=2,
        linestyle='-'
    )
    ax1.add_patch(rect)
    ax1.set_title(title_full, fontsize=14, fontweight='bold', pad=10)
    ax1.grid(False)

    # --- Right Plot: Cropped Zoom-In ---
    ax2.imshow(cropped_image, **imshow_kwargs)

    # Visual continuity: matching red borders around the crop frame
    for spine in ax2.spines.values():
        spine.set_edgecolor('#ff3333')
        spine.set_linewidth(2)

    ax2.set_title(title_crop, fontsize=14, fontweight='bold', pad=10)
    ax2.grid(False)

    # ---- Save Figure Setup (Paste right before plt.show()) ----
    # Expects `save_name` (str or None) passed into the function's arguments
    if 'save_name' in locals() or 'save_name' in globals():
        if save_name is not None:
            import os
            output_dir = "Plots"

            # Create the directory safely if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Combine directory and filename
            save_path = os.path.join(output_dir, save_name)

            # Save the figure (bbox_inches='tight' prevents labels from cutting off)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

            if 'logger' in locals() and logger is not None:
                logger.info(f"Plot successfully saved to: {save_path}")
            else:
                print(f"Plot successfully saved to: {save_path}")
    # -----------------------------------------------------------

    # Clean up layout
    plt.tight_layout()
    plt.show()


def plot_dark_histograms(
    stats: dict,
    logger=None,
    title_all="All Points (with hotpoints)",
    title_no_hot="Without Hotpoints",
    title_only_hot="Only Hotpoints",
    title_only_clean="Only Clean Points",
    save_name=None,
    **fig_kwargs
):
    """
    Generates a 2x2 grid of histograms analyzing the distribution of hot and clean pixels.
    """
    if logger:
        logger.info("Generating dark frame histograms...")

    # Extract needed variables for clean code
    df, hp, cp = stats["data_flat"], stats["hot_points"], stats["clean_points"]
    mean, std, N = stats["exp_value"], stats["stdev"], stats["N"]

    # Calculate dynamic logarithmic bins safely
    bins_global = np.logspace(np.log10(np.min(df[df > 0])), np.log10(np.max(df)), 100)
    bins_hot = np.logspace(np.log10(np.min(hp[hp > 0])), np.log10(np.max(hp)), 100) if len(hp) > 0 else 100
    bins_clean = np.logspace(np.log10(np.min(cp[cp > 0])), np.log10(np.max(cp)), 100)

    # Setup figure
    figsize = fig_kwargs.get("figsize", (16, 12))
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Helper function to avoid repeating axis formatting
    def format_ax(ax, title, xlabel="Value", ylabel="Frequency"):
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.6)

    # ---- (0,0): All points ----
    axes[0, 0].hist(df, bins=bins_global, edgecolor="black", alpha=0.7)
    axes[0, 0].axvline(mean, color="red", linestyle="--", linewidth=2, label="Mean")
    axes[0, 0].axvline(mean + N * std, color="orange", linestyle="--", linewidth=2, label=f"Mean + {N}σ")
    format_ax(axes[0, 0], title_all)
    axes[0, 0].legend()

    # ---- (0,1): Without hot points ----
    axes[0, 1].hist(cp, bins=bins_global, edgecolor="black", alpha=0.7)
    axes[0, 1].axvline(mean, color="red", linestyle="--", linewidth=2, label="Mean")
    axes[0, 1].axvline(mean + N * std, color="orange", linestyle="--", linewidth=2, label=f"Mean + {N}σ")
    format_ax(axes[0, 1], title_no_hot)
    axes[0, 1].legend()

    # ---- (1,0): Only hot points ----
    if len(hp) > 0:
        axes[1, 0].hist(hp, bins=bins_hot, edgecolor="black", alpha=0.7, color="orange")
    format_ax(axes[1, 0], title_only_hot)

    # ---- (1,1): Only clean points ----
    axes[1, 1].hist(cp, bins=bins_clean, edgecolor="black", alpha=0.7, color="green")
    format_ax(axes[1, 1], title_only_clean)

    # ---- Save Figure Setup (Paste right before plt.show()) ----
    # Expects `save_name` (str or None) passed into the function's arguments
    if 'save_name' in locals() or 'save_name' in globals():
        if save_name is not None:
            import os
            output_dir = "Plots"

            # Create the directory safely if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Combine directory and filename
            save_path = os.path.join(output_dir, save_name)

            # Save the figure (bbox_inches='tight' prevents labels from cutting off)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

            if 'logger' in locals() and logger is not None:
                logger.info(f"Plot successfully saved to: {save_path}")
            else:
                print(f"Plot successfully saved to: {save_path}")
    # -----------------------------------------------------------

    plt.tight_layout()
    plt.show()

    if logger:
        logger.info("Histograms plotted successfully.")


def plot_hot_pixel_map(
    stats: dict,
    logger=None,
    title_clipped="Clipped Master Dark (No Hot Points)",
    title_highlighted="Master Dark with Hot Pixels Highlighted",
    save_name=None,
    **fig_kwargs
):
    """
    Plots the master dark image side-by-side: one clipped to the threshold,
    and one highlighting the exact locations of the hot pixels.
    """
    if logger:
        logger.info("Generating hot pixel spatial maps...")

    img = stats["image"]
    threshold, stdev = stats["threshold"], stats["stdev"]
    hy, hx, hv = stats["hot_y"], stats["hot_x"], stats["hot_values"]

    # Calculate dynamic scatter properties
    sizes = 20 + 5 * (hv - threshold) / stdev
    colors = (hv - threshold) / stdev

    figsize = fig_kwargs.get("figsize", (16, 7))
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ---- Left: Master dark clipped to threshold ----
    im0 = axes[0].imshow(img, cmap="gray", origin="lower", vmax=threshold)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Pixel value")
    axes[0].set_title(title_clipped, fontsize=14, fontweight='bold')
    axes[0].set_xlabel("X pixel")
    axes[0].set_ylabel("Y pixel")

    # ---- Right: Master dark + hot pixel scatter ----
    im1 = axes[1].imshow(img, cmap="gray", origin="lower")
    if len(hx) > 0:
        scatter = axes[1].scatter(hx, hy, s=sizes, c=colors, cmap="inferno",
                                  alpha=0.7, edgecolor="white", linewidth=0.5)
        fig.colorbar(scatter, ax=axes[1], fraction=0.046, pad=0.04, label="Hotness (σ above threshold)")

    axes[1].set_title(title_highlighted, fontsize=14, fontweight='bold')
    axes[1].set_xlabel("X pixel")
    axes[1].set_ylabel("Y pixel")

    # ---- Save Figure Setup (Paste right before plt.show()) ----
    # Expects `save_name` (str or None) passed into the function's arguments
    if 'save_name' in locals() or 'save_name' in globals():
        if save_name is not None:
            import os
            output_dir = "Plots"

            # Create the directory safely if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Combine directory and filename
            save_path = os.path.join(output_dir, save_name)

            # Save the figure (bbox_inches='tight' prevents labels from cutting off)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

            if 'logger' in locals() and logger is not None:
                logger.info(f"Plot successfully saved to: {save_path}")
            else:
                print(f"Plot successfully saved to: {save_path}")
    # -----------------------------------------------------------

    plt.tight_layout()
    plt.show()

    if logger:
        logger.info("Spatial maps plotted successfully.")


def print_dark_analysis_table(
        stats: dict,
        full_precision: bool = False,
        plot_table: bool = False,
        title_plot: str = "Dark Frame Analysis Summary",
        max_char_width: int = 45,  # Maximum characters allowed per line in a cell
        save_name: str = None,
        logger=None
):
    """
    Generates a cleanly formatted summary table from the dark frame analysis.
    Uses spaces as thousands separators for large numbers (e.g., 1 000 000).
    """
    if logger:
        logger.info("Generating summary table data...")

    # 1. Format the Raw Values
    def fmt(value, is_count=False):
        if is_count:
            # Replaces standard commas with spaces for large integer counts
            return f"{int(value):,}".replace(",", " ")
        if full_precision:
            return str(value)
        if abs(value) < 0.01:
            return f"{value:.6f}"
        return f"{value:.3f}"

    total_pixels = len(stats["data_flat"])
    num_hot = len(stats["hot_points"])
    num_clean = len(stats["clean_points"])
    pct_hot = (num_hot / total_pixels) * 100
    min_val = np.min(stats["data_flat"])
    max_val = np.max(stats["data_flat"])

    headers = ["Metric / Parameter", "Value", "Description"]

    # Raw rows for clean tracking
    raw_rows = [
        ["Sigma Threshold (N)", fmt(stats["N"]), "Multiplier used to define a hot pixel"],
        ["Global Mean (μ)", fmt(stats["exp_value"]), "Average baseline dark current value"],
        ["Global Std Dev (σ)", fmt(stats["stdev"]), "Standard deviation / noise floor"],
        ["Hot Threshold", fmt(stats["threshold"]), "Cutoff value (Mean + N*Std)"],
        ["Total Pixels", fmt(total_pixels, is_count=True), "Total array size analyzed"],
        ["Clean Pixels", fmt(num_clean, is_count=True), "Pixels remaining below hot threshold"],
        ["Hot Pixels Identified", fmt(num_hot, is_count=True), f"Pixels strictly > Mean + {stats['N']}σ"],
        ["Hot Pixel Ratio", f"{pct_hot:.4f}%" if not full_precision else f"{pct_hot}%",
         "Percentage of defective pixels on sensor"],
        ["Min Pixel Intensity", fmt(min_val), "Lowest recorded ADU value in dark frame"],
        ["Max Pixel Intensity", fmt(max_val), "Highest recorded ADU value (saturated hot pixel)"]
    ]

    # --- Word Wrapping Layer ---
    wrapped_rows = []
    for row in raw_rows:
        wrapped_rows.append([
            "\n".join(textwrap.wrap(str(row[0]), width=max_char_width)),
            "\n".join(textwrap.wrap(str(row[1]), width=max_char_width)),
            "\n".join(textwrap.wrap(str(row[2]), width=max_char_width))
        ])

    # 2. Text Table Output Logic
    def get_max_line_width(cell_str):
        lines = cell_str.split('\n')
        return max(len(line) for line in lines)

    w_col0 = max(max(get_max_line_width(row[0]) for row in wrapped_rows), len(headers[0]))
    w_col1 = max(max(get_max_line_width(row[1]) for row in wrapped_rows), len(headers[1]))
    w_col2 = max(max(get_max_line_width(row[2]) for row in wrapped_rows), len(headers[2]))

    border = f"+{'-' * (w_col0 + 2)}+{'-' * (w_col1 + 2)}+{'-' * (w_col2 + 2)}+"
    text_lines = [border, f"| {headers[0].ljust(w_col0)} | {headers[1].ljust(w_col1)} | {headers[2].ljust(w_col2)} |",
                  border]

    for i, row in enumerate(wrapped_rows):
        r0_lines, r1_lines, r2_lines = row[0].split('\n'), row[1].split('\n'), row[2].split('\n')
        max_lines = max(len(r0_lines), len(r1_lines), len(r2_lines))

        for line_idx in range(max_lines):
            l0 = r0_lines[line_idx] if line_idx < len(r0_lines) else ""
            l1 = r1_lines[line_idx] if line_idx < len(r1_lines) else ""
            l2 = r2_lines[line_idx] if line_idx < len(r2_lines) else ""
            text_lines.append(f"| {l0.ljust(w_col0)} | {l1.ljust(w_col1)} | {l2.ljust(w_col2)} |")

        if i in [3, 7]:
            text_lines.append(f"|+{'-' * (w_col0 + 2)}+{'-' * (w_col1 + 2)}+{'-' * (w_col2 + 2)}+|")

    text_lines.append(border)
    table_text = "\n" + "\n".join(text_lines)

    if logger:
        logger.info(table_text)
    else:
        print(table_text)

    # 3. Visual Matplotlib Table Handling (Triggered if plotting OR saving)
    if plot_table or save_name is not None:
        if logger:
            logger.info("Generating visual table image canvas...")

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.axis('tight')
        ax.axis('off')

        mpl_table = ax.table(
            cellText=wrapped_rows,
            colLabels=headers,
            loc='center',
            cellLoc='left'
        )

        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(10)
        mpl_table.scale(1.2, 2.0)  # Generous cell height padding

        # Styling headers and rows
        for (row_idx, col_idx), cell in mpl_table.get_celld().items():
            if row_idx == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#2c3e50')
            else:
                if row_idx % 2 == 0:
                    cell.set_facecolor('#f8f9fa')
                else:
                    cell.set_facecolor('#ffffff')

        plt.title(title_plot, fontsize=14, fontweight='bold', pad=20)

        # ---- Process Saving Block ----
        if save_name is not None:
            output_dir = "Plots"
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, save_name)

            plt.savefig(save_path, bbox_inches='tight', dpi=300)

            if logger is not None:
                logger.info(f"Plot successfully saved to: {save_path}")
            else:
                print(f"Plot successfully saved to: {save_path}")

        # ---- Process Display Block ----
        if plot_table:
            plt.show()
        else:
            plt.close(fig)  # Closes backend engine silently if not displaying on screen

    return table_text


def analyze_and_plot_rect(
        imgs: np.ndarray,
        xmin: int,
        xmax: int,
        ymin: int,
        ymax: int,
        num_plots: int = None,
        vmax: float = 50000,
        save_name: str = None,
        logger=None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Crops images to a specific ROI rectangle, calculates the mean and standard
    deviation values within that box, and uniformly plots N sample frames.

    Parameters:
    -----------
    imgs : np.ndarray
        Array stack of shape (N, H, W) or (N, H, W, C).
    xmin, xmax, ymin, ymax : int
        Bounding spatial coordinates for the ROI rectangle.
    num_plots : int, optional
        Number of images to uniformly select and plot on screen.
    vmax : float, default 50000
        Maximum display scaling cutoff for display normalization.
    save_name : str, optional
        Filename to save the final plot in the 'Plots' folder.
    """
    if logger:
        logger.info("Initializing ROI analysis workflow...")

    # 1. Execute the spatial slice crop
    images_cropped = crop_images(imgs, xmin, ymin, xmax, ymax, logger=logger)

    # 2. Extract statistics inside the bounding box array space
    # Handles dynamic dimensional tracking if analyzing a single frame vs a stack
    if imgs.ndim > 2:
        mean_values = np.mean(images_cropped, axis=(1, 2))
        std_values = np.std(images_cropped, axis=(1, 2))
    else:
        mean_values = np.array([np.mean(images_cropped)])
        std_values = np.array([np.std(images_cropped)])

    if logger:
        logger.info(f"ROI Stats Calculated. Mean range: [{np.min(mean_values):.1f}, {np.max(mean_values):.1f}]")

    # 3. Process Uniform Grid Graphing Logic
    if num_plots is not None and num_plots > 0:
        num_total_imgs = len(imgs) if imgs.ndim > 2 else 1

        # Calculate evenly spaced indices across the frame index sequence
        plot_indices = np.linspace(0, num_total_imgs - 1, num_plots, dtype=int)
        plot_indices = np.unique(plot_indices)  # Guardrail against duplicate selections
        actual_plot_count = len(plot_indices)

        # Build dynamic grid coordinates
        cols = 2 if actual_plot_count >= 2 else 1
        rows = int(np.ceil(actual_plot_count / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        # Ensure flat access arrays even for a single column plot grid configuration
        ax_flat = axes.ravel() if actual_plot_count > 1 else [axes]

        for idx, img_idx in enumerate(plot_indices):
            ax = ax_flat[idx]
            img_to_show = imgs[img_idx] if imgs.ndim > 2 else imgs

            # Draw baseline full framework layer
            ax.imshow(img_to_show, cmap="gray", origin="lower", vmax=vmax)
            ax.set_title(f"Image Index: {img_idx} (Full Field View)")

            # Anchor bounding visual tracking rectangle
            rect = Rectangle(
                xy=(xmin, ymin),
                width=(xmax - xmin),
                height=(ymax - ymin),
                color="red",
                fill=False,
                linewidth=1.5
            )
            ax.add_patch(rect)

        # Prune any unused empty window frames from grid structures
        for extra_ax in ax_flat[actual_plot_count:]:
            fig.delaxes(extra_ax)

        plt.tight_layout()

        # Save logic execution guardrail
        if save_name is not None:
            output_dir = "Plots"
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, save_name)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            if logger:
                logger.info(f"ROI verification grid saved to: {save_path}")

        plt.show()

    return mean_values, std_values


def plot_stats(
        means: np.ndarray,
        stds: np.ndarray,
        time_series: np.ndarray = None,
        plot_graphs: bool = True,
        title_mean: str = "Mean Pixel Value Profile",
        title_std: str = "Standard Deviation (Noise Profile)",
        x_label_custom: str = None,
        save_name: str = None,
        logger=None,
        **fig_kwargs
):
    """
    Plots mean and standard deviation arrays side-by-side.
    Supports frame tracking indices or custom time-series arrays as the X-axis.
    """
    if logger:
        logger.info("Generating side-by-side ROI metrics graphs...")

    # 1. Resolve X-Axis Data and Labeling
    if time_series is not None:
        x_data = time_series
        x_label = x_label_custom if x_label_custom else "Time"
        if len(x_data) != len(means):
            raise ValueError(f"Time series length ({len(x_data)}) must match means/stds length ({len(means)}).")
    else:
        x_data = np.arange(len(means))
        x_label = x_label_custom if x_label_custom else "Frame Index"

    # 2. Visual Layout Processing (Triggered if plotting OR saving)
    if plot_graphs or save_name is not None:
        figsize = fig_kwargs.get("figsize", (15, 5))
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # ---- Left Plot: Mean Intensity Over Time ----
        axes[0].plot(x_data, means, color="#1f77b4", linewidth=2, linestyle="-", marker="o", markersize=3, alpha=0.8)
        axes[0].set_title(title_mean, fontsize=12, fontweight='bold', pad=10)
        axes[0].set_xlabel(x_label, fontsize=10)
        axes[0].set_ylabel("Mean Intensity (ADU)", fontsize=10)
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # ---- Right Plot: Standard Deviation Over Time ----
        axes[1].plot(x_data, stds, color="#d62728", linewidth=2, linestyle="-", marker="s", markersize=3, alpha=0.8)
        axes[1].set_title(title_std, fontsize=12, fontweight='bold', pad=10)
        axes[1].set_xlabel(x_label, fontsize=10)
        axes[1].set_ylabel("Standard Deviation (σ)", fontsize=10)
        axes[1].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()

        # ---- Process Saving Block ----
        if save_name is not None:
            output_dir = "Plots"
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, save_name)

            plt.savefig(save_path, bbox_inches='tight', dpi=300)

            if logger is not None:
                logger.info(f"Statistics plots successfully saved to: {save_path}")
            else:
                print(f"Statistics plots successfully saved to: {save_path}")

        # ---- Process Display Block ----
        if plot_graphs:
            plt.show()
        else:
            plt.close(fig)  # Safely close figure object to free system memory if silent


def analyze_and_plot_circ(
        imgs: np.ndarray,
        xc: float,
        yc: float,
        r: float,
        num_plots: int = None,
        plot_graphs: bool = True,
        vmax: float = None,
        save_name: str = None,
        logger=None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Isolates a circular Region of Interest (ROI), calculates the mean and
    standard deviation strictly within that circle across all frames, and
    uniformly plots N sample frames with a circular boundary overlay.

    Parameters:
    -----------
    imgs : np.ndarray
        Image stack of shape (N, H, W) or single image of shape (H, W).
    xc, yc : float
        Center coordinates of the circle mask.
    r : float
        Radius of the circle mask.
    num_plots : int, optional
        Number of frames to uniformly select and plot.
    plot_graphs : bool, default True
        If False, suppresses screen display (useful for silent automation).
    vmax : float, optional
        Maximum scaling intensity limit for visual normalization.
    save_name : str, optional
        Filename to save the generated figure inside the 'Plots' directory.
    """
    if logger:
        logger.info(f"Initializing circular ROI mask calculation (Center X={xc}, Y={yc}, R={r})...")

    # 1. Determine dimensions and normalize shape
    if imgs.ndim == 2:
        H, W = imgs.shape
        num_frames = 1
        working_stack = imgs[np.newaxis, ...]  # Temporarily expand to 3D for loop uniformity
    else:
        num_frames, H, W = imgs.shape[:3]
        working_stack = imgs

    # 2. Generate the coordinate mesh grid and boolean circular mask
    y_indices, x_indices = np.ogrid[:H, :W]
    # Circle Equation: (x - xc)^2 + (y - yc)^2 <= r^2
    circle_mask = (x_indices - xc) ** 2 + (y_indices - yc) ** 2 <= r ** 2

    # 3. Iterate through frames and extract masked metrics
    mean_values = np.zeros(num_frames)
    std_values = np.zeros(num_frames)

    for i in range(num_frames):
        frame = working_stack[i]
        # If the image is RGB/Color, handle the spatial mask across the color channels
        if frame.ndim == 3:
            pixel_pool = frame[circle_mask, :]
            mean_values[i] = np.mean(pixel_pool)  # Global average across pool
            std_values[i] = np.std(pixel_pool)
        else:
            pixel_pool = frame[circle_mask]
            mean_values[i] = np.mean(pixel_pool)
            std_values[i] = np.std(pixel_pool)

    if logger:
        logger.info(f"Circular mask processed successfully over {num_frames} frame(s).")

    # 4. Uniform Grid Plotting and Saving Layout
    if (num_plots is not None and num_plots > 0) or save_name is not None:
        # Calculate evenly spaced array indices
        plot_indices = np.linspace(0, num_frames - 1, num_plots if num_plots else 4, dtype=int)
        plot_indices = np.unique(plot_indices)
        actual_plot_count = len(plot_indices)

        # Handle grid configuration shapes
        cols = 2 if actual_plot_count >= 2 else 1
        rows = int(np.ceil(actual_plot_count / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
        ax_flat = axes.ravel() if actual_plot_count > 1 else [axes]

        for idx, img_idx in enumerate(plot_indices):
            ax = ax_flat[idx]
            img_to_show = working_stack[img_idx]

            # Draw standard grayscale image
            ax.imshow(img_to_show, cmap="gray", origin="lower", vmax=vmax)
            ax.set_title(f"Frame Index: {img_idx}", fontsize=11, fontweight='bold')

            # Create a completely unique circle patch instance for this subplot
            visual_circle = Circle((xc, yc), r, color="red", fill=False, linewidth=1.5)
            ax.add_patch(visual_circle)

        # Clear out empty axes if any remain
        for extra_ax in ax_flat[actual_plot_count:]:
            fig.delaxes(extra_ax)

        plt.tight_layout()

        # Save to disk logic block
        if save_name is not None:
            output_dir = "Plots"
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, save_name)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            if logger:
                logger.info(f"Circular ROI verification plot saved to: {save_path}")
            else:
                print(f"Circular ROI verification plot saved to: {save_path}")

        # Display control block
        if plot_graphs and num_plots is not None:
            plt.show()
        else:
            plt.close(fig)

    # Return elements as clean flattened arrays matching original function behavior
    return mean_values, std_values


def plot_eruption_contours(
        imgs: np.ndarray,
        xc: float,
        yc: float,
        r: float,
        mean_ref: np.ndarray,
        std_ref: np.ndarray,
        num_plots: int = 4,
        max_sigma: int = 5,
        vmax: float = None,
        plot_graphs: bool = True,
        save_name: str = None,
        logger=None
):
    """
    Plots dynamic sigma-level contours strictly inside a circular mask area
    for uniformly sampled images across an eruption sequence.

    Parameters:
    -----------
    imgs : np.ndarray
        The eruption image stack of shape (N, H, W).
    xc, yc, r : float
        Circle parameters defining the active eruption ROI area.
    mean_ref : np.ndarray
        Array of baseline reference mean values (from the rectangular background).
    std_ref : np.ndarray
        Array of baseline reference standard deviations (from the rectangular background).
    num_plots : int, default 4
        Number of images to uniformly sample across the stack sequence.
    max_sigma : int, default 5
        The highest σ multiplier threshold to calculate contours up to (e.g., 1σ to 5σ).
    vmax : float, optional
        Maximum display cap value for the background imagery.
    plot_graphs : bool, default True
        If False, suppresses screen window generation and runs silently.
    save_name : str, optional
        Filename to export the generated grid image to the 'Plots' directory.
    """
    if logger:
        logger.info("Initializing dynamic visual contour matrix...")

    # Normalize dimensions for a single frame vs a stack
    if imgs.ndim == 2:
        num_frames = 1
        H, W = imgs.shape
        working_stack = imgs[np.newaxis, ...]
    else:
        num_frames, H, W = imgs.shape[:3]
        working_stack = imgs

    # Create the coordinate circle mask stencil once
    y_indices, x_indices = np.ogrid[:H, :W]
    circle_mask = (x_indices - xc) ** 2 + (y_indices - yc) ** 2 <= r ** 2

    # Select uniformly spaced image indices across the frame sequence
    plot_indices = np.linspace(0, num_frames - 1, num_plots, dtype=int)
    plot_indices = np.unique(plot_indices)
    actual_plot_count = len(plot_indices)

    if actual_plot_count == 0:
        return

    # Build dynamic layout grid structures
    cols = 2 if actual_plot_count >= 2 else 1
    rows = int(np.ceil(actual_plot_count / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
    ax_flat = axes.ravel() if actual_plot_count > 1 else [axes]

    # Use a smooth continuous sequential map for dynamic filled contour layers
    contour_cmap = plt.cm.get_cmap("YlOrRd")

    for idx, img_idx in enumerate(plot_indices):
        ax = ax_flat[idx]
        img = working_stack[img_idx]

        # Pull background references corresponding to this exact frame
        m_val = mean_ref[img_idx]
        s_val = std_ref[img_idx]

        # Use masked array to isolate data inside the circle for contour processing
        masked_img = np.ma.masked_array(img, mask=~circle_mask)
        img_min, img_max = masked_img.min(), masked_img.max()

        # Generate candidate thresholds (e.g., Mean + 1σ, Mean + 2σ...)
        candidate_levels = [m_val + k * s_val for k in range(1, max_sigma + 1)]

        # SMART SELECTION: Filter out levels that do not exist within this frame's values
        valid_levels = [lv for lv in candidate_levels if img_min < lv < img_max]

        # 1. Draw background full image layer
        im = ax.imshow(img, cmap="gray_r", origin="lower", vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Intensity (ADU)")

        # 2. Render Contour overlays if valid levels survived filtering
        if valid_levels:
            # Filled contours with automatic gradient mapping based on active levels
            cf = ax.contourf(masked_img, levels=valid_levels, cmap=contour_cmap, alpha=0.35)

            # Sharp outer boundary contour lines
            cs = ax.contour(masked_img, levels=valid_levels, colors="red", linewidths=1.0)

            # Structural labels on contour lines
            ax.clabel(cs, inline=True, fontsize=9, fmt="%.0f", colors="black")

        # 3. Add the yellow visual reference target boundary ring
        visual_circle = Circle((xc, yc), r, color="yellow", fill=False, linewidth=2, linestyle="--")
        ax.add_patch(visual_circle)

        # Labels & Details
        ax.set_title(f"Frame {img_idx} ({len(valid_levels)} Active σ-Contours)", fontsize=11, fontweight='bold')
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")

    # Clean out empty grid blocks
    for extra_ax in ax_flat[actual_plot_count:]:
        fig.delaxes(extra_ax)

    plt.tight_layout()

    # ---- Process Saving Block ----
    if save_name is not None:
        output_dir = "Plots"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        if logger:
            logger.info(f"Contour plot successfully saved to: {save_path}")
        else:
            print(f"Contour plot successfully saved to: {save_path}")

    # ---- Process Display Block ----
    if plot_graphs:
        plt.show()
    else:
        plt.close(fig)


def plot_eruption_histogram(
        imgs: np.ndarray,
        slice_end: int = None,  # Replicates your `[:-100]` frame slicing rule
        clip_min: float = 0.001,
        bins_count: int = 100,
        plot_graph: bool = True,
        title: str = "Intensity Distribution (All Points)",
        save_name: str = None,
        logger=None
) -> dict:
    """
    Flattens an image array, handles zero values via clipping, calculates key
    distribution statistics, and plots a Log-Log frequency histogram.
    """
    if logger:
        logger.info("Initializing data flattening and histogram calculations...")

    # 1. Apply frame slicing if working with a 3D stack, then flatten to 1D
    if imgs.ndim == 3 and slice_end is not None:
        data_master = imgs[:slice_end].flatten()
    else:
        data_master = imgs.flatten()

    # 2. Extract Inventory and Dynamic Range Stats
    total_pixels = len(data_master)
    non_zero_count = np.count_nonzero(data_master)
    zero_count = total_pixels - non_zero_count

    min_val_raw = np.min(data_master)
    max_val_raw = np.max(data_master)

    # 3. Handle Zero Values for Logarithmic Calculations
    # Clipping ensures we don't pass 0 or negative values to log10
    data_clipped = np.clip(data_master, clip_min, None)

    # Calculate distribution center and variance
    exp_value = np.mean(data_clipped)
    stdev = np.std(data_clipped)

    # Helper to print big numbers with spaces as thousands separators
    def fmt_int(val):
        return f"{int(val):,}".replace(",", " ")

    # Log/Print statistics summary text block
    stats_summary = (
        f"\n{'-' * 30}\n"
        f"HISTOGRAM DATA METRICS:\n"
        f"  Total Pixels Analyzed : {fmt_int(total_pixels)}\n"
        f"  True Zero Pixels      : {fmt_int(zero_count)}\n"
        f"  Raw Minimum Intensity : {min_val_raw}\n"
        f"  Raw Maximum Intensity : {max_val_raw}\n"
        f"  Distribution Mean (μ) : {exp_value:.3f}\n"
        f"  Standard Deviation (σ): {stdev:.3f}\n"
        f"{'-' * 30}"
    )

    if logger:
        logger.info(stats_summary)
    else:
        print(stats_summary)

    # 4. Generate Logarithmic Spacing Bins
    # Ensuring our bins stretch correctly across the clipped dynamic range
    bins_global = np.logspace(np.log10(np.min(data_clipped)), np.log10(np.max(data_clipped)), bins_count)

    # 5. Visual Rendering Module
    if plot_graph or save_name is not None:
        fig, ax = plt.subplots(figsize=(12, 8))

        # Render the raw counts histogram
        ax.hist(data_clipped, bins=bins_global, edgecolor="black", color="#34495e", alpha=0.7)

        # Overlay the baseline average reference line
        ax.axvline(exp_value, color="#e74c3c", linestyle="--", linewidth=2, label=f"Mean (μ = {exp_value:.2f})")

        # Apply standard log scaling configurations
        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Pixel Value (ADU / Intensity)", fontsize=11)
        ax.set_ylabel("Frequency (Counts)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5, which="both")  # "both" shows major & minor log gridlines
        ax.legend(fontsize=11)

        plt.tight_layout()

        # Save Logic execution
        if save_name is not None:
            output_dir = "Plots"
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, save_name)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            if logger:
                logger.info(f"Histogram successfully exported to: {save_path}")
            else:
                print(f"Histogram successfully exported to: {save_path}")

        if plot_graph:
            plt.show()
        else:
            plt.close(fig)

    # Return a summary dict in case you want to use these values downstream
    return {
        "mean": exp_value,
        "std": stdev,
        "total_pixels": total_pixels,
        "zero_pixels": zero_count
    }


def plot_average_intensity(
        arr: np.ndarray,
        xarr: np.ndarray = None,
        plot_graph: bool = True,
        title: str = "Averaged Frame Intensity History",
        x_label: str = None,
        y_label: str = "Mean Intensity (ADU)",
        logger=None,
        save_name=None,
) -> np.ndarray:
    """
    Computes the spatial average of an image stack and delegates rendering
    to plot_single_series with fully customizable text labels.
    """
    if logger:
        logger.info("Calculating average intensity profile across spatial coordinates...")

    # 1. Execute your custom averaging helper function
    average_int = average_numpy_array(arr, axis=(1, 2), logger=logger)

    # 2. Dynamic fallback logic if the user doesn't pass a custom X-axis label
    if x_label is None:
        x_label = "Time / Custom Scale" if xarr is not None else "Frame Index"

    # 3. Pass all custom strings down to the centralized plotting engine
    plot_single_series(
        data=average_int,
        x_series=xarr,
        plot_graph=plot_graph,
        title=title,
        x_label_custom=x_label,
        y_label=y_label,
        line_color="#2c3e50",  # Dark slate gray
        save_name=save_name,
        logger=logger
    )

    return average_int