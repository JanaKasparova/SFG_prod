import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from math import ceil, sqrt
from typing import Optional, Sequence
from astropy.io import fits
from processing import crop_images
import textwrap
from analysis import *
from processing import *
from data_io import *
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import numpy as np
import os


def plot_single_series(
        data: np.ndarray,
        time_series: np.ndarray = None,
        plot_graph: bool = True,
        title: str = "Data Profile",
        x_label: str = None,
        y_label: str = "Value",
        line_color: str = "#1f77b4",
        save_name: str = None,
        num_ticks: int = None,  # <--- Added parameter for tick density control
        logger=None,
        **fig_kwargs
):
    """
    Plots a single 1D data array with an option to supply a custom X-axis or time series.
    Safely unpacks Astropy/Pandas time structures and normalizes parameter names.
    """
    # 1. Gracefully normalize overlapping keyword arguments from the caller
    x_label = x_label or fig_kwargs.pop("xlabel", None)
    y_label = y_label if y_label != "Value" else fig_kwargs.pop("ylabel", y_label)
    save_name = save_name or fig_kwargs.pop("save_filename", None)
    num_ticks = num_ticks or fig_kwargs.pop("num_ticks", None) or fig_kwargs.pop("n_xticks", None)

    if logger:
        logger.info(f"Preparing single series plot for array of size: {len(data)}")

    # 2. Resolve X-Axis Data, converting Astropy Time or Pandas timestamps on the fly
    if time_series is not None:
        if hasattr(time_series, "datetime"):
            x_data = time_series.datetime
        elif hasattr(time_series, "to_pydatetime"):
            x_data = time_series.to_pydatetime()
        else:
            x_data = time_series

        x_label = x_label if x_label else "Time"
        if len(x_data) != len(data):
            raise ValueError(f"X-series length ({len(x_data)}) must match the data array length ({len(data)}).")
    else:
        x_data = np.arange(len(data))
        x_label = x_label if x_label else "Index"

    # 3. Setup Figure Layout
    if plot_graph or save_name is not None:
        figsize = fig_kwargs.get("figsize", (10, 5))
        fig, ax = plt.subplots(figsize=figsize)

        # Plot line with a subtle marker for data points
        ax.plot(x_data, data, color=line_color, linewidth=2, linestyle="-", marker="o", markersize=3, alpha=0.8)

        # Dynamic Tick Adjustments
        if num_ticks is not None:
            if time_series is not None:
                # Controls max ticks for date objects
                ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=num_ticks))
            else:
                # Controls max ticks for numerical/index arrays
                ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=num_ticks))

        # Style layout
        ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)

        # Auto-format dates cleanly if the x-axis contains datetime objects
        if time_series is not None:
            fig.autofmt_xdate()

        # 4. Process Saving Block
        if save_name is not None:
            if os.path.dirname(save_name):
                save_path = save_name
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            else:
                output_dir = "Plots"
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, save_name)

            plt.savefig(save_path, bbox_inches='tight', dpi=300)

            if logger is not None:
                logger.info(f"Plot successfully saved to: {save_path}")
            else:
                print(f"Plot successfully saved to: {save_path}")

        # 5. Process Display Block
        if plot_graph:
            plt.show()
        else:
            plt.close(fig)


def plot_image_grid(
        images: np.ndarray,
        titles: Optional[Sequence[str]] = None,
        n_images: int = 6,
        suptitle: Optional[str] = None,
        cmap: str = "gray",
        figsize_scale: float = 5.5,
        save_name: str = None,
        plot_image: bool = True,
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
        # ---- Save Figure Setup ----
        if save_name is not None:
            import os

            # Standardize path
            save_path = save_name

            # Extract parent directory and create all necessary subfolder(s)
            parent_dir = os.path.dirname(save_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

            # Save the figure
            plt.savefig(save_path, bbox_inches="tight", dpi=300)

            if "logger" in locals() and logger is not None:
                logger.info(f"Plot successfully saved to: {save_path}")
            else:
                print(f"Plot successfully saved to: {save_path}")
        # ---------------------------
    # -----------------------------------------------------------
    if plot_image:
        plt.show()
    else:
        plt.close()


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
            import os

            save_path = save_name

            # Safely create parent directories if they don't exist
            parent_dir = os.path.dirname(save_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

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
                    output_dir: str | None = None,
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
        plot_graph: bool = True,
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
    plot_graph : bool, optional
        Whether to display the interactive plot window (default: True).
    save_name : str, optional
        Exact save path or filename for the plot image.
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

    plt.tight_layout()

    # ---- Save Figure Setup ----
    if save_name is not None:
        import os

        save_path = save_name

        # Safely create parent directories if they don't exist
        parent_dir = os.path.dirname(save_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        plt.savefig(save_path, bbox_inches='tight', dpi=300)

        if logger is not None:
            logger.info(f"Plot successfully saved to: {save_path}")
        else:
            print(f"Plot successfully saved to: {save_path}")

    # ---- Process Display Block ----
    if plot_graph:
        plt.show()
    else:
        plt.close(fig)  # Prevent background memory leaks


def plot_dark_histograms(
        stats: dict,
        logger=None,
        title_all="All Points (with hotpoints)",
        title_no_hot="Without Hotpoints",
        title_only_hot="Only Hotpoints",
        title_only_clean="Only Clean Points",
        save_name=None,
        plot_histograms=True,
        **fig_kwargs
):
    """
    Generates a 2x2 grid of histograms analyzing the distribution of hot and clean pixels.
    Handles nested directory creation safely if a full save path is provided.
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
    def format_ax(ax, title, xlabel="Pixel Value / Intenzity", ylabel="Frequency"):
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

    # ---- Save Figure Setup ----
    if save_name is not None:
        import os

        # If save_name is just a filename with no directory path, default to "Plots"
        if not os.path.dirname(save_name):
            save_name = os.path.join("Plots", save_name)

        # Extract whatever directory path actually exists and build it recursively
        dir_name = os.path.dirname(save_name)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        # Save the figure (bbox_inches='tight' prevents labels from cutting off)
        plt.savefig(save_name, bbox_inches='tight', dpi=300)

        if logger is not None:
            logger.info(f"Plot successfully saved to: {save_name}")
        else:
            print(f"Plot successfully saved to: {save_name}")

    if plot_histograms:
        plt.tight_layout()
        plt.show()
    else:
        plt.close()

    if logger:
        logger.info("Histograms plotted successfully.")


def plot_hot_pixel_map(
        stats: dict,
        logger=None,
        title_clipped="Clipped Master Dark (No Hot Points)",
        title_highlighted="Master Dark with Hot Pixels Highlighted",
        save_name=None,
        plot_map: bool = True,
        **fig_kwargs
):
    """
    Plots the master dark image side-by-side: one clipped to the threshold,
    and one highlighting the exact locations of the hot pixels.
    Handles nested directory creation safely if a full save path is provided.
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

    # ---- Save Figure Setup ----
    if save_name is not None:
        import os

        # If save_name is just a filename with no directory path, default to "Plots"
        if not os.path.dirname(save_name):
            save_name = os.path.join("Plots", save_name)

        # Extract whatever directory path actually exists and build it recursively
        dir_name = os.path.dirname(save_name)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        # Save the figure (bbox_inches='tight' prevents labels from cutting off)
        plt.savefig(save_name, bbox_inches='tight', dpi=300)

        if logger is not None:
            logger.info(f"Plot successfully saved to: {save_name}")
        else:
            print(f"Plot successfully saved to: {save_name}")

    if plot_map:
        plt.tight_layout()
        plt.show()
    else:
        plt.close()

    if logger:
        logger.info("Spatial maps plotted successfully.")


import textwrap


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
    Handles nested directory creation safely if a full save path is provided.
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
            import os

            # If save_name is just a filename with no directory path, default to "Plots"
            if not os.path.dirname(save_name):
                save_name = os.path.join("Plots", save_name)

            # Extract whatever directory path actually exists and build it recursively
            dir_name = os.path.dirname(save_name)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            # Save the figure
            plt.savefig(save_name, bbox_inches='tight', dpi=300)

            if logger is not None:
                logger.info(f"Plot successfully saved to: {save_name}")
            else:
                print(f"Plot successfully saved to: {save_name}")

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
        vmax: float = 1000,
        save_name: str = None,
        plot_graphs: bool = True,
        logger=None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Crops images to a specific ROI rectangle, calculates the mean and standard
    deviation values within that box, and uniformly plots N sample frames.
    Handles nested directory creation safely if a full save path is provided.

    Parameters:
    -----------
    imgs : np.ndarray
        Array stack of shape (N, H, W) or (N, H, W, C).
    xmin, xmax, ymin, ymax : int
        Bounding spatial coordinates for the ROI rectangle.
    num_plots : int, optional
        Number of images to uniformly select and plot on screen.
    vmax : float, default 1000
        Maximum display scaling cutoff for display normalization.
    save_name : str, optional
        Filename or complete path to save the final plot.
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

        # ---- Save Logic Execution ----
        if save_name is not None:
            import os

            # If save_name is just a filename with no directory path, default to "Plots"
            if not os.path.dirname(save_name):
                save_name = os.path.join("Plots", save_name)

            # Extract whatever directory path actually exists and build it recursively
            dir_name = os.path.dirname(save_name)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            plt.savefig(save_name, bbox_inches='tight', dpi=300)
            if logger:
                logger.info(f"ROI verification grid saved to: {save_name}")

        if plot_graphs:
            plt.show()
        else:
            plt.close()

    return mean_values, std_values


def plot_stats(
        means: np.ndarray,
        stds: np.ndarray,
        time_series: np.ndarray = None,
        plot_stats: bool = True,
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
    Handles nested directory creation safely if a full save path is provided.
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
    if plot_stats or save_name is not None:
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
            import os

            # If save_name is just a filename with no directory path, default to "Plots"
            if not os.path.dirname(save_name):
                save_name = os.path.join("Plots", save_name)

            # Extract whatever directory path actually exists and build it recursively
            dir_name = os.path.dirname(save_name)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            # Save the figure (bbox_inches='tight' prevents labels from cutting off)
            plt.savefig(save_name, bbox_inches='tight', dpi=300)

            if logger is not None:
                logger.info(f"Statistics plots successfully saved to: {save_name}")
            else:
                print(f"Statistics plots successfully saved to: {save_name}")

        # ---- Process Display Block ----
        if plot_stats:
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
    Handles nested directory creation safely if a full save path is provided.

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
        Filename or complete path to save the generated figure.
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

    # 4. Uniform Grid Graphing and Saving Layout
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

        # ---- Save to disk logic block ----
        if save_name is not None:
            import os

            # If save_name is just a filename with no directory path, default to "Plots"
            if not os.path.dirname(save_name):
                save_name = os.path.join("Plots", save_name)

            # Extract whatever directory path actually exists and build it recursively
            dir_name = os.path.dirname(save_name)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            plt.savefig(save_name, bbox_inches='tight', dpi=300)
            if logger:
                logger.info(f"Circular ROI verification plot saved to: {save_name}")
            else:
                print(f"Circular ROI verification plot saved to: {save_name}")

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
        num_contours: int = 5,
        min_sigma: float = 5.0,  # Added customizable minimum sigma parameter
        vmax: float = None,
        plot_graphs: bool = True,
        save_name: str = None,
        logger=None
):
    """
    Plots dynamic threshold contours over an eruption image stack within a circular Region of Interest (ROI).

    This function isolates highly energetic eruptive features by filtering out quiet background
    noise using a customizable sigma-threshold floor. It samples frames uniformly across the input
    dataset's timeline, calculates the unique dynamic intensity range of the eruption for each
    sampled frame, and spreads a specified number of contours evenly from the baseline floor up to
    just below the frame's peak intensity. High-intensity pixels exceeding the upper threshold
    are cleanly captured and enclosed inside the highest contour loop.
    Handles nested directory creation safely if a full save path is provided.

    Parameters:
    -----------
    imgs : np.ndarray
        The input image or image stack data. Supports a 2D array of shape (H, W) for a single
        frame, or a 3D array of shape (N, H, W) representing a time-series stack of N frames.
    xc : float
        The X-coordinate of the center point for the circular tracking mask.
    yc : float
        The Y-coordinate of the center point for the circular tracking mask.
    r : float
        The radius of the circular tracking mask defining the active boundary perimeter.
    mean_ref : np.ndarray
        An array containing the baseline reference mean intensity values calculated from a
        background control region. Must contain a index value for every frame in the stack.
    std_ref : np.ndarray
        An array containing the baseline reference standard deviation values calculated from
        a background control region. Must contain an index value for every frame in the stack.
    num_plots : int, default 4
        The number of subplots to generate in the final figure grid by uniformly sampling
        frames across the entire sequence timeline.
    num_contours : int, default 5
        The number of contour intervals to calculate and plot above the minimum sigma floor.
    min_sigma : float, default 5.0
        The minimum sigma multiplier used to establish the baseline threshold floor:
        Floor = Mean + (min_sigma * Standard Deviation)
        Only pixel signals exceeding this value are treated as part of the eruption.
    vmax : float, optional
        Maximum display intensity cutoff passed to the underlying image map rendering engine.
        If None, the grayscale map normalizes to the image's natural limits.
    plot_graphs : bool, default True
        If True, flushes the canvas immediately to open the interactive visualization window.
        If False, closes the figure canvas silently to protect background system memory.
    save_name : str, optional
        The filename or complete path to export the final figure grid.
    logger : logging.Logger, optional
        An optional pipeline tracking logger instance used to transmit processing state updates
        and export milestones to standard output.
    """
    if logger:
        logger.info(f"Generating localized eruption contour maps (Floor = Mean + {min_sigma}σ)...")

    if imgs.ndim == 2:
        num_frames = 1
        H, W = imgs.shape
        working_stack = imgs[np.newaxis, ...]
    else:
        num_frames, H, W = imgs.shape[:3]
        working_stack = imgs

    # 1. Create the spatial circle mask
    y_indices, x_indices = np.ogrid[:H, :W]
    circle_mask = (x_indices - xc) ** 2 + (y_indices - yc) ** 2 <= r ** 2

    # 2. Sample frame indices uniformly across the timeline
    plot_indices = np.linspace(0, num_frames - 1, num_plots, dtype=int)
    plot_indices = np.unique(plot_indices)
    actual_plot_count = len(plot_indices)

    if actual_plot_count == 0:
        return

    # 3. Setup Layout Grid (Optimized for a square aspect ratio)
    if actual_plot_count == 1:
        cols = 1
        rows = 1
    else:
        # Calculate the ideal square matrix edge size
        cols = int(np.ceil(np.sqrt(actual_plot_count)))
        rows = int(np.ceil(actual_plot_count / cols))

    # Keep subplot dimensions uniform (e.g., 6x5.5 per plot) so the figure stays square
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5.5 * rows))

    # Ensure flat access arrays even for a single-plot configuration
    if actual_plot_count == 1:
        ax_flat = [axes]
    else:
        ax_flat = axes.ravel()

    contour_cmap = plt.cm.get_cmap("YlOrRd")

    # 4. Loop Through Sampled Frames
    for idx, img_idx in enumerate(plot_indices):
        ax = ax_flat[idx]
        img = working_stack[img_idx]

        m_val = mean_ref[img_idx]
        s_val = std_ref[img_idx]

        # Isolate the eruptive region data
        masked_img = np.ma.masked_array(img, mask=~circle_mask)
        img_max = masked_img.max()

        # Define the dynamic bottom boundary based on the chosen min_sigma
        eruption_floor = m_val + min_sigma * s_val

        # ---- DYNAMIC CONTOUR SPACING ENGINE ----
        if img_max > eruption_floor:
            # Space contours evenly from the floor up to 98% of the peak value
            valid_levels = np.linspace(eruption_floor, img_max * 0.98, num_contours)
            valid_levels = np.unique(valid_levels)
        else:
            valid_levels = []

        # Draw baseline grayscale framework
        im = ax.imshow(img, cmap="gray_r", origin="lower", vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Intensity (ADU)")

        # Render Convoluted Heat Zones if the eruption breaches the floor
        if len(valid_levels) > 1:
            cf = ax.contourf(masked_img, levels=valid_levels, cmap=contour_cmap, alpha=0.4)
            cs = ax.contour(masked_img, levels=valid_levels, colors="red", linewidths=1.0)
            ax.clabel(cs, inline=True, fontsize=8, fmt="%.0f", colors="black")
            active_contours_count = len(valid_levels)
        else:
            active_contours_count = 0

        # Overlay yellow verification ring perimeter
        visual_circle = Circle((xc, yc), r, color="yellow", fill=False, linewidth=1.5, linestyle="--")
        ax.add_patch(visual_circle)

        # Context Formatting with dynamic title
        ax.set_title(f"Frame {img_idx} ({active_contours_count} Levels Above {min_sigma}σ)", fontsize=11,
                     fontweight='bold')

    # Prune empty window spots
    for extra_ax in ax_flat[actual_plot_count:]:
        fig.delaxes(extra_ax)

    plt.tight_layout()

    # ---- Save Module ----
    if save_name is not None:
        import os

        # If save_name is just a filename with no directory path, default to "Plots"
        if not os.path.dirname(save_name):
            save_name = os.path.join("Plots", save_name)

        # Extract whatever directory path actually exists and build it recursively
        dir_name = os.path.dirname(save_name)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        plt.savefig(save_name, bbox_inches='tight', dpi=300)
        if logger:
            logger.info(f"Contours saved to: {save_name}")
        else:
            print(f"Contours saved to: {save_name}")

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



def load_WL_spectrum():
    wlc = np.load("./FICUS/useful_files/WL_range_C.npy")
    wld = np.load("./FICUS/useful_files/WL_range_D.npy")
    return wlc, wld


def plot_spectrum_at_time(mC, mD, target_time, save_filename: str = None, show_plot: bool = True):
    """
    Plots the spectrum from spectrometers C and D at a specified target time.

    Parameters:
        mC: The Light object for spectrometer C (loaded via load_hdf_light).
        mD: The Light object for spectrometer D (loaded via load_hdf_light).
        target_time: The timestamp to plot (datetime object or parsable string).
        save_filename (str, optional): Filepath to save the figure. If None, saving is skipped.
        show_plot (bool): If True, calls plt.show() to display the plot. Default is True.
    """

    # 1. Flexible helper to convert time strings/objects into datetime
    def parse_to_datetime(t_input):
        if isinstance(t_input, datetime):
            return t_input
        if hasattr(t_input, 'datetime'):  # Handles pandas/astropy wrappers
            return t_input.datetime
        if isinstance(t_input, str):
            t_str = t_input.strip('"\' \n\t')
            for fmt in (
                    "%Y-%m-%d %H:%M:%S.%f",
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%d %H:%M",
                    "%Y-%m-%dT%H:%M:%S.%f",
                    "%Y-%m-%dT%H:%M:%S"
            ):
                try:
                    return datetime.strptime(t_str, fmt)
                except ValueError:
                    continue
        raise TypeError(f"Unsupported time format: {type(t_input)}")

    # Parse target time
    dt_target = parse_to_datetime(target_time)

    # 2. Determine index offset ('diff') using ROW_DELTA
    # Safely look for ROW_DELTA attribute (defaults to 500ms)
    row_delta_ms = getattr(mC, "row_delta", getattr(mC, "row_delta_ms", getattr(mC, "ROW_DELTA", 500)))
    step_seconds = row_delta_ms / 1000.0

    # Calculate time delta and find the closest index row
    time_difference = dt_target - mC.t_row0
    diff = int(round(time_difference.total_seconds() / step_seconds))

    # 3. Bounds check to protect against index errors
    num_rows = mC.data.shape[0]
    if diff < 0 or diff >= num_rows:
        dt_end = mC.t_row0 + timedelta(seconds=(num_rows - 1) * step_seconds)
        raise IndexError(
            f"Target time {dt_target} is out of bounds for the dataset.\n"
            f"Dataset range: {mC.t_row0} to {dt_end}.\n"
            f"Calculated index row: {diff} (valid range: 0 to {num_rows - 1})."
        )

    # Calculate the exact timestamp being plotted
    exact_plot_time = mC.t_row0 + timedelta(seconds=diff * step_seconds)
    exact_time_str = exact_plot_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    # 4. Load the wavelength grids
    wlc, wld = load_WL_spectrum()

    # 5. Build the plot
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=120)

    # Plot spectra
    ax.plot(wlc, mC.data[diff], label=f"mC (Index: {diff})", color="#0984e3", alpha=0.85, lw=1.2)
    ax.plot(wld, mD.data[diff], label=f"mD (Index: {diff})", color="#d63031", alpha=0.85, lw=1.2)

    # Labels & Title
    ax.set_xlabel("Wavelength (nm)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Intensity [data units]", fontsize=11, fontweight="bold")
    ax.set_title(f"Spectra at {exact_time_str} (Target: {dt_target.strftime('%H:%M:%S')})",
                 fontsize=12, fontweight="bold", pad=10)

    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout()

    # 6. Save handler
    if save_filename:
        plt.savefig(save_filename, dpi=300, bbox_inches="tight")
        print(f"Spectrum plot saved successfully to: {save_filename}")

    # 7. Show handler
    if show_plot:
        plt.show()
    else:
        plt.close(fig)  # Free figure memory if not showing



def plot_flare_summary(goes_flare, gradient, time_series, h_alpha_data, time_array, intensity,
                       save_name=None, plot_graph=True, num_ticks=None):
    """
    Plots a 4-panel synchronized timeline matching your working setup exactly.
    Accepts full file paths for save_name to match the other save parameters seamlessly.
    Includes an optional 'num_ticks' parameter to control x-axis tick density.
    """
    # Explicitly convert GOES time axis (Astropy Time) to standard plottable datetimes
    goes_time_plot = goes_flare.time.datetime if hasattr(goes_flare.time, "datetime") else goes_flare.time

    # Safe-unpack for Astropy Time objects/arrays passed into timerange and time_array
    t_range_plot = time_series.datetime if hasattr(time_series, "datetime") else time_series
    if isinstance(time_series, np.ndarray) and time_series.dtype == object and len(time_series) > 0:
        if hasattr(time_series[0], "datetime"):
            t_range_plot = [t.datetime for t in time_series]

    t_array_plot = time_array.datetime if hasattr(time_array, "datetime") else time_array
    if isinstance(time_array, np.ndarray) and time_array.dtype == object and len(time_array) > 0:
        if hasattr(time_array[0], "datetime"):
            t_array_plot = [t.datetime for t in time_array]

    # Initialize a 4-panel shared X-axis layout
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    # Panel 1: GOES Flux
    ax1.plot(goes_time_plot, goes_flare.quantity("xrsb"), color="tab:blue")
    ax1.set_ylabel("Flux (W m$^{-2}$)")
    ax1.set_title("GOES Flare, Gradient, and H alpha")

    # Panel 2: GOES Gradient (Using the explicit passed parameter paired with the converted time axis)
    ax2.plot(goes_time_plot, gradient, color="tab:red")
    ax2.set_ylabel("Gradient (W m$^{-2}$ s$^{-1}$)")

    # Panel 3: Spectrometer H-alpha
    ax3.plot(t_range_plot, h_alpha_data, color="tab:green")
    ax3.set_ylabel("H alpha")

    # Panel 4: SlitJaw Intensity Profile
    ax4.plot(t_array_plot, intensity, color="tab:blue")
    ax4.set_ylabel("Intensity")
    ax4.set_xlabel("Time")

    # Dynamic x-axis tick adjustment across the shared subplots
    if num_ticks is not None:
        ax4.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=num_ticks))

    # Layout cleanup
    fig.autofmt_xdate()
    plt.tight_layout()

    # Save tracking - dynamically checks and builds the containing directory from the path parameter
    if save_name is not None:
        dir_name = os.path.dirname(save_name)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_name, bbox_inches='tight', dpi=300)

    if plot_graph:
        plt.show()
    else:
        plt.close(fig)

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle


def animate_eruption_region(
        imgs: np.ndarray,
        xc: float = None,
        yc: float = None,
        r: float = None,
        time_series=None,
        fps: int = 10,
        vmax: float = 2.0,
        save_name: str = "eruption_animation.mp4",
        logger=None
):
    """
    Generates an animated video/GIF from the cropped eruption stack.
    Overlays the circular analysis region and dynamic timestamp per frame.
    """
    if logger:
        logger.info(f"🎥 Generating eruption animation for {len(imgs)} frames...")
    else:
        print(f"🎥 Generating eruption animation for {len(imgs)} frames...")

    fig, ax = plt.subplots(figsize=(6, 6))

    # Initial frame layout setup
    im = ax.imshow(imgs[0], cmap='inferno', origin='lower', vmin=0, vmax=vmax)

    # Only draw the circle patch if circle arguments are explicitly provided
    if xc is not None and yc is not None and r is not None:
        circle = Circle((xc, yc), r, color='cyan', fill=False, linewidth=1.8, linestyle='--')
        ax.add_patch(circle)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Intensity", fontsize=10)
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")

    # Frame update loop
    def update(frame_idx):
        im.set_data(imgs[frame_idx])
        if time_series is not None and frame_idx < len(time_series):
            # Formats datetime object cleanly
            t_obj = time_series[frame_idx]
            t_str = t_obj.strftime("%Y-%m-%d %H:%M:%S") if hasattr(t_obj, "strftime") else str(t_obj)
            ax.set_title(f"Time: {t_str} (Frame {frame_idx + 1}/{len(imgs)})", fontsize=10, fontweight='bold')
        else:
            ax.set_title(f"Frame {frame_idx + 1}/{len(imgs)}", fontsize=10, fontweight='bold')
        return [im]

    anim = animation.FuncAnimation(fig, update, frames=len(imgs), interval=1000 / fps, blit=False)

    # Path resolution
    dir_name = os.path.dirname(save_name)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    # Save animation (Attempts MP4 with ffmpeg, falls back to Pillow GIF)
    if save_name.endswith('.gif'):
        anim.save(save_name, writer='pillow', fps=fps)
    else:
        try:
            anim.save(save_name, writer='ffmpeg', fps=fps, extra_args=['-vcodec', 'libx264'])
        except Exception as e:
            gif_path = os.path.splitext(save_name)[0] + ".gif"
            msg = f"⚠️ FFmpeg writer failed/unavailable ({e}). Falling back to GIF format: {gif_path}"
            if logger:
                logger.warning(msg)
            else:
                print(msg)
            anim.save(gif_path, writer='pillow', fps=fps)
            save_name = gif_path

    plt.close(fig)

    if logger:
        logger.info(f"🎬 Movie successfully saved to: {save_name}")
    else:
        print(f"🎬 Movie successfully saved to: {save_name}")
