from data_io import *
import numpy as np
from typing import Any
import os
import matplotlib.pyplot as plt


def average_numpy_array(
    arr: np.ndarray,
    out_dtype: np.dtype = None,
    axis = 0,
    logger=None
) -> np.ndarray:
    """
    Compute the average of a NumPy array along a given axis and cast the result
    to the requested output dtype.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    out_dtype : np.dtype, optional
        Output data type (default: np.int16).
    axis : int, optional
        Axis along which the average is computed (default: 0).

    Returns
    -------
    np.ndarray
        Averaged array cast to ``out_dtype``.
    """
    if out_dtype is None:
        out_dtype = arr.dtype
    if logger is not None:
        logger.debug(f"Averageing a numpy array, axis={axis}, dtype={out_dtype}")
        logger.debug(f"Shape of averaged numpy array before: {arr.shape}")
    temp = arr.astype(np.float64, copy=False)
    avg = np.average(temp, axis=axis)
    if logger is not None:
        logger.debug(f"Shape of averaged numpy array after: {avg.shape}")
    return avg.astype(out_dtype, copy=False)

def average_hdf_light(
    hdf_dir: str,
    light_idx: int,
    axis: int = 0,
    out_dtype: np.dtype = np.int16,
    save=False,
    names=("master_image",["C","D"]),
    logger=None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load C and D light HDF files (spectrum) from a directory, compute their averages
    along a given axis, and cast to the requested output dtype.

    Parameters
    ----------
    hdf_dir : str
        Directory containing C and D .hdf files.
    light_idx : int
        Light index passed to the Light constructor.
    axis : int, optional
        Axis along which the average is computed (default: 0).
    out_dtype : np.dtype, optional
        Output data type (default: np.int16).
    logger : optional
        Logger with .debug() method.

    Returns
    -------
    (avg_C, avg_D) : tuple of np.ndarray
        Averaged C and D light data arrays.
    """
    if logger is not None:
        logger.debug(
            "Averaging C and D light from %s, axis=%d, dtype=%s",
            hdf_dir,
            axis,
            out_dtype,
        )

    mC, mD = load_hdf_light(hdf_dir=hdf_dir, light_idx=light_idx)

    avg_C = average_numpy_array(
        mC,
        axis=axis,
        out_dtype=out_dtype,
    )
    avg_D = average_numpy_array(
        mD,
        axis=axis,
        out_dtype=out_dtype,
    )
    return avg_C, avg_D

def analyze_dark_frame(
        image: np.ndarray,
        N: float = 10.0,
        logger = None
) -> dict:
    """
    Analyzes a dark frame to find global statistics and identify hot pixels.
    """
    if logger:
        logger.info(f"Analyzing dark frame with threshold N={N} sigma.")

    data_flat = image.flatten()
    exp_value = np.mean(data_flat)
    stdev = np.std(data_flat)
    threshold = exp_value + N * stdev

    # 1D arrays for histograms
    hot_mask_1d = data_flat > threshold
    hot_points = data_flat[hot_mask_1d]
    clean_points = data_flat[~hot_mask_1d]

    # 2D coordinates for image mapping
    hot_y, hot_x = np.where(image > threshold)
    hot_values = image[hot_y, hot_x]

    if logger:
        logger.info(f"Analysis complete: Mean={exp_value:.2f}, Std={stdev:.2f}, Threshold={threshold:.2f}")
        logger.info(f"Found {len(hot_points)} hot pixels out of {len(data_flat)} total pixels.")

    return {
        "image": image,
        "data_flat": data_flat,
        "exp_value": exp_value,
        "stdev": stdev,
        "threshold": threshold,
        "N": N,
        "hot_points": hot_points,
        "clean_points": clean_points,
        "hot_y": hot_y,
        "hot_x": hot_x,
        "hot_values": hot_values
    }


def calculate_erupting_pixels(
        imgs: np.ndarray,
        xc: float,
        yc: float,
        r: float,
        mean_ref: np.ndarray,
        std_ref: np.ndarray,
        sigma_level: float = 5.0,
        plot_graphs: bool = False,
        save_name: str = None,
        logger=None
) -> tuple[np.ndarray, int]:
    """
    Calculates the normalized ratio of erupting pixels inside a circular mask
    based on a dynamic reference threshold (mean_ref + sigma_level * std_ref).

    Parameters:
    -----------
    imgs : np.ndarray
        The cropped image stack of shape (N, H, W) containing the eruption site.
    xc, yc, r : float
        Coordinates defining the center and radius of the circular target mask.
    mean_ref : np.ndarray
        Array of baseline reference mean values calculated from a background/control region.
    std_ref : np.ndarray
        Array of baseline reference standard deviations from a background/control region.
    sigma_level : float, default 5.0
        The multiplier for the noise floor threshold.
    plot_graphs : bool, default True
        If True, displays a line graph showing the eruption profile over time.
    save_name : str, optional
        Filename to save the generated trend plot inside the 'Plots' directory.

    Returns:
    --------
    tuple[np.ndarray, int]
        - normalized_erupting_pixels: Array containing the ratio of active pixels per frame (0.0 to 1.0).
        - total_mask_pixels: The absolute integer count of total pixels inside the circle stencil.
    """
    if logger:
        logger.info(f"Initializing eruption calculation using {sigma_level}σ threshold criteria...")

    # 1. Coordinate array dimensional normalization
    if imgs.ndim == 2:
        num_frames = 1
        H, W = imgs.shape
        working_stack = imgs[np.newaxis, ...]
    else:
        num_frames, H, W = imgs.shape[:3]
        working_stack = imgs

    # Safety structural check
    if len(mean_ref) != num_frames or len(std_ref) != num_frames:
        raise ValueError("The lengths of mean_ref and std_ref must match the number of frames in imgs.")

    # 2. Build Circle Meshgrid ONCE outside the main execution loop
    y_indices, x_indices = np.ogrid[:H, :W]
    circle_mask = (x_indices - xc) ** 2 + (y_indices - yc) ** 2 <= r ** 2
    total_mask_pixels = int(np.sum(circle_mask))

    if total_mask_pixels == 0:
        raise ValueError("The circular mask area is 0 pixels! Check your coordinates and radius.")

    # 3. Analyze Frame Threshold Iterations
    normalized_erupting_pixels = np.zeros(num_frames)

    for i in range(num_frames):
        # Calculate dynamic threshold for this specific frame
        frame_threshold = mean_ref[i] + sigma_level * std_ref[i]

        # Count pixels that are BOTH inside the circle and greater than the threshold
        active_count = np.sum((working_stack[i] > frame_threshold) & circle_mask)

        # Normalize by the total pixel real estate inside the stencil mask
        normalized_erupting_pixels[i] = active_count / total_mask_pixels

    # Format large integer outputs with spaces instead of commas for scannability
    formatted_pixel_count = f"{total_mask_pixels:,}".replace(",", " ")
    if logger:
        logger.info(f"Analysis complete. Circle bounds isolated {formatted_pixel_count} total pixels.")

    # 4. Optional Graphical Presentation Module
    if plot_graphs or save_name is not None:
        fig, ax = plt.subplots(figsize=(10, 5))

        x_axis = np.arange(num_frames)
        ax.plot(x_axis, normalized_erupting_pixels, color="#e67e22", linewidth=2, marker="o", markersize=4)

        ax.set_title(f"Normalized Erupting Pixels Profile ({sigma_level}σ Cutoff)", fontsize=13, fontweight='bold',
                     pad=12)
        ax.set_xlabel("Frame Index", fontsize=11)
        ax.set_ylabel("Ratio of Active Pixels (Erupting / Total)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim(-0.05, 1.05)  # Bound ratio timeline framework cleanly

        # Save Logic Execution Block
        if save_name is not None:
            output_dir = "Plots"
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, save_name)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            if logger:
                logger.info(f"Eruption trend profile graph saved to: {save_path}")
            else:
                print(f"Eruption trend profile graph saved to: {save_path}")

        if plot_graphs:
            plt.show()
        else:
            plt.close(fig)

    return normalized_erupting_pixels, total_mask_pixels





def run_analysis(calibrated_data):
    print("  Running analysis")
    return {"result": 42}
