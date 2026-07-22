from data_io import *
import numpy as np
from typing import Any
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d


def average_numpy_array(
        arr: np.ndarray,
        out_dtype: np.dtype = None,
        axis=0,
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
        names=("master_image", ["C", "D"]),
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
        logger=None
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


import os
import re
import numpy as np
from astropy.time import Time


def extract_astropy_time(filepath: str, logger=None) -> Time:
    """
    Extracts a timestamp from a SlitJaw filename string and converts
    it into an astropy.time.Time object using explicit microsecond definitions.
    """
    filename = os.path.basename(filepath)

    # Separates the 2-digit seconds (\d{2}) and the trailing microseconds (\d+) explicitly
    pattern = r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\.(\d+)"
    match = re.search(pattern, filename)

    if not match:
        error_msg = f"Could not extract a valid timestamp pattern from filename: {filename}"
        if logger:
            logger.error(error_msg)
        raise ValueError(error_msg)

    # Unpack all groups, explicitly isolating the microsecond digits
    year, month, day, hour, minute, second, microsecond = match.groups()

    # Construct the high-precision ISO definition string
    iso_string = f"{year}-{month}-{day} {hour}:{minute}:{second}.{microsecond}"

    return Time(iso_string, format="iso", scale="utc")


import os
import re
import numpy as np
from astropy.time import Time


def compile_directory_timestamps(directory_path: str, logger=None) -> Time:
    """
    Scans a directory for files matching the SlitJaw timestamp pattern,
    extracts their times, and returns a single unified, vectorized Astropy Time array.
    """
    if not os.path.isdir(directory_path):
        error_msg = f"Provided directory path does not exist: {directory_path}"
        if logger:
            logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    if logger:
        logger.info(f"Scanning directory for timestamps: {directory_path}")

    # Naming pattern to identify valid target files and filter out system artifacts
    target_pattern = r"\d{8}_\d{6}\.\d+"

    # 1. Gather and sort filenames to maintain chronological sequence order
    all_files = sorted(os.listdir(directory_path))
    valid_files = [f for f in all_files if re.search(target_pattern, f)]

    if not valid_files:
        if logger:
            logger.warning(f"No files matching the timestamp pattern were found in {directory_path}")
        return Time([])  # Returns an empty vectorized Time object

    # 2. Process files sequentially through the extractor
    time_objects_list = []
    for file_name in valid_files:
        full_path = os.path.join(directory_path, file_name)
        try:
            t_obj = extract_astropy_time(full_path, logger=logger)
            time_objects_list.append(t_obj)
        except ValueError:
            # Skip file if parsing fails edge-cases
            continue

    # 3. Pack into a unified, vectorized Astropy Time array object
    # This converts a list of scalar Time objects into a single cohesive time array
    time_array = Time(time_objects_list)

    if logger:
        logger.info(f"Successfully compiled {len(time_array)} time steps into a unified Time array.")

    return time_array




def calculate_goes_gradient(
    goes_obj,
    channel: str = "xrsb",
    smooth_window: float = 0,
    smooth_method: str = "gaussian"
) -> np.ndarray:
    """
    Vypočíta časový gradient z originálneho SunPy TimeSeries objektu s možnosťou vyhladenia toku.

    Parametre:
        goes_obj: SunPy TimeSeries objekt (vrátený z get_goes_flux)
        channel (str): Kanál pre výpočet gradientu ("xrsb" alebo "xrsa")
        smooth_window (float/int): Úroveň vyhladenia (0 = bez vyhladenia):
                                   - Pre "gaussian": hodnota sigma (štandardná odchýlka filtra v počte bodov, napr. 3-10).
                                   - Pre "moving_avg" a "savgol": veľkosť okna v počte bodov (napr. 15 alebo 31).
        smooth_method (str): Metóda vyhladenia:
                             - "gaussian" (odporúčané, plynulý Gaussov filter)
                             - "moving_avg" (kĺzavý priemer)
                             - "savgol" (Savitzky-Golay filter)

    Návratová hodnota:
        np.ndarray: Jednorozmerné pole s vypočítaným gradientom v jednotkách W/(m²·s).
    """
    # Získanie hodnôt toku z objektu
    flux = goes_obj.quantity(channel).value.copy()

    # Vyhladenie toku pred výpočtom gradientu (potláča šum pred deriváciou)
    if smooth_window > 0:
        if smooth_method == "gaussian":
            flux = gaussian_filter1d(flux, sigma=smooth_window)
        elif smooth_method == "moving_avg":
            flux = uniform_filter1d(flux, size=int(smooth_window))
        elif smooth_method == "savgol":
            from scipy.signal import savgol_filter
            window_length = int(smooth_window)
            # Savitzky-Golay vyžaduje nepárne číslo okna > polyorder
            if window_length % 2 == 0:
                window_length += 1
            if window_length < 5:
                window_length = 5
            flux = savgol_filter(flux, window_length=window_length, polyorder=2)
        else:
            raise ValueError(f"Neznáma metóda vyhladenia: '{smooth_method}'. Použite 'gaussian', 'moving_avg' alebo 'savgol'.")

    # Získanie časovej osi z Astropy Time (ktorá je vnútri objektu) a prepočet na sekundy
    times = goes_obj.time
    time_seconds = (times - times[0]).to_value('s')

    # Výpočet gradientu (derivácie) vzhľadom na čas v sekundách
    gradient = np.gradient(flux, time_seconds)

    return gradient


from datetime import datetime
import numpy as np
from astropy.time import Time


def slice_and_calculate_h_alpha(light_obj, t_start, t_end, center_idx: int = 1379, half_width: int = 2):
    """
    Slices a Light object based on start and end times, and extracts/calculates
    the H-alpha intensities around the specified spectral index range.

    Parameters:
        light_obj: The Light spectrometer object (contains .data and .t_range).
        t_start: Start of the slice window (datetime, str, or Astropy/Pandas wrapper).
        t_end: End of the slice window (datetime, str, or Astropy/Pandas wrapper).
        center_idx (int): The central H-alpha pixel index (default: 1379).
        half_width (int): Index radius around the center (default: 2, yielding a range of center_idx - 2 to center_idx + 2).

    Returns:
        tuple: (timerange, h_alpha_integrated, h_alpha_raw)
            - timerange (astropy.time.Time): Sliced time range as an Astropy Time object.
            - h_alpha_integrated (np.ndarray): 1D array of summed intensities across the spectral range for each time step.
            - h_alpha_raw (np.ndarray): 2D slice of raw intensities (time rows x spectral columns).
    """

    # 1. Flexible time parsing helper
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

    # Parse inputs to datetime objects
    dt_start = parse_to_datetime(t_start)
    dt_end = parse_to_datetime(t_end)

    # 2. Clean and convert t_range to a numpy array of datetimes
    t_range_array = np.array(light_obj.t_range)

    # 3. Find the exact insertion indices inside the timeline
    start_idx = np.searchsorted(t_range_array, dt_start, side='left')
    end_idx = np.searchsorted(t_range_array, dt_end, side='right')

    # 4. Slice the time array and convert to Astropy Time
    subarray = t_range_array[start_idx:end_idx]
    timerange = Time(subarray)

    # 5. Define spectral bounds (default center 1379 with half_width 2 means indices 1377 to 1381 inclusive)
    start_col = center_idx - half_width
    end_col = center_idx + half_width + 1  # Add 1 because python slicing is exclusive

    # Bounds-check the column indices against the data dimensions
    num_cols = light_obj.data.shape[1]
    start_col = max(0, start_col)
    end_col = min(num_cols, end_col)

    # 6. Slice the 2D spectral data array [time_interval, wavelength_interval]
    h_alpha_raw = light_obj.data[start_idx:end_idx, start_col:end_col]

    # Calculate integrated (summed) intensity across the spectral line for each time step
    h_alpha_integrated = np.mean(h_alpha_raw, axis=1)

    return timerange, h_alpha_integrated / h_alpha_integrated[5]


def sum_circle_values(images_array,
                      crop_bounds=None,
                      circle_center=None,
                      circle_radius=None):
    """
    Vectorized function: crops images and sums values inside a circular mask.
    Raises ValueError if any structural bounds or dimensions are missing.
    """
    # Throw an error if any of the required spatial parameters are omitted
    if crop_bounds is None or circle_center is None or circle_radius is None:
        raise ValueError(
            f"Missing required arguments! You must explicitly provide: \n"
            f"  - crop_bounds (tuple)\n"
            f"  - circle_center (tuple)\n"
            f"  - circle_radius (int)\n"
            f"Received: crop_bounds={crop_bounds}, circle_center={circle_center}, circle_radius={circle_radius}"
        )

    x_min, x_max, y_min, y_max = crop_bounds
    ny, nx = y_max - y_min, x_max - x_min

    # Build circular mask once
    y, x = np.ogrid[:ny, :nx]
    mask = (x - circle_center[0]) ** 2 + (y - circle_center[1]) ** 2 <= circle_radius ** 2

    # Crop all images at once
    cropped_images = images_array[:, y_min:y_max, x_min:x_max]

    # Apply mask with broadcasting and sum across spatial dimensions
    values = np.sum(cropped_images * mask, axis=(1, 2))
    return values

def run_analysis(calibrated_data):
    print("  Running analysis")
    return {"result": 42}
