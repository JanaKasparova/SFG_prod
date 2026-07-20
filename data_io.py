import os
from astropy.io import fits
import numpy as np
from pathlib import Path
from typing import List, Optional
from FICUS.PYTHON.OCAS_lib import Light, Calibration, Measurement
from FICUS.PYTHON.NormalizationModule import Normalization, Linearity
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, Button
from datetime import datetime, timedelta
from astropy.visualization import time_support
from sunpy import timeseries as ts
from sunpy.net import Fido
from sunpy.net import attrs as a
from plotting import *
from processing import *
from analysis import *
import re
from astropy.time import Time


def load_fits_img(path: str, logger=None):
    """
    Load a single FITS file and return its data as a numpy array.

    Parameters:
        path : str
            Path to the FITS file.

    Returns:
        np.ndarray
            The data from the FITS file as a numpy array.
    """
    with fits.open(path) as hdul:
        data = hdul[0].data
    if logger is not None:
        logger.debug(f"Loaded FITS image from {path}")
        logger.debug(f"Shape of array: {data.shape}")

    return np.array(data)


def load_fits(folder_path, logger=None):
    """
    Load all FITS files from a folder.

    Parameters:
        folder_path : str
            Path to the folder containing FITS files.

    Returns:
        list of np.ndarray
            List of FITS images (as numpy arrays).
    """
    fits_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.fits', '.fit'))]
    fits_files.sort()  # optional: sort alphabetically

    images = []
    for f in fits_files:
        path = os.path.join(folder_path, f)
        with fits.open(path) as hdul:
            data = hdul[0].data
            if data is None:
                continue  # skip empty HDUs
            images.append(np.array(data))
    images = np.array(images)
    if logger is not None:
        logger.debug(f"Loaded {len(images)} FITS files from {folder_path}")
        logger.debug(f"Shape of stacked array: {images.shape}")
    return images


from pathlib import Path
from typing import List, Optional, Union


def get_hdf_paths(
        folder: str,
        CD: bool = False,
        return_path: bool = True,
        logger=None
) -> List[Optional[Union[Path, str]]]:
    """
    Parameters
    ----------
    folder : str
        Directory containing .hdf files.
    CD : bool
        - False -> return all .hdf files
        - True  -> return [C_file, D_file]
    return_path : bool
        - True  -> return Path objects
        - False -> return strings

    Returns
    -------
    list
        - If CD is False: list of Path or str
        - If CD is True: [C_file, D_file], missing entries are None
    """
    folder_path = Path(folder)
    hdf_files = list(folder_path.glob("*.hdf"))

    def _out(p: Optional[Path]) -> Optional[Union[Path, str]]:
        if p is None:
            return None
        return p if return_path else str(p)

    if not CD:
        temp = [_out(p) for p in hdf_files]
        if logger is not None:
            logger.debug(f"Loaded paths of {len(temp)} HDF files from {folder}")
        return temp

    c_file = None
    d_file = None

    for f in hdf_files:
        name = f.name
        if "_HR4C" in name:
            c_file = f
        elif "_HR4D" in name:
            d_file = f
    if logger is not None:
        logger.debug(f"Loaded paths of C and D (HDF) values from {folder}")
    return [_out(c_file), _out(d_file)]


def load_hdf(hdf_dir, logger=None):
    print(f"  Reading HDF5 from {hdf_dir}")
    return


def load_hdf_light(path: str, idx: int, logger=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Load C and D light HDF files (spectrum) from a directory and return their data arrays.

    Parameters
    ----------
    hdf_dir : str
        Directory containing C and D .hdf files.
    logger : optional
        Logger with .info() method; falls back to print if None.

    Returns
    -------
    (data_C, data_D) : tuple of np.ndarray
    """
    msg = f"Reading HDF5 as C and D light from {path}"
    if logger is not None:
        logger.info(msg)

    # Get [C, D] file paths as strings
    c_path, d_path = get_hdf_paths(
        path,
        CD=True,
        return_path=False
    )

    if c_path is None or d_path is None:
        raise FileNotFoundError(
            f"Could not find both C and D HDF files in {path}"
        )
    if type(idx) is not int:
        raise TypeError(f"Invalid type(idx) = {type(idx)} != int")

    # Load HDF files using Light
    mC = Light(c_path, idx)
    mD = Light(d_path, idx)

    return mC, mD



def load_flats(flat_dir, logger=None):
    "dummy function"
    print(f"  Reading flats from {flat_dir} DUMMY")
    return {"dummy_flats": None}


def load_darks(dark_dir, logger=None):
    "dummy function"
    print(f"  Reading darks from {dark_dir} DUMMY")
    return {"dummy_darks": None}


def select_roi_with_sliders(imgs: np.ndarray, vmax: float = 50000, cmap: str = "gray"):
    """
    Launches an interactive UI window with sliders to dynamically change
    xmin, xmax, ymin, and ymax boundaries. Includes a button to output
    copy-pasteable code parameters directly to your console screen.
    """
    # 1. Isolate a single 2D frame as our structural template reference
    if imgs.ndim == 3:
        ref_frame = imgs[0]  # Sample the very first frame of the stack
    elif imgs.ndim == 2:
        ref_frame = imgs
    else:
        raise ValueError(f"Unexpected image dimensions: {imgs.shape}")

    h, w = ref_frame.shape[:2]

    # 2. Initialize the Figure Layout Window
    fig, ax = plt.subplots(figsize=(9, 8))
    # Leave generous empty margin real estate at bottom for UI sliders controls
    plt.subplots_adjust(bottom=0.32)

    # Render base map
    im = ax.imshow(ref_frame, cmap=cmap, origin="lower", vmax=vmax)
    ax.set_title("Interactive ROI Target Box Definition Window", fontsize=12, fontweight="bold")

    # Establish initial bounding box guesses (centered at 25% to 75%)
    init_xmin, init_xmax = int(w * 0.25), int(w * 0.75)
    init_ymin, init_ymax = int(h * 0.25), int(h * 0.75)

    # Anchor visual overlay patch tracking vector template
    rect_patch = Rectangle(
        xy=(init_xmin, init_ymin),
        width=(init_xmax - init_xmin),
        height=(init_ymax - init_ymin),
        color="red",
        fill=False,
        linewidth=2.0
    )
    ax.add_patch(rect_patch)

    # 3. Setup UI Sliders Geometry [left, bottom, width, height]
    ax_xmin = plt.axes([0.15, 0.24, 0.70, 0.03])
    ax_xmax = plt.axes([0.15, 0.19, 0.70, 0.03])
    ax_ymin = plt.axes([0.15, 0.14, 0.70, 0.03])
    ax_ymax = plt.axes([0.15, 0.09, 0.70, 0.03])

    slider_xmin = Slider(ax_xmin, "X Min", 0, w, valinit=init_xmin, valstep=1)
    slider_xmax = Slider(ax_xmax, "X Max", 0, w, valinit=init_xmax, valstep=1)
    slider_ymin = Slider(ax_ymin, "Y Min", 0, h, valinit=init_ymin, valstep=1)
    slider_ymax = Slider(ax_ymax, "Y Max", 0, h, valinit=init_ymax, valstep=1)

    # 4. Live Real-Time Updating Trigger Function
    def update_bounds(val):
        x_min_val = int(slider_xmin.val)
        x_max_val = int(slider_xmax.val)
        y_min_val = int(slider_ymin.val)
        y_max_val = int(slider_ymax.val)

        # Logical Guardrails: prevent crossing borders which flips width definitions inverted
        if x_min_val >= x_max_val:
            x_max_val = x_min_val + 1
        if y_min_val >= y_max_val:
            y_max_val = y_min_val + 1

        # Redefine active rendering attributes on the live patch
        rect_patch.set_xy((x_min_val, y_min_val))
        rect_patch.set_width(x_max_val - x_min_val)
        rect_patch.set_height(y_max_val - y_min_val)

        # Request immediate canvas update redraw refresh cycle
        fig.canvas.draw_idle()

    # Link events to update function
    slider_xmin.on_changed(update_bounds)
    slider_xmax.on_changed(update_bounds)
    slider_ymin.on_changed(update_bounds)
    slider_ymax.on_changed(update_bounds)

    # 5. Dynamic Console Code Generator Button Module
    ax_button = plt.axes([0.35, 0.02, 0.30, 0.04])  # Anchor location dimensions
    export_btn = Button(ax_button, "Copy-Paste ROI Variables", color="#2ecc71", hovercolor="#27ae60")

    def print_current_coordinates(event):
        x_m = int(slider_xmin.val)
        x_M = int(slider_xmax.val)
        y_m = int(slider_ymin.val)
        y_M = int(slider_ymax.val)

        # Generate clean string outputs directly to console standard output stream
        print(f"\n" + "=" * 45)
        print("  TARGET ROI COORDINATES DETECTED  ")
        print("=" * 45)
        print(f"xmin={x_m}, xmax={x_M}, ymin={y_m}, ymax={y_M}")
        print("-" * 45)
        print(f"Use directly inside your pipeline function:")
        print(
            f"means, stds = analyze_and_plot_rect(images, xmin={x_m}, xmax={x_M}, ymin={y_m}, ymax={y_M}, num_plots=4)")
        print("=" * 45 + "\n")

    export_btn.on_clicked(print_current_coordinates)

    plt.show()

    # Keep references alive inside system memory tracking registries so widgets don't freeze up
    return [slider_xmin, slider_xmax, slider_ymin, slider_ymax, export_btn]


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


def compile_directory_timestamps(directory_path: str, logger=None) -> np.ndarray:
    """
    Scans a directory for files matching the SlitJaw timestamp pattern,
    extracts their times, and returns a sorted NumPy array of Astropy Time objects.
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
        return np.array([], dtype=object)

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

    # 3. Pack into a standard object-type NumPy array
    time_array = np.array(time_objects_list, dtype=object)

    if logger:
        logger.info(f"Successfully compiled {len(time_array)} time objects into NumPy array.")

    return time_array


from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sunpy import timeseries as ts
from sunpy.net import Fido
from sunpy.net import attrs as a


def get_goes_flux(t_start, t_end, filename: str, satellite: int = 16, buffer_hours: float = 2.0):
    """
    Downloads, cleans, and plots GOES XRS flux data for a window around the
    provided start and end times, saving the plot directly to a file.

    Parameters:
        t_start: Start of the event (datetime, str, or Astropy Time).
        t_end: End of the event (datetime, str, or Astropy Time).
        filename (str): The path/filename where the plot should be saved (e.g., 'goes_plot.png').
        satellite (int): GOES satellite number (Default: 16).
        buffer_hours (float): Hours of padding to query/plot before and after.

    Returns:
        tuple: (df_event, goes_event)
            - df_event (pd.DataFrame): Truncated flux DataFrame for the exact event window.
            - goes_event (sunpy.timeseries.TimeSeries): Truncated SunPy TimeSeries object.
    """

    # Flexible parser to convert inputs into standard Python datetimes
    def parse_to_datetime(t_input):
        if isinstance(t_input, datetime):
            return t_input
        if hasattr(t_input, 'datetime'):  # Handles pandas/astropy wrappers
            return t_input.datetime
        if isinstance(t_input, str):
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d %H:%M:%S"):
                try:
                    return datetime.strptime(t_input, fmt)
                except ValueError:
                    continue
        raise TypeError(f"Unsupported time format: {type(t_input)}")

    dt_start = parse_to_datetime(t_start)
    dt_end = parse_to_datetime(t_end)

    query_start = dt_start - timedelta(hours=buffer_hours)
    query_end = dt_end + timedelta(hours=buffer_hours)

    print(f"Searching for GOES-{satellite} data from {query_start} to {query_end}...")

    # Search for data (high-resolution 1-second first)
    search_result = Fido.search(
        a.Time(query_start.isoformat(), query_end.isoformat()),
        a.Instrument("XRS"),
        a.goes.SatelliteNumber(satellite),
        a.Resolution("flx1s")
    )

    if not search_result:
        print("1s resolution unavailable, trying standard resolution...")
        search_result = Fido.search(
            a.Time(query_start.isoformat(), query_end.isoformat()),
            a.Instrument("XRS"),
            a.goes.SatelliteNumber(satellite)
        )
        if not search_result:
            raise RuntimeError("Data for this time range were not found.")

    downloaded_files = Fido.fetch(search_result, progress=True)

    # Load and clean data quality flags
    goes_ts = ts.TimeSeries(downloaded_files, concatenate=True)
    df = goes_ts.to_dataframe()

    if "xrsa_quality" in df.columns and "xrsb_quality" in df.columns:
        df = df[(df["xrsa_quality"] == 0) & (df["xrsb_quality"] == 0)]
        goes_ts = ts.TimeSeries(df, goes_ts.meta, goes_ts.units)

    # Truncate for the plot (including buffer)
    goes_buffered = goes_ts.truncate(query_start.isoformat(), query_end.isoformat())
    df_buffered = goes_buffered.to_dataframe()

    # Render and save the plot
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=120)
    ax.plot(df_buffered.index, df_buffered["xrsb"], label="XRS-B (1-8 Å)", color="#d63031", lw=1.6)
    ax.plot(df_buffered.index, df_buffered["xrsa"], label="XRS-A (0.5-4 Å)", color="#0984e3", lw=1.2, alpha=0.7)

    ax.axvspan(dt_start, dt_end, color="#fdcb6e", alpha=0.18, label="Observation Window")
    ax.axvline(dt_start, color="#e17055", linestyle="--", lw=1.5)
    ax.axvline(dt_end, color="#e17055", linestyle="--", lw=1.5)

    ax.set_yscale("log")
    ax.set_ylabel("Flux (W / m²)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (UTC)", fontsize=11, fontweight="bold")
    ax.set_title(f"GOES-{satellite} Solar X-Ray Flux", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(loc="upper left")

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {filename}")

    # -------------------------------------------------------------
    # PREPARE OUTPUTS (truncated precisely to the event duration)
    # -------------------------------------------------------------
    # 1. Clean flux DataFrame
    df_event = df.loc[dt_start:dt_end]

    # 2. Complete SunPy TimeSeries object (your GOES object)
    goes_event = goes_ts.truncate(dt_start.isoformat(), dt_end.isoformat())

    return df_event, goes_event


import h5py
import numpy as np


def make_metadata_dict(path_to_hdf: str) -> list:
    """
    Recursively walks an HDF5 file to build a list of metadata dictionaries
    containing attributes, names, shapes, and types for every group and dataset.

    Parameters:
        path_to_hdf (str): Path to the HDF5 file.

    Returns:
        list: A list of dicts containing the metadata for each node in the file.
    """
    metadata_list = []

    def decode_value(val):
        """Helper function to decode bytes and numpy byte strings into UTF-8 strings."""
        if isinstance(val, bytes):
            return val.decode("utf-8", errors="ignore")
        if isinstance(val, np.ndarray):
            if val.dtype.kind in ('S', 'U'):  # Byte or unicode strings in numpy arrays
                return [v.decode("utf-8", errors="ignore") if isinstance(v, bytes) else v for v in val]
            if val.size == 1:  # Flatten single-element numpy arrays
                return val.item()
            return val.tolist()
        return val

    def visitor_function(name, obj):
        """Callback function executed for every group and dataset found in the HDF5 file."""
        # Extract all custom attributes and decode them
        metadata = {k: decode_value(v) for k, v in obj.attrs.items()}

        # Add structural information
        metadata["name"] = f"/{name}"  # Absolute path inside the HDF5 structure
        metadata["type"] = "dataset" if isinstance(obj, h5py.Dataset) else "group"

        # If it's a dataset, record its physical dimensions and data type
        if isinstance(obj, h5py.Dataset):
            metadata["shape"] = obj.shape
            metadata["dtype"] = str(obj.dtype)

        metadata_list.append(metadata)

    # Open the file in read-only mode and recursively traverse the tree
    with h5py.File(path_to_hdf, "r") as hdf_file:
        # Manually parse the root group "/" which visititems skips
        root_metadata = {k: decode_value(v) for k, v in hdf_file.attrs.items()}
        root_metadata["name"] = "/"
        root_metadata["type"] = "group"
        metadata_list.append(root_metadata)

        # Recursively visit every subdirectory and dataset
        hdf_file.visititems(visitor_function)

    return metadata_list


