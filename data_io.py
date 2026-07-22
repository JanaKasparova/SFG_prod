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


def get_fits_filepaths(folder_path):
    """Returns a sorted list of FITS file paths in a given folder."""
    if not os.path.exists(folder_path):
        return []
    valid_exts = ('.fits', '.fit', '.fts', '.FITS', '.FIT', '.FTS')
    files = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.endswith(valid_exts)
    ]
    return sorted(files)


def load_fits_batch(file_paths: list[str]) -> np.ndarray:
    """Load only a specific batch of FITS file paths into RAM."""
    batch_data = []
    for path in file_paths:
        with fits.open(path) as hdul:
            data = hdul[0].data
            if data is not None:
                batch_data.append(data.astype(np.float32))
    return np.array(batch_data)


def load_sampled_fits(folder_path: str, max_samples: int = 400, logger=None) -> np.ndarray:
    """
    Finds all FITS files in folder_path, uniformly samples up to max_samples frames,
    and loads only those sampled frames into RAM.
    """
    fits_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(('.fits', '.fit'))
    ]
    fits_files.sort()

    total_files = len(fits_files)
    if total_files == 0:
        raise FileNotFoundError(f"No FITS files found in {folder_path}")

    # Determine uniform sample indices
    if total_files <= max_samples:
        selected_paths = fits_files
        if logger:
            logger.info(f"📂 Total files ({total_files}) <= {max_samples}. Loading all frames for ROI selection.")
    else:
        indices = np.linspace(0, total_files - 1, max_samples, dtype=int)
        selected_paths = [fits_files[idx] for idx in indices]
        if logger:
            logger.info(f"📊 Uniformly sampling {max_samples} frames out of {total_files} total FITS files.")

    # Load only selected frames
    sampled_images = []
    for path in selected_paths:
        with fits.open(path) as hdul:
            data = hdul[0].data
            if data is not None:
                sampled_images.append(data.astype(np.float32))

    return np.array(sampled_images)


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
    logger : optional
        Logger with .debug() or .warning() methods.

    Returns
    -------
    list
        - If CD is False: list of Path or str
        - If CD is True: [C_file, D_file], missing entries are None
    """
    folder_path = Path(folder)

    def _out(p: Optional[Path]) -> Optional[Union[Path, str]]:
        if p is None:
            return None
        return p if return_path else str(p)

    # Handled non-existent directory safely
    if not folder_path.exists() or not folder_path.is_dir():
        if logger is not None:
            logger.warning(f"⚠️ Directory '{folder}' does not exist.")
        return [None, None] if CD else []

    # Case-insensitive match for HDF file extensions (.hdf, .hdf5, .h5)
    hdf_files = [
        f for f in folder_path.iterdir()
        if f.is_file() and f.suffix.lower() in ('.hdf', '.hdf5', '.h5')
    ]

    if not CD:
        temp = [_out(p) for p in hdf_files]
        if logger is not None:
            logger.debug(f"Loaded paths of {len(temp)} HDF files from {folder}")
        return temp

    c_file = None
    d_file = None

    # Case-insensitive check for _HR4C and _HR4D tags
    for f in hdf_files:
        name_upper = f.name.upper()
        if "_HR4C" in name_upper and c_file is None:
            c_file = f
        elif "_HR4D" in name_upper and d_file is None:
            d_file = f

    if logger is not None:
        c_status = "found" if c_file else "MISSING (None)"
        d_status = "found" if d_file else "MISSING (None)"
        logger.debug(f"HDF paths search in '{folder}': C={c_status}, D={d_status}")

    # Explicitly returns [C_path_or_None, D_path_or_None]
    return [_out(c_file), _out(d_file)]


def load_hdf(hdf_dir, logger=None):
    print(f"  Reading HDF5 from {hdf_dir}")
    return


def load_hdf_light(path: str, idx: int, logger=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Load C and D light HDF files (spectrum) from a directory and return their data arrays/objects.
    If either C or D file path is missing (None), returns a NumPy array of length 3840 for that path.

    Parameters
    ----------
    path : str
        Directory containing C and D .hdf files.
    idx : int
        Index of the target dataset inside the HDF5 file.
    logger : optional
        Logger with .info() and .warning() methods; falls back to print if None.

    Returns
    -------
    (data_C, data_D) : tuple
        Tuple containing Light objects for valid paths or np.ndarray of length 3840 for missing paths.
    """
    if not isinstance(idx, int):
        raise TypeError(f"Invalid type(idx) = {type(idx)} != int")

    msg = f"Reading HDF5 as C and D light from {path}"
    if logger is not None:
        logger.info(msg)

    # Get [C, D] file paths as strings
    c_path, d_path = get_hdf_paths(
        path,
        CD=True,
        return_path=False
    )

    # Process C file
    if c_path is not None:
        mC = Light(c_path, idx)
    else:
        warn_msg = f"⚠️ C HDF5 path is None in {path}. Returning empty NumPy array of length 3840."
        if logger is not None:
            logger.warning(warn_msg)
        else:
            print(warn_msg)
        mC = np.zeros(3840)

    # Process D file
    if d_path is not None:
        mD = Light(d_path, idx)
    else:
        warn_msg = f"⚠️ D HDF5 path is None in {path}. Returning empty NumPy array of length 3840."
        if logger is not None:
            logger.warning(warn_msg)
        else:
            print(warn_msg)
        mD = np.zeros(3840)

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


import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sunpy.net import Fido, attrs as a
from sunpy import timeseries as ts


def get_goes_flux(t_start, t_end, filename: str, satellite: int = 16, buffer_hours: float = 2.0):
    """
    Downloads, cleans, categorizes, and plots GOES XRS flux data for a window
    around provided start and end times, saving the plot directly to a file.

    Parameters:
        t_start: Start of the event (datetime, str, or Astropy Time).
        t_end: End of the event (datetime, str, or Astropy Time).
        filename (str): The path/filename where the plot should be saved.
        satellite (int): GOES satellite number (Default: 16).
        buffer_hours (float): Hours of padding to query/plot before and after.

    Returns:
        tuple: (df_event, goes_event)
            - df_event (pd.DataFrame): Truncated flux DataFrame for the exact event window.
            - goes_event (sunpy.timeseries.TimeSeries): Truncated SunPy TimeSeries object.
    """

    # Helper function to categorize peak flux based on SWPC NOAA standards
    def classify_flare(peak_flux_w_m2):
        if peak_flux_w_m2 is None or np.isnan(peak_flux_w_m2) or peak_flux_w_m2 <= 0:
            return "Unknown"
        if peak_flux_w_m2 < 1e-7:
            return f"A{peak_flux_w_m2 / 1e-8:.1f}"
        elif peak_flux_w_m2 < 1e-6:
            return f"B{peak_flux_w_m2 / 1e-7:.1f}"
        elif peak_flux_w_m2 < 1e-5:
            return f"C{peak_flux_w_m2 / 1e-6:.1f}"
        elif peak_flux_w_m2 < 1e-4:
            return f"M{peak_flux_w_m2 / 1e-5:.1f}"
        else:
            return f"X{peak_flux_w_m2 / 1e-4:.1f}"

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

    # -------------------------------------------------------------
    # EXTRACT EVENT WINDOW DATA & CALCULATE FLARE CATEGORY
    # -------------------------------------------------------------
    df_event = df.loc[dt_start:dt_end]
    goes_event = goes_ts.truncate(dt_start.isoformat(), dt_end.isoformat())

    if "xrsb" in df_event.columns and not df_event["xrsb"].empty:
        peak_xrsb = df_event["xrsb"].max()
        flare_category = classify_flare(peak_xrsb)
    else:
        peak_xrsb = 0.0
        flare_category = "N/A"

    print(f"--> Event Peak XRS-B Flux: {peak_xrsb:.2e} W/m² | Categorized Flare Class: {flare_category}")

    # Attach class metadata directly to the SunPy TimeSeries object
    if hasattr(goes_event, 'meta'):
        goes_event.meta['flare_class'] = flare_category
        goes_event.meta['peak_xrsb'] = peak_xrsb

    # Truncate for plotting (including buffer)
    goes_buffered = goes_ts.truncate(query_start.isoformat(), query_end.isoformat())
    df_buffered = goes_buffered.to_dataframe()

    # -------------------------------------------------------------
    # RENDER PLOT WITH NOAA FLARE CLASS THRESHOLDS
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=120)
    ax.plot(df_buffered.index, df_buffered["xrsb"], label="XRS-B (1-8 Å)", color="#d63031", lw=1.6)
    ax.plot(df_buffered.index, df_buffered["xrsa"], label="XRS-A (0.5-4 Å)", color="#0984e3", lw=1.2, alpha=0.7)

    ax.axvspan(dt_start, dt_end, color="#fdcb6e", alpha=0.18, label="Observation Window")
    ax.axvline(dt_start, color="#e17055", linestyle="--", lw=1.5)
    ax.axvline(dt_end, color="#e17055", linestyle="--", lw=1.5)

    # NOAA Class Reference Horizontal Threshold Lines
    flare_classes = {
        "A": 1e-8,
        "B": 1e-7,
        "C": 1e-6,
        "M": 1e-5,
        "X": 1e-4
    }

    y_min_data = min(df_buffered["xrsb"].min(), df_buffered["xrsa"].min())
    y_max_data = max(df_buffered["xrsb"].max(), df_buffered["xrsa"].max())

    for cls_name, cls_val in flare_classes.items():
        if y_min_data * 0.5 <= cls_val <= y_max_data * 2.0:
            ax.axhline(cls_val, color="gray", linestyle=":", alpha=0.35, lw=0.9)
            ax.text(df_buffered.index[0], cls_val * 1.15, f" Class {cls_name}",
                    color="gray", fontsize=8, fontweight="bold", alpha=0.7)

    ax.set_yscale("log")
    ax.set_ylabel("Flux (W / m²)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (UTC)", fontsize=11, fontweight="bold")
    ax.set_title(f"GOES-{satellite} Solar X-Ray Flux — Peak Class: {flare_category}", fontsize=12, fontweight="bold",
                 pad=12)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(loc="upper left")

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {filename}")

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


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, Button


def select_rectangle_roi(
        imgs: np.ndarray,
        title: str = "Rectangle ROI Selector",
        vmax_default: float = 2.0,
        cmap: str = "inferno",
        var_prefix: str = "CALIB"
) -> tuple[int, int, int, int]:
    """
    Launches an interactive GUI with sliders to tune xmin, xmax, ymin, ymax,
    frame scrubbing, and image brightness (vmax).

    Returns (xmin, xmax, ymin, ymax) as a 4-element tuple upon window closure.
    """
    # 1. Normalize array geometry (handle 2D or 3D inputs)
    if imgs.ndim == 2:
        stack = imgs[np.newaxis, ...]
    elif imgs.ndim == 3:
        stack = imgs
    else:
        raise ValueError(f"Invalid image dimensions: {imgs.shape}")

    n_frames, h, w = stack.shape

    # Initial crop values (Centered 25% to 75%)
    init_xmin, init_xmax = int(w * 0.25), int(w * 0.75)
    init_ymin, init_ymax = int(h * 0.25), int(h * 0.75)
    init_frame = 0

    # 2. Setup Figure Layout
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.38)  # Leave room for control panel below

    im = ax.imshow(stack[init_frame], cmap=cmap, origin="lower", vmin=0, vmax=vmax_default)
    ax.set_title(f"{title} - Frame {init_frame + 1}/{n_frames}", fontsize=11, fontweight="bold")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")

    # Overlay Rectangle Patch
    rect_patch = Rectangle(
        xy=(init_xmin, init_ymin),
        width=(init_xmax - init_xmin),
        height=(init_ymax - init_ymin),
        color="cyan",
        fill=False,
        linewidth=2.0,
        linestyle="--"
    )
    ax.add_patch(rect_patch)

    # 3. Widget Axes Placement [left, bottom, width, height]
    ax_frame = plt.axes([0.15, 0.30, 0.70, 0.025])
    ax_vmax = plt.axes([0.15, 0.26, 0.70, 0.025])
    ax_xmin = plt.axes([0.15, 0.20, 0.70, 0.025])
    ax_xmax = plt.axes([0.15, 0.16, 0.70, 0.025])
    ax_ymin = plt.axes([0.15, 0.12, 0.70, 0.025])
    ax_ymax = plt.axes([0.15, 0.08, 0.70, 0.025])

    # 4. Sliders Creation
    slider_frame = Slider(ax_frame, "Frame", 0, n_frames - 1, valinit=init_frame, valstep=1)
    slider_vmax = Slider(ax_vmax, "VMax", 0.1, np.nanmax(stack), valinit=vmax_default)
    slider_xmin = Slider(ax_xmin, "X Min", 0, w - 1, valinit=init_xmin, valstep=1)
    slider_xmax = Slider(ax_xmax, "X Max", 1, w, valinit=init_xmax, valstep=1)
    slider_ymin = Slider(ax_ymin, "Y Min", 0, h - 1, valinit=init_ymin, valstep=1)
    slider_ymax = Slider(ax_ymax, "Y Max", 1, h, valinit=init_ymax, valstep=1)

    # Update Callback
    def update(val):
        x_min = int(slider_xmin.val)
        x_max = int(slider_xmax.val)
        y_min = int(slider_ymin.val)
        y_max = int(slider_ymax.val)

        # Enforce Logical Boundaries
        if x_min >= x_max:
            x_max = x_min + 1
        if y_min >= y_max:
            y_max = y_min + 1

        # Live Image Update
        f_idx = int(slider_frame.val)
        v_val = slider_vmax.val
        im.set_data(stack[f_idx])
        im.set_clim(0, v_val)
        ax.set_title(f"{title} - Frame {f_idx + 1}/{n_frames}", fontsize=11, fontweight="bold")

        # Live Bounding Box Update
        rect_patch.set_xy((x_min, y_min))
        rect_patch.set_width(x_max - x_min)
        rect_patch.set_height(y_max - y_min)

        fig.canvas.draw_idle()

    for s in [slider_frame, slider_vmax, slider_xmin, slider_xmax, slider_ymin, slider_ymax]:
        s.on_changed(update)

    # 5. Export Button
    ax_btn = plt.axes([0.35, 0.02, 0.30, 0.04])
    btn_export = Button(ax_btn, "📋 Export Config Code", color="#2ecc71", hovercolor="#27ae60")

    def export_code(event):
        x_min = int(slider_xmin.val)
        x_max = int(slider_xmax.val)
        y_min = int(slider_ymin.val)
        y_max = int(slider_ymax.val)

        print("\n" + "=" * 55)
        print(f"    COPY-PASTE CONFIGURATION ({var_prefix})")
        print("=" * 55)
        print(f"{var_prefix}_XMIN, {var_prefix}_XMAX = {x_min}, {x_max}")
        print(f"{var_prefix}_YMIN, {var_prefix}_YMAX = {y_min}, {y_max}")
        print("=" * 55 + "\n")

    btn_export.on_clicked(export_code)

    # Store UI elements on the figure object to avoid garbage collection
    fig._ui_widgets = [slider_frame, slider_vmax, slider_xmin, slider_xmax, slider_ymin, slider_ymax, btn_export]

    # Execution blocks here until user closes the window
    plt.show()

    # Read final values from sliders after closure
    final_xmin = int(slider_xmin.val)
    final_xmax = int(slider_xmax.val)
    final_ymin = int(slider_ymin.val)
    final_ymax = int(slider_ymax.val)

    # Boundary safety checks
    if final_xmin >= final_xmax:
        final_xmax = final_xmin + 1
    if final_ymin >= final_ymax:
        final_ymax = final_ymin + 1

    return final_xmin, final_xmax, final_ymin, final_ymax


def select_circle_roi(
        imgs: np.ndarray,
        title: str = "Circular Eruption Mask Selector",
        vmax_default: float = 2.0,
        cmap: str = "inferno"
) -> tuple[tuple[int, int], int]:
    """
    Launches an interactive GUI with sliders to set Center X (xc), Center Y (yc),
    and Radius (r) for circular analysis masks.

    Returns ((center_x, center_y), radius) upon window closure.
    """
    if imgs.ndim == 2:
        stack = imgs[np.newaxis, ...]
    elif imgs.ndim == 3:
        stack = imgs
    else:
        raise ValueError(f"Invalid image dimensions: {imgs.shape}")

    n_frames, h, w = stack.shape

    init_xc, init_yc = int(w / 2), int(h / 2)
    init_r = int(min(w, h) * 0.3)
    init_frame = 0

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.34)

    im = ax.imshow(stack[init_frame], cmap=cmap, origin="lower", vmin=0, vmax=vmax_default)
    ax.set_title(f"{title} - Frame {init_frame + 1}/{n_frames}", fontsize=11, fontweight="bold")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")

    # Overlay Circle Patch
    circle_patch = Circle((init_xc, init_yc), init_r, color="cyan", fill=False, linewidth=2.0, linestyle="--")
    ax.add_patch(circle_patch)

    # Controls Geometry
    ax_frame = plt.axes([0.15, 0.25, 0.70, 0.025])
    ax_vmax = plt.axes([0.15, 0.21, 0.70, 0.025])
    ax_xc = plt.axes([0.15, 0.16, 0.70, 0.025])
    ax_yc = plt.axes([0.15, 0.12, 0.70, 0.025])
    ax_r = plt.axes([0.15, 0.08, 0.70, 0.025])

    slider_frame = Slider(ax_frame, "Frame", 0, n_frames - 1, valinit=init_frame, valstep=1)
    slider_vmax = Slider(ax_vmax, "VMax", 0.1, np.nanmax(stack), valinit=vmax_default)
    slider_xc = Slider(ax_xc, "Center X", 0, w, valinit=init_xc, valstep=1)
    slider_yc = Slider(ax_yc, "Center Y", 0, h, valinit=init_yc, valstep=1)
    slider_r = Slider(ax_r, "Radius R", 1, min(w, h), valinit=init_r, valstep=1)

    def update(val):
        f_idx = int(slider_frame.val)
        v_val = slider_vmax.val
        xc = int(slider_xc.val)
        yc = int(slider_yc.val)
        r = int(slider_r.val)

        im.set_data(stack[f_idx])
        im.set_clim(0, v_val)
        ax.set_title(f"{title} - Frame {f_idx + 1}/{n_frames}", fontsize=11, fontweight="bold")

        circle_patch.set_center((xc, yc))
        circle_patch.set_radius(r)

        fig.canvas.draw_idle()

    for s in [slider_frame, slider_vmax, slider_xc, slider_yc, slider_r]:
        s.on_changed(update)

    # Export Button
    ax_btn = plt.axes([0.35, 0.02, 0.30, 0.04])
    btn_export = Button(ax_btn, "📋 Export Config Code", color="#2ecc71", hovercolor="#27ae60")

    def export_code(event):
        xc = int(slider_xc.val)
        yc = int(slider_yc.val)
        r = int(slider_r.val)

        print("\n" + "=" * 55)
        print("   COPY-PASTE CONFIGURATION (ERUPTION MASK)")
        print("=" * 55)
        print(f"ERUPTION_CENTER = ({xc}, {yc})  # (xC, yC)")
        print(f"ERUPTION_RADIUS = {r}  # R")
        print("=" * 55 + "\n")

    btn_export.on_clicked(export_code)

    fig._ui_widgets = [slider_frame, slider_vmax, slider_xc, slider_yc, slider_r, btn_export]

    plt.show()

    # Extract final values after window closure
    final_xc = int(slider_xc.val)
    final_yc = int(slider_yc.val)
    final_r = int(slider_r.val)

    return (final_xc, final_yc), final_r


import os
from typing import Optional
import numpy as np


def load_cached_array(
        cache_dir: str,
        filename: str = "normalized_erupting_pixels.npy",
        logger=None
) -> Optional[np.ndarray]:
    """
    Loads a cached NumPy array (.npy) if it exists in the specified directory.

    Parameters
    ----------
    cache_dir : str
        Directory containing the cache file.
    filename : str
        Name of the .npy file (default: "normalized_erupting_pixels.npy").
    logger : optional
        Logger instance for logging output.

    Returns
    -------
    np.ndarray or None
        The loaded array if found, otherwise None.
    """
    cache_filepath = os.path.join(cache_dir, filename)

    if os.path.exists(cache_filepath):
        data = np.load(cache_filepath)
        msg = f"Loaded cached array from {cache_filepath} (Shape: {data.shape})"
        if logger:
            logger.info(msg)
        else:
            print(msg)
        return data
    else:
        msg = f"Cache file not found at {cache_filepath}"
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return None
