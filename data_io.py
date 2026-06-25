import os
from astropy.io import fits
import numpy as np
from pathlib import Path
from typing import List, Optional
from FICUS.PYTHON.OCAS_lib import Light, Calibration, Measurement
from FICUS.PYTHON.NormalizationModule import Normalization, Linearity
from typing import Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, Button
from plotting import *
from processing import *
from analysis import *

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




def load_hdf_light(hdf_dir: str, light_idx: int, logger=None) -> tuple[np.ndarray, np.ndarray]:
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
    msg = f"Reading HDF5 as C and D light from {hdf_dir}"
    if logger is not None:
        logger.info(msg)


    # Get [C, D] file paths as strings
    c_path, d_path = get_hdf_paths(
        hdf_dir,
        CD=True,
        return_path=False
    )

    if c_path is None or d_path is None:
        raise FileNotFoundError(
            f"Could not find both C and D HDF files in {hdf_dir}"
        )
    if type(light_idx) is not int:
        raise TypeError(f"Invalid type(light_idx) = {type(light_idx)} != int")

    # Load HDF files using Light
    mC = Light(c_path, light_idx)
    mD = Light(d_path, light_idx)

    return mC.data, mD.data


def load_WL_spectrum():
    wlc = np.load("./FICUS/useful_files/WL_range_C.npy")
    wld = np.load("./FICUS/useful_files/WL_range_D.npy")
    return wlc, wld

def load_flats(flat_dir, logger=None):
    print(f"  Reading flats from {flat_dir}")
    return {"dummy_flats": None}

def load_darks(dark_dir, logger=None):
    print(f"  Reading darks from {dark_dir}")
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