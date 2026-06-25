import numpy as np
import cv2
from astropy.io import fits
from typing import Any, List, Union
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift, binary_dilation, median_filter
import logging
from colorlog import ColoredFormatter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, Button


def calibrate(fits_data, flat_data, dark_data, hdf_data):
    print("  Applying calibration")
    return {"calibrated": True}

def fits_to_cv(data: Any) -> np.ndarray:
    """
    Convert FITS-like data objects into OpenCV-compatible NumPy arrays.

    This function converts numerical data originating from FITS files
    (or FITS-like containers) into uint8 OpenCV-compatible images.

    The output is ALWAYS a NumPy array:
    - Single input  -> shape (H, W, 3)
    - Multiple input -> shape (N, H, W, 3)

    Supported inputs:
    - astropy.io.fits.HDUList
    - Individual FITS HDU objects (with `.data`)
    - NumPy arrays
    - Lists or tuples of the above

    Conversion steps:
    1. Extract numerical data
    2. Replace NaNs / infs
    3. Min–max normalize to [0, 255]
    4. Convert to uint8
    5. Ensure 3-channel BGR format

    Parameters
    ----------
    data : Any
        FITS HDU, HDUList, NumPy array, or list/tuple of these.

    Returns
    -------
    np.ndarray
        OpenCV-compatible image array:
        - (H, W, 3) or
        - (N, H, W, 3)

    Raises
    ------
    NotImplementedError
        If input type is unsupported.
    ValueError
        If no numerical data is found or stacking fails.
    """

    def _hdu_to_array(hdu: Any) -> np.ndarray:
        """Convert a single FITS-like object into a BGR uint8 image."""

        # Extract numerical array
        if isinstance(hdu, fits.HDUList):
            arr = hdu[0].data
        elif hasattr(hdu, "data"):
            arr = hdu.data
        elif isinstance(hdu, np.ndarray):
            arr = hdu
        else:
            raise NotImplementedError(
                f"Unsupported input type: {type(hdu)}"
            )

        if arr is None:
            raise ValueError("No data found in FITS object")

        # Sanitize and normalize
        arr = np.nan_to_num(arr).astype(np.float32)
        mn, mx = arr.min(), arr.max()

        if mx == mn:
            img = np.zeros(arr.shape, dtype=np.uint8)
        else:
            img = ((arr - mn) / (mx - mn) * 255.0).astype(np.uint8)

        # Ensure OpenCV-compatible 3-channel format
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return img

    # Normalize input to iterable
    if isinstance(data, (list, tuple)):
        imgs = [_hdu_to_array(x) for x in data]
        try:
            return np.stack(imgs, axis=0)
        except ValueError as exc:
            raise ValueError(
                "All images must have identical shape to stack"
            ) from exc
    else:
        return _hdu_to_array(data)

def crop_images(imgs: np.ndarray, xmin: int, ymin: int, xmax: int, ymax: int, convert=None, logger=None) -> np.ndarray:
    """
    Crop a batch of images to the specified bounding box.

    Parameters
    ----------
    imgs : np.ndarray
        Input image array of shape (N, H, W, C) or (H, W, C).
    xmin : int
        Minimum x-coordinate of the crop box.
    ymin : int
        Minimum y-coordinate of the crop box.
    xmax : int
        Maximum x-coordinate of the crop box.
    ymax : int
        Maximum y-coordinate of the crop box.
    convert : optional
        If not None, convert cropped images to this dtype.
    logger : optional
        Logger with .debug() method.

    Returns
    -------
    np.ndarray
        Cropped (and optionally converted) image array.
    """
    if logger is not None:
        logger.info(
            "Cropping images to box (%d, %d, %d, %d), convert=%s",
            xmin,
            ymin,
            xmax,
            ymax,
            str(convert),
        )
    if imgs.ndim == 2:
        # single grayscale image (W, H) → (1, W, H)
        cropped = imgs[ymin:ymax, xmin:xmax]
        if convert is not None:
            cropped = cropped.astype(convert)
        if logger is not None:
            logger.info(f"Cropped image: {cropped.shape}")
        return cropped

    elif imgs.ndim != 3:
        raise ValueError(f"Expected shape (N, W, H) or (W, H), got {imgs.shape}")
    cropped = imgs[:, ymin:ymax, xmin:xmax]
    if convert is not None:
        cropped = cropped.astype(convert)

    if logger is not None:
        logger.info(f"Shape of cropped images array: {cropped.shape}")
    return cropped

# def correct_dark(imgs: np.ndarray, dark: np.ndarray, logger=None) -> np.ndarray:
#     """
#     Subtract dark frame from images.
#
#     Parameters
#     ----------
#     imgs : np.ndarray
#         Input image array of shape (N, H, W, C) or (H, W, C).
#     dark : np.ndarray
#         Dark frame to subtract.
#     logger : optional
#         Logger with .debug() method.
#
#     Returns
#     -------
#     np.ndarray
#         Dark-corrected image array.
#     """
#     if logger is not None:
#         logger.info("Correcting images with dark frame")
#     corrected = imgs - dark
#     corrected = np.clip(corrected, 0, None)  # Ensure no negative values
#     if logger is not None:
#         logger.info(f"Shape of dark corrected images array: {corrected.shape}")
#     return corrected


def correct_dark(
        imgs: np.ndarray,
        dark: np.ndarray,
        align_dark: bool = False,
        num_peaks: int = 20,
        logger=None
) -> np.ndarray:
    """
    Subtract dark frame from images, with automatic coordinate-based
    hot pixel alignment to ensure a perfect match before subtraction.
    """
    working_dark = dark.copy()

    if align_dark:
        if logger is not None:
            logger.info("Automatically aligning dark frame hot pixels...")

        # 1. Isolate spatial 2D templates for peak detection
        img_ref = imgs[0] if imgs.ndim > 2 else imgs
        if img_ref.ndim == 3:
            img_ref = np.mean(img_ref, axis=-1)
        dark_ref = dark if dark.ndim == 2 else np.mean(dark, axis=-1)

        # 2. Find coordinates of the brightest 'num_peaks' pixels in both frames
        # Flat indices of top peaks
        img_flat_idx = np.argsort(img_ref.ravel())[-num_peaks:]
        dark_flat_idx = np.argsort(dark_ref.ravel())[-num_peaks:]

        # Convert to (Y, X) coordinate pairs
        img_coords = np.array(np.unravel_index(img_flat_idx, img_ref.shape)).T
        dark_coords = np.array(np.unravel_index(dark_flat_idx, dark_ref.shape)).T

        # 3. Calculate the shift by finding the average matching displacement
        # We calculate the median difference between the coordinates
        # Sorting them by magnitude helps match the relative structural pairs
        img_coords_sorted = img_coords[np.lexsort((img_coords[:, 1], img_coords[:, 0]))]
        dark_coords_sorted = dark_coords[np.lexsort((dark_coords[:, 1], dark_coords[:, 0]))]

        shifts = img_coords_sorted - dark_coords_sorted

        # Take the median shift vector to ignore outliers (like cosmic rays or stars)
        computed_shift = np.median(shifts, axis=0)

        if logger is not None:
            logger.info(f"Automatically aligned dark frame shift (Y, X): {computed_shift}")

        # 4. Apply the spatial shift to the entire dark frame
        if dark.ndim == 3:
            spatial_shift = (computed_shift[0], computed_shift[1], 0)
            working_dark = shift(dark, spatial_shift, mode='nearest')
        else:
            working_dark = shift(dark, computed_shift, mode='nearest')

    # 5. Subtract the aligned dark frame
    if logger is not None:
        logger.info("Subtracting aligned dark frame")

    corrected = imgs - working_dark
    corrected = np.clip(corrected, 0, None)

    if logger is not None:
        logger.info(f"Shape of dark corrected images array: {corrected.shape}")

    return corrected

def correct_flat(
        imgs: np.ndarray,
        flat: np.ndarray,
        align_flat: bool = False,
        max_allowable_shift: float = 50.0,  # Max pixels the flat is allowed to move
        logger=None
) -> np.ndarray:
    """
    Divide images by flat field to correct for illumination variations,
    with an optional phase cross-correlation alignment toggle.

    Parameters
    ----------
    imgs : np.ndarray
        Input image array of shape (N, H, W, C), (H, W, C), or (H, W).
    flat : np.ndarray
        Flat field to divide by. Match spatial dimensions (H, W) of imgs.
    align_flat : bool, default False
        If True, automatically registers and shifts the flat to match the
        spatial features of the input images using phase cross-correlation.
    max_allowable_shift : float, default 50.0
        Maximum pixels the flat is allowed to move during alignment.
    Divide images by flat field to correct for illumination variations safely,
    preventing divide-by-zero errors and protecting against wild alignment shifts.
    logger : optional
        Logger with .info() or .debug() methods.
    """
    working_flat = flat.copy()

    if align_flat:
        if logger is not None:
            logger.info("Aligning flat field via phase cross-correlation...")

        # Extract 2D layer for structural tracking
        img_ref = imgs[0] if imgs.ndim > 2 else imgs
        if img_ref.ndim == 3:
            img_ref = np.mean(img_ref, axis=-1)

        flat_ref = flat if flat.ndim == 2 else np.mean(flat, axis=-1)

        # 1. Compute structural offset
        shift_vector, error, diffphase = phase_cross_correlation(
            img_ref,
            flat_ref,
            upsample_factor=10
        )

        # Safety Check: Did the algorithm miscalculate a massive shift?
        absolute_shift = np.linalg.norm(shift_vector)
        if absolute_shift > max_allowable_shift:
            if logger is not None:
                logger.warning(
                    f"Ignored extreme shift vector {shift_vector} (Magnitude: {absolute_shift:.1f}px). "
                    f"Exceeds max limits of {max_allowable_shift}px. Using unshifted flat field instead."
                )
            full_shift = np.zeros_like(shift_vector)
        else:
            if logger is not None:
                logger.info(f"Calculated stable flat alignment shift (Y, X): {shift_vector}")
            full_shift = shift_vector

        # 2. Shift the flat array over spatial boundaries
        # mode='nearest' copies edge pixels instead of placing 0.0 or mirroring artifacts
        if flat.ndim == 3:
            spatial_shift = (full_shift[0], full_shift[1], 0)
            working_flat = shift(flat, spatial_shift, mode='nearest')
        else:
            working_flat = shift(flat, full_shift, mode='nearest')

    # 3. Absolute Zero-Division Prevention Guardrail
    # Force any values close to 0 to a tiny positive float value
    working_flat = np.where(working_flat <= 0, 1e-6, working_flat)

    # 4. Perform division safely
    if logger is not None:
        logger.info("Dividing images with processed flat field")

    corrected = imgs / working_flat
    corrected = np.clip(corrected, 0, None)

    if logger is not None:
        logger.info(f"Shape of flat corrected images array: {corrected.shape}")

    return corrected


def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()

    formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s | %(levelname)s | %(message)s",
        log_colors={
            "DEBUG": "green",     # inside functions
            "INFO": "blue",       # pipeline-level info
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


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