import numpy as np
import cv2
from astropy.io import fits
from typing import Any, List, Union


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

