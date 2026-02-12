from data_io import load_hdf_light
import numpy as np
from typing import Any


def average_numpy_array(
    arr: np.ndarray,
    out_dtype: np.dtype = np.int16,
    axis: int = 0,
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
    if logger is not None:
        logger.debug(f"Averageing a numpy array, axis={axis}, dtype={out_dtype}")
    temp = arr.astype(np.float64, copy=False)
    avg = np.average(temp, axis=axis)
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


def run_analysis(calibrated_data):
    print("  Running analysis")
    return {"result": 42}
