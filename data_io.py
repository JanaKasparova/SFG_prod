import os
from astropy.io import fits
import numpy as np
from pathlib import Path
from typing import List, Optional
from FICUS.PYTHON.OCAS_lib import Light, Calibration, Measurement
from FICUS.PYTHON.NormalizationModule import Normalization, Linearity
from typing import Tuple, Optional


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
    if logger is not None:
        logger.debug(f"Loaded {len(images)} FITS files from {folder_path}")
    return np.array(images)

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

def load_flats(flat_dir, logger=None):
    print(f"  Reading flats from {flat_dir}")
    return {"dummy_flats": None}

def load_darks(dark_dir, logger=None):
    print(f"  Reading darks from {dark_dir}")
    return {"dummy_darks": None}