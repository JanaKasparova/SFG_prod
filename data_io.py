import os
from astropy.io import fits
import numpy as np

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


def load_hdf(hdf_dir, logger=None):
    print(f"  Reading HDF5 from {hdf_dir}")
    return {"dummy_hdf": None}

def load_flats(flat_dir, logger=None):
    print(f"  Reading flats from {flat_dir}")
    return {"dummy_flats": None}

def load_darks(dark_dir, logger=None):
    print(f"  Reading darks from {dark_dir}")
    return {"dummy_darks": None}