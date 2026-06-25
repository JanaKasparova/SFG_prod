import gc
from data_io import *
from processing import *
from analysis import *
from plotting import *

if __name__ == "__main__":
    logger = setup_logger()

    master_flat = load_fits_img("MASTER_SAVE/master_flat.fits")