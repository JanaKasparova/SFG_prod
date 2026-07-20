import gc
from data_io import *
from processing import *
from analysis import *
from plotting import *
from FICUS.PYTHON.OCAS_lib import *
from FICUS.PYTHON.NormalizationModule import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
from pipeline import Pipeline


def save_data_flat(flat_folder, idx, save_folder, logger=None):
    # flat image calculation
    folder_flat = flat_folder
    flat_raw = load_fits(folder_flat, logger=logger)

    # first we find which images are usable for flat calculation
    plot_average_intensity(flat_raw,
                           logger=logger,
                           title="Intensity of all flat images",
                           save_name="intensity_of_all_flat_images",
                           plot_graph=False)

    print(flat_raw.shape)

    usable_flat = flat_raw[idx[0]:idx[1], :, :]
    print(usable_flat.shape)
    plot_average_intensity(usable_flat,
                           logger=logger,
                           title="Intensity of usable flat images",
                           save_name="intensity_of_usable_flat_images",
                           plot_graph=False)

    # averaging for master flat
    average_flat = average_numpy_array(usable_flat, axis=0, logger=logger)
    plot_single_image(average_flat,
                      logger=logger,
                      title="Picture of master flat",
                      save_name="picture_of_master_flat",
                      plot_graph=False)

    # save the master flat to the specified folder
    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, "master_flat.fits")
        fits.writeto(save_path, average_flat, overwrite=True)
        if logger is not None:
            logger.info(f"Master flat saved to {save_path}")


def testing_flat(flat_folder, idx=None, logger=None):
    # flat image calculation
    folder_flat = flat_folder
    flat_raw = load_fits(folder_flat, logger=logger)

    # first we find which images are usable for flat calculation
    plot_average_intensity(flat_raw, logger=logger, title="sun_area")
    print(flat_raw.shape)

    if idx is not None:
        usable_flat = flat_raw[idx[0]:idx[1], :, :]
        print(usable_flat.shape)
        plot_average_intensity(usable_flat, logger=logger, title="sun_area")

        # averaging for master flat
        average_flat = average_numpy_array(usable_flat, axis=0, logger=logger)
        plot_single_image(average_flat, logger=logger, title="master flat")


def save_dark(dark_folder, save_folder=None, logger=None):
    fits_dark = load_fits(dark_folder, logger=logger)
    average_dark = average_numpy_array(fits_dark, axis=0, logger=logger)
    plot_single_image(average_dark,
                      logger=logger,
                      title="master dark",
                      save_name="picture_of_master_dark",
                      plot_graph=False,
                      vmax=1000)

    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
        save_fits_image(average_dark, output_dir=save_folder, filename="master_dark.fits", logger=logger)


if __name__ == "__main__":
    logger = setup_logger()
    flat_folder = "./sun_area_flat/SlitJaw"
    idx = (0, 80)
    save_folder = "./MASTER_SAVE"

    # first run testing to finetune our flat data
    testing_flat(flat_folder, idx)

    # then we run save_data to save our chosen data
    # save_data_flat(flat_folder, idx, save_folder)

    # make dark

    dark_folder = "./2024-07-29/sun_area_dark/SlitJaw"
    save_dark(dark_folder, save_folder, logger=logger)
