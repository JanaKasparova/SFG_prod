# import gc
# import os
# from data_io import *
# from processing import *
# from analysis import *
# from plotting import *
# from FICUS.PYTHON.OCAS_lib import *
# from FICUS.PYTHON.NormalizationModule import *
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Circle
# from matplotlib.patches import Rectangle
# from pipeline import Pipeline
#
#
# def save_data_flat(flat_folder, idx, save_folder, logger=None):
#     # flat image calculation
#     folder_flat = flat_folder
#     flat_raw = load_fits(folder_flat, logger=logger)
#
#     # first we find which images are usable for flat calculation
#     plot_average_intensity(flat_raw,
#                            logger=logger,
#                            title="Intensity of all flat images",
#                            save_name="intensity_of_all_flat_images",
#                            plot_graph=False)
#
#     print(flat_raw.shape)
#
#     usable_flat = flat_raw[idx[0]:idx[1], :, :]
#     print(usable_flat.shape)
#     plot_average_intensity(usable_flat,
#                            logger=logger,
#                            title="Intensity of usable flat images",
#                            save_name="intensity_of_usable_flat_images",
#                            plot_graph=False)
#
#     # averaging for master flat
#     average_flat = average_numpy_array(usable_flat, axis=0, logger=logger)
#     plot_single_image(average_flat,
#                       logger=logger,
#                       title="Picture of master flat",
#                       save_name="picture_of_master_flat",
#                       plot_graph=False)
#
#     # save the master flat to the specified folder
#     if save_folder is not None:
#         os.makedirs(save_folder, exist_ok=True)
#         save_path = os.path.join(save_folder, "master_flat.fits")
#         fits.writeto(save_path, average_flat, overwrite=True)
#         if logger is not None:
#             logger.info(f"Master flat saved to {save_path}")
#
#
# def testing_flat(flat_folder, idx=None, logger=None):
#     # flat image calculation
#     folder_flat = flat_folder
#     flat_raw = load_fits(folder_flat, logger=logger)
#
#     # first we find which images are usable for flat calculation
#     plot_average_intensity(flat_raw, logger=logger, title="sun_area")
#     print(flat_raw.shape)
#
#     if idx is not None:
#         usable_flat = flat_raw[idx[0]:idx[1], :, :]
#         print(usable_flat.shape)
#         plot_average_intensity(usable_flat, logger=logger, title="sun_area")
#
#         # averaging for master flat
#         average_flat = average_numpy_array(usable_flat, axis=0, logger=logger)
#         plot_single_image(average_flat, logger=logger, title="master flat")
#
#
# def save_dark(dark_folder, save_folder=None, logger=None):
#     fits_dark = load_fits(dark_folder, logger=logger)
#     average_dark = average_numpy_array(fits_dark, axis=0, logger=logger)
#     plot_single_image(average_dark,
#                       logger=logger,
#                       title="master dark",
#                       save_name="picture_of_master_dark",
#                       plot_graph=False,
#                       vmax=1000)
#
#     if save_folder is not None:
#         os.makedirs(save_folder, exist_ok=True)
#         save_fits_image(average_dark, output_dir=save_folder, filename="master_dark.fits", logger=logger)
#
#
# if __name__ == "__main__":
#     logger = setup_logger()
#     flat_folder = "./sun_area_flat/SlitJaw"
#     idx = (0, 80)
#     parent_save_folder = "./MASTER_SAVE"
#     astropy_time_array = compile_directory_timestamps(flat_folder)
#     date_suffix = astropy_time_array[0].datetime.strftime("%Y-%m-%d") if len(astropy_time_array) > 0 else "0000-00-00"
#     save_folder = os.path.join(parent_save_folder, date_suffix)
#
#
#     # first run testing to finetune our flat data
#     # testing_flat(flat_folder, idx)
#
#     # then we run save_data to save our chosen data
#     save_data_flat(flat_folder, idx, save_folder)
#
#     # make dark
#
#     dark_folder = "./2024-07-29/sun_area_dark/SlitJaw"
#     save_dark(dark_folder, save_folder, logger=logger)

#
# import gc
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Circle, Rectangle
#
# from data_io import *
# from processing import *
# from analysis import *
# from plotting import *
# from FICUS.PYTHON.OCAS_lib import *
# from FICUS.PYTHON.NormalizationModule import *
# from pipeline import Pipeline
#
#
# def save_data_flat(flat_folder, idx, save_folder, logger=None, show_plots=False):
#     # flat image calculation
#     folder_flat = flat_folder
#     flat_raw = load_fits(folder_flat, logger=logger)
#
#     # Target plot paths saved directly inside save_folder
#     save_all_flat = os.path.join(save_folder, "intensity_of_all_flat_images") if save_folder else "intensity_of_all_flat_images"
#     save_usable_flat = os.path.join(save_folder, "intensity_of_usable_flat_images") if save_folder else "intensity_of_usable_flat_images"
#     save_master_flat = os.path.join(save_folder, "picture_of_master_flat") if save_folder else "picture_of_master_flat"
#
#     # first we find which images are usable for flat calculation
#     plot_average_intensity(flat_raw,
#                            logger=logger,
#                            title="Intensity of all flat images",
#                            save_name=save_all_flat,
#                            plot_graph=show_plots)
#
#     print(f"Raw flat shape: {flat_raw.shape}")
#
#     usable_flat = flat_raw[idx[0]:idx[1], :, :]
#     print(f"Usable flat shape: {usable_flat.shape}")
#     plot_average_intensity(usable_flat,
#                            logger=logger,
#                            title="Intensity of usable flat images",
#                            save_name=save_usable_flat,
#                            plot_graph=show_plots)
#
#     # averaging for master flat
#     average_flat = average_numpy_array(usable_flat, axis=0, logger=logger)
#     plot_single_image(average_flat,
#                       logger=logger,
#                       title="Picture of master flat",
#                       save_name=save_master_flat,
#                       plot_graph=show_plots)
#
#     # save the master flat to the specified folder
#     if save_folder is not None:
#         os.makedirs(save_folder, exist_ok=True)
#         save_path = os.path.join(save_folder, "master_flat.fits")
#         fits.writeto(save_path, average_flat, overwrite=True)
#         if logger is not None:
#             logger.info(f"Master flat saved to {save_path}")
#
#
# def testing_flat(flat_folder, idx=None, logger=None, show_plots=True):
#     # flat image calculation
#     folder_flat = flat_folder
#     flat_raw = load_fits(folder_flat, logger=logger)
#
#     # first we find which images are usable for flat calculation
#     plot_average_intensity(flat_raw, logger=logger, title="sun_area", plot_graph=show_plots)
#     print(f"Raw flat shape: {flat_raw.shape}")
#
#     if idx is not None:
#         usable_flat = flat_raw[idx[0]:idx[1], :, :]
#         print(f"Usable flat shape: {usable_flat.shape}")
#         plot_average_intensity(usable_flat, logger=logger, title="sun_area usable", plot_graph=show_plots)
#
#         # averaging for master flat
#         average_flat = average_numpy_array(usable_flat, axis=0, logger=logger)
#         plot_single_image(average_flat, logger=logger, title="master flat preview", plot_graph=show_plots)
#
#
# def save_dark(dark_folder, save_folder=None, logger=None, vmax=1000, show_plots=False):
#     fits_dark = load_fits(dark_folder, logger=logger)
#     average_dark = average_numpy_array(fits_dark, axis=0, logger=logger)
#
#     save_master_dark = os.path.join(save_folder, "picture_of_master_dark") if save_folder else "picture_of_master_dark"
#
#     plot_single_image(average_dark,
#                       logger=logger,
#                       title="master dark",
#                       save_name=save_master_dark,
#                       plot_graph=show_plots,
#                       vmax=vmax)
#
#     if save_folder is not None:
#         os.makedirs(save_folder, exist_ok=True)
#         save_fits_image(average_dark, output_dir=save_folder, filename="master_dark.fits", logger=logger)
#
#
# if __name__ == "__main__":
#     # =====================================================================
#     # CONFIGURATION PARAMETERS (All modifications go here!)
#     # =====================================================================
#     # --- Input Folders ---
#     FLAT_FOLDER = "./sun_area_flat/SlitJaw"
#     DARK_FOLDER = "./2024-07-29/sun_area_dark/SlitJaw"
#
#     # --- Output Folder ---
#     PARENT_SAVE_FOLDER = "./MASTER_SAVE"
#
#     # --- Slicing & Calibration Settings ---
#     FLAT_IDX = (0, 80)          # Frame index slice (start, end) for usable flats
#     DARK_VMAX = 1000            # Intensity cap for dark frame preview plot
#
#     # --- Execution Toggles ---
#     RUN_FLAT_TESTING = True    # Set to True to test/preview flat index selection
#     PROCESS_MASTER_FLAT = False  # Set to True to calculate & save master_flat.fits
#     PROCESS_MASTER_DARK = True  # Set to True to calculate & save master_dark.fits
#     SHOW_PLOTS = True          # Set to True to display interactive plot windows
#     # =====================================================================
#
#     # Initialize logger ecosystem
#     logger = setup_logger()
#
#     # Extract date suffix automatically from folder timestamps
#     ref_folder = FLAT_FOLDER if os.path.exists(FLAT_FOLDER) else DARK_FOLDER
#     astropy_time_array = compile_directory_timestamps(ref_folder)
#     date_suffix = astropy_time_array[0].datetime.strftime("%Y-%m-%d") if len(astropy_time_array) > 0 else "0000-00-00"
#
#     save_folder = os.path.join(PARENT_SAVE_FOLDER, date_suffix)
#     os.makedirs(save_folder, exist_ok=True)
#
#     # 1. OPTIONAL: Flat Range Testing Mode
#     if RUN_FLAT_TESTING:
#         logger.info("--- Running Flat Field Slicing Test ---")
#         testing_flat(flat_folder=FLAT_FOLDER, idx=FLAT_IDX, logger=logger, show_plots=True)
#
#     # 2. Compute and Save Master Flat
#     if PROCESS_MASTER_FLAT:
#         logger.info("--- Processing Master Flat ---")
#         save_data_flat(flat_folder=FLAT_FOLDER, idx=FLAT_IDX, save_folder=save_folder, logger=logger, show_plots=SHOW_PLOTS)
#
#     # 3. Compute and Save Master Dark
#     if PROCESS_MASTER_DARK:
#         logger.info("--- Processing Master Dark ---")
#         save_dark(dark_folder=DARK_FOLDER, save_folder=save_folder, logger=logger, vmax=DARK_VMAX, show_plots=SHOW_PLOTS)

import gc
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from data_io import *
from processing import *
from analysis import *
from plotting import *
from FICUS.PYTHON.OCAS_lib import *
from FICUS.PYTHON.NormalizationModule import *
from pipeline import Pipeline


def save_data_flat(flat_folder, idx, save_folder, master_dark=None, logger=None, show_plots=False):
    # flat image calculation
    folder_flat = flat_folder
    flat_raw = load_fits(folder_flat, logger=logger)

    # Apply Master Dark correction if provided
    if master_dark is not None:
        if logger:
            logger.info("🌒 Applying Master Dark correction to flat frames...")
        else:
            print("🌒 Applying Master Dark correction to flat frames...")
        flat_raw = flat_raw - master_dark

    # Target plot paths saved directly inside save_folder
    save_all_flat = os.path.join(save_folder, "intensity_of_all_flat_images") if save_folder else "intensity_of_all_flat_images"
    save_usable_flat = os.path.join(save_folder, "intensity_of_usable_flat_images") if save_folder else "intensity_of_usable_flat_images"
    save_master_flat = os.path.join(save_folder, "picture_of_master_flat") if save_folder else "picture_of_master_flat"

    title_suffix = " (Dark-Corrected)" if master_dark is not None else ""

    # first we find which images are usable for flat calculation
    plot_average_intensity(flat_raw,
                           logger=logger,
                           title=f"Intensity of all flat images{title_suffix}",
                           save_name=save_all_flat,
                           plot_graph=show_plots)

    print(f"Raw flat shape: {flat_raw.shape}")

    usable_flat = flat_raw[idx[0]:idx[1], :, :]
    print(f"Usable flat shape: {usable_flat.shape}")
    plot_average_intensity(usable_flat,
                           logger=logger,
                           title=f"Intensity of usable flat images{title_suffix}",
                           save_name=save_usable_flat,
                           plot_graph=show_plots)

    # averaging for master flat
    average_flat = average_numpy_array(usable_flat, axis=0, logger=logger)
    plot_single_image(average_flat,
                      logger=logger,
                      title=f"Picture of master flat{title_suffix}",
                      save_name=save_master_flat,
                      plot_graph=show_plots)

    # save the master flat to the specified folder
    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, "master_flat.fits")
        fits.writeto(save_path, average_flat, overwrite=True)
        if logger is not None:
            logger.info(f"Master flat saved to {save_path}")

    return average_flat


def testing_flat(flat_folder, idx=None, master_dark=None, logger=None, show_plots=True):
    # flat image calculation
    folder_flat = flat_folder
    flat_raw = load_fits(folder_flat, logger=logger)

    if master_dark is not None:
        flat_raw = flat_raw - master_dark

    title_suffix = " (Dark-Corrected)" if master_dark is not None else ""

    # first we find which images are usable for flat calculation
    plot_average_intensity(flat_raw, logger=logger, title=f"sun_area{title_suffix}", plot_graph=show_plots)
    print(f"Raw flat shape: {flat_raw.shape}")

    if idx is not None:
        usable_flat = flat_raw[idx[0]:idx[1], :, :]
        print(f"Usable flat shape: {usable_flat.shape}")
        plot_average_intensity(usable_flat, logger=logger, title=f"sun_area usable{title_suffix}", plot_graph=show_plots)

        # averaging for master flat
        average_flat = average_numpy_array(usable_flat, axis=0, logger=logger)
        plot_single_image(average_flat, logger=logger, title=f"master flat preview{title_suffix}", plot_graph=show_plots)


def save_dark(dark_folder, save_folder=None, logger=None, vmax=1000, show_plots=False):
    fits_dark = load_fits(dark_folder, logger=logger)
    average_dark = average_numpy_array(fits_dark, axis=0, logger=logger)

    save_master_dark = os.path.join(save_folder, "picture_of_master_dark") if save_folder else "picture_of_master_dark"

    plot_single_image(average_dark,
                      logger=logger,
                      title="master dark",
                      save_name=save_master_dark,
                      plot_graph=show_plots,
                      vmax=vmax)

    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
        save_fits_image(average_dark, output_dir=save_folder, filename="master_dark.fits", logger=logger)

    return average_dark


if __name__ == "__main__":
    # =====================================================================
    # CONFIGURATION PARAMETERS (All modifications go here!)
    # =====================================================================
    # --- Input Folders ---
    FLAT_FOLDER = "./sun_area_flat/SlitJaw"
    DARK_FOLDER = "./2024-07-29/sun_area_dark/SlitJaw"

    # --- Output Folder ---
    PARENT_SAVE_FOLDER = "./MASTER_SAVE"

    # --- Slicing & Calibration Settings ---
    FLAT_IDX = (0, 80)          # Frame index slice (start, end) for usable flats
    DARK_VMAX = 1000            # Intensity cap for dark frame preview plot

    # --- Execution Toggles ---
    RUN_FLAT_TESTING = True     # Set to True to test/preview flat index selection
    PROCESS_MASTER_DARK = True  # Set to True to calculate & save master_dark.fits
    PROCESS_MASTER_FLAT = True # Set to True to calculate & save master_flat.fits
    SHOW_PLOTS = True           # Set to True to display interactive plot windows
    # =====================================================================

    # Initialize logger ecosystem
    logger = setup_logger()

    # Extract date suffix automatically from folder timestamps
    ref_folder = FLAT_FOLDER if os.path.exists(FLAT_FOLDER) else DARK_FOLDER
    astropy_time_array = compile_directory_timestamps(ref_folder)
    date_suffix = astropy_time_array[0].datetime.strftime("%Y-%m-%d") if len(astropy_time_array) > 0 else "0000-00-00"

    save_folder = os.path.join(PARENT_SAVE_FOLDER, date_suffix)
    os.makedirs(save_folder, exist_ok=True)

    master_dark = None

    # 1. Compute and Save Master Dark FIRST (Required for Dark-Correcting Flat)
    if PROCESS_MASTER_DARK:
        logger.info("--- Processing Master Dark ---")
        master_dark = save_dark(dark_folder=DARK_FOLDER, save_folder=save_folder, logger=logger, vmax=DARK_VMAX, show_plots=SHOW_PLOTS)
    else:
        # Fallback: check if master_dark.fits already exists on disk
        existing_dark_path = os.path.join(save_folder, "master_dark.fits")
        if os.path.exists(existing_dark_path):
            logger.info(f"--- Loading existing Master Dark from: {existing_dark_path} ---")
            master_dark = load_fits_img(existing_dark_path)

    # 2. OPTIONAL: Flat Range Testing Mode (includes dark correction if available)
    if RUN_FLAT_TESTING:
        logger.info("--- Running Flat Field Slicing Test ---")
        testing_flat(flat_folder=FLAT_FOLDER, idx=FLAT_IDX, master_dark=master_dark, logger=logger, show_plots=True)

    # 3. Compute and Save Dark-Corrected Master Flat
    if PROCESS_MASTER_FLAT:
        logger.info("--- Processing Dark-Corrected Master Flat ---")
        save_data_flat(flat_folder=FLAT_FOLDER, idx=FLAT_IDX, save_folder=save_folder, master_dark=master_dark, logger=logger, show_plots=SHOW_PLOTS)