# import gc
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Circle, Rectangle
# from astropy.io import fits
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
#
# def save_data_flat(flat_folder, idx, save_folder, master_dark=None, logger=None, show_plots=False, batch_size=50):
#     flat_files = get_fits_filepaths(flat_folder)
#     if not flat_files:
#         raise FileNotFoundError(f"No FITS files found in {flat_folder}")
#
#     # Target plot paths saved directly inside save_folder
#     save_all_flat = os.path.join(save_folder,
#                                  "intensity_of_all_flat_images") if save_folder else "intensity_of_all_flat_images"
#     save_usable_flat = os.path.join(save_folder,
#                                     "intensity_of_usable_flat_images") if save_folder else "intensity_of_usable_flat_images"
#     save_master_flat = os.path.join(save_folder, "picture_of_master_flat") if save_folder else "picture_of_master_flat"
#
#     title_suffix = " (Dark-Corrected)" if master_dark is not None else ""
#
#     if logger:
#         logger.info(f"--- Calculating intensity profile for all {len(flat_files)} flat images ---")
#
#     # Process all flat images in batches to compute per-frame intensity profile
#     _, all_intensities = process_fits_in_batches(flat_files, master_dark=master_dark, batch_size=batch_size,
#                                                  logger=logger)
#
#     # Pass lightweight (N, 1, 1) dummy shape to plotting function to avoid RAM spikes
#     plot_average_intensity(all_intensities[:, None, None],
#                            logger=logger,
#                            title=f"Intensity of all flat images{title_suffix}",
#                            save_name=save_all_flat,
#                            plot_graph=show_plots)
#
#     print(f"Raw flat count: {len(flat_files)}")
#
#     # Slice usable files
#     usable_files = flat_files[idx[0]:idx[1]]
#     print(f"Usable flat count: {len(usable_files)}")
#
#     if logger:
#         logger.info(f"--- Computing Master Flat from {len(usable_files)} usable frames ---")
#
#     average_flat, usable_intensities = process_fits_in_batches(usable_files, master_dark=master_dark,
#                                                                batch_size=batch_size, logger=logger)
#
#     plot_average_intensity(usable_intensities[:, None, None],
#                            logger=logger,
#                            title=f"Intensity of usable flat images{title_suffix}",
#                            save_name=save_usable_flat,
#                            plot_graph=show_plots)
#
#     plot_single_image(average_flat,
#                       logger=logger,
#                       title=f"Picture of master flat{title_suffix}",
#                       save_name=save_master_flat,
#                       plot_graph=show_plots)
#
#     # Save the master flat to the specified folder
#     if save_folder is not None:
#         os.makedirs(save_folder, exist_ok=True)
#         save_path = os.path.join(save_folder, "master_flat.fits")
#         fits.writeto(save_path, average_flat, overwrite=True)
#         if logger is not None:
#             logger.info(f"Master flat saved to {save_path}")
#
#     return average_flat
#
#
# def testing_flat(flat_folder, idx=None, master_dark=None, logger=None, show_plots=True, batch_size=50):
#     flat_files = get_fits_filepaths(flat_folder)
#     if not flat_files:
#         raise FileNotFoundError(f"No FITS files found in {flat_folder}")
#
#     title_suffix = " (Dark-Corrected)" if master_dark is not None else ""
#
#     if logger:
#         logger.info(f"--- Testing flat intensity across {len(flat_files)} files ---")
#
#     _, all_intensities = process_fits_in_batches(flat_files, master_dark=master_dark, batch_size=batch_size,
#                                                  logger=logger)
#     plot_average_intensity(all_intensities[:, None, None], logger=logger, title=f"sun_area{title_suffix}",
#                            plot_graph=show_plots)
#     print(f"Raw flat count: {len(flat_files)}")
#
#     if idx is not None:
#         usable_files = flat_files[idx[0]:idx[1]]
#         print(f"Usable flat count: {len(usable_files)}")
#
#         average_flat, usable_intensities = process_fits_in_batches(usable_files, master_dark=master_dark,
#                                                                    batch_size=batch_size, logger=logger)
#         plot_average_intensity(usable_intensities[:, None, None], logger=logger, title=f"sun_area usable{title_suffix}",
#                                plot_graph=show_plots)
#         plot_single_image(average_flat, logger=logger, title=f"master flat preview{title_suffix}",
#                           plot_graph=show_plots)
#
#
# def save_dark(dark_folder, save_folder=None, logger=None, vmax=1000, show_plots=False, batch_size=50):
#     dark_files = get_fits_filepaths(dark_folder)
#     if not dark_files:
#         raise FileNotFoundError(f"No FITS files found in {dark_folder}")
#
#     if logger:
#         logger.info(f"--- Computing Master Dark from {len(dark_files)} files in batches of {batch_size} ---")
#
#     average_dark, _ = process_fits_in_batches(dark_files, master_dark=None, batch_size=batch_size, logger=logger)
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
#     return average_dark
#
#
# if __name__ == "__main__":
#     # =====================================================================
#     # CONFIGURATION PARAMETERS (All modifications go here!)
#     # =====================================================================
#     # --- Input Folders ---
#     FLAT_FOLDER = "./2024-05-10/sun_area/SlitJaw"
#     DARK_FOLDER = "./2024-05-10/sun_dark/SlitJaw"
#
#     # --- Output Folder ---
#     PARENT_SAVE_FOLDER = "./MASTER_SAVE"
#
#     # --- Memory & Processing Settings ---
#     BATCH_SIZE = 150  # Number of FITS files to load into RAM at once
#     FLAT_IDX = (0, 1200)  # Frame index slice (start, end) for usable flats
#     DARK_VMAX = 1000  # Intensity cap for dark frame preview plot
#
#     # --- Execution Toggles ---
#     RUN_FLAT_TESTING = True  # Set to True to test/preview flat index selection
#     PROCESS_MASTER_DARK = True  # Set to True to calculate & save master_dark.fits
#     PROCESS_MASTER_FLAT = True  # Set to True to calculate & save master_flat.fits
#     SHOW_PLOTS = True  # Set to True to display interactive plot windows
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
#     master_dark = None
#
#     # 1. Compute and Save Master Dark FIRST (Required for Dark-Correcting Flat)
#     if PROCESS_MASTER_DARK:
#         logger.info("--- Processing Master Dark ---")
#         master_dark = save_dark(dark_folder=DARK_FOLDER, save_folder=save_folder, logger=logger, vmax=DARK_VMAX,
#                                 show_plots=SHOW_PLOTS, batch_size=BATCH_SIZE)
#     else:
#         # Fallback: check if master_dark.fits already exists on disk
#         existing_dark_path = os.path.join(save_folder, "master_dark.fits")
#         if os.path.exists(existing_dark_path):
#             logger.info(f"--- Loading existing Master Dark from: {existing_dark_path} ---")
#             master_dark = load_fits_img(existing_dark_path)
#
#     # 2. OPTIONAL: Flat Range Testing Mode (includes dark correction if available)
#     if RUN_FLAT_TESTING:
#         logger.info("--- Running Flat Field Slicing Test ---")
#         testing_flat(flat_folder=FLAT_FOLDER, idx=FLAT_IDX, master_dark=master_dark, logger=logger,
#                      show_plots=SHOW_PLOTS, batch_size=BATCH_SIZE)
#
#     # 3. Compute and Save Dark-Corrected Master Flat
#     if PROCESS_MASTER_FLAT:
#         logger.info("--- Processing Dark-Corrected Master Flat ---")
#         save_data_flat(flat_folder=FLAT_FOLDER, idx=FLAT_IDX, save_folder=save_folder, master_dark=master_dark,
#                        logger=logger, show_plots=SHOW_PLOTS, batch_size=BATCH_SIZE)
#
#


import gc
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from astropy.io import fits

from data_io import *
from processing import *
from analysis import *
from plotting import *
from FICUS.PYTHON.OCAS_lib import *
from FICUS.PYTHON.NormalizationModule import *
from pipeline import Pipeline

def create_flat_animation(flat_folder, idx, save_folder, master_dark=None, fps=15, vmax=None, logger=None):
    """
    Loads flat images in the specified `idx` range, applies dark correction (if provided),
    extracts timestamps, cleans up unnecessary memory, and generates an MP4 animation
    without drawing a circular overlay region.
    """
    flat_files = get_fits_filepaths(flat_folder)
    if not flat_files:
        raise FileNotFoundError(f"No FITS files found in {flat_folder}")

    usable_files = flat_files[idx[0]:idx[1]]
    num_frames = len(usable_files)
    if num_frames == 0:
        raise ValueError("Selected index range contains no files.")

    if logger:
        logger.info(f"--- Loading {num_frames} flat images for animation ---")

    # 1. Inspect first image to pre-allocate NumPy array directly in RAM
    first_img = fits.getdata(usable_files[0]).astype(np.float32)
    h, w = first_img.shape
    imgs = np.empty((num_frames, h, w), dtype=np.float32)

    # Insert frame 0
    if master_dark is not None:
        first_img -= master_dark
    imgs[0] = first_img
    del first_img  # Free single frame buffer immediately

    # Load remaining frames directly into pre-allocated array (avoids memory duplication)
    for i, fpath in enumerate(usable_files[1:], start=1):
        img = fits.getdata(fpath).astype(np.float32)
        if master_dark is not None:
            img -= master_dark
        imgs[i] = img

    # 2. Extract and slice timestamps
    timestamps = compile_directory_timestamps(flat_folder)
    plot_time_axis = timestamps[idx[0]:idx[1]] if (timestamps and len(timestamps) >= idx[1]) else None
    del timestamps  # Clear raw timestamps list from memory

    save_path = os.path.join(save_folder, f"flat_sequence_animation_{idx[0]}_{idx[1]}.mp4") if save_folder else f"flat_sequence_animation_{idx[0]}_{idx[1]}.mp4"

    # 3. Clean up unneeded objects & trigger garbage collection before rendering
    if logger:
        logger.info("--- Performing garbage collection prior to animation rendering ---")
    gc.collect()

    if logger:
        logger.info(f"--- Generating animation: {save_path} ---")

    # 4. Generate animation video (xc, yc, r set to None to omit circle drawing)
    animate_eruption_region(
        imgs=imgs,
        xc=None,
        yc=None,
        r=None,
        time_series=plot_time_axis,
        fps=fps,
        vmax=vmax,
        save_name=save_path,
        logger=logger
    )

    # Final cleanup
    del imgs
    gc.collect()


def save_data_flat(flat_folder, idx, save_folder, master_dark=None, logger=None, show_plots=False, batch_size=50):
    flat_files = get_fits_filepaths(flat_folder)
    if not flat_files:
        raise FileNotFoundError(f"No FITS files found in {flat_folder}")

    # Target plot paths saved directly inside save_folder
    save_all_flat = os.path.join(save_folder,
                                 "intensity_of_all_flat_images") if save_folder else "intensity_of_all_flat_images"
    save_usable_flat = os.path.join(save_folder,
                                    "intensity_of_usable_flat_images") if save_folder else "intensity_of_usable_flat_images"
    save_master_flat = os.path.join(save_folder, "picture_of_master_flat") if save_folder else "picture_of_master_flat"

    title_suffix = " (Dark-Corrected)" if master_dark is not None else ""

    if logger:
        logger.info(f"--- Calculating intensity profile for all {len(flat_files)} flat images ---")

    # Process all flat images in batches to compute per-frame intensity profile
    _, all_intensities = process_fits_in_batches(flat_files, master_dark=master_dark, batch_size=batch_size,
                                                 logger=logger)

    # Pass lightweight (N, 1, 1) dummy shape to plotting function to avoid RAM spikes
    plot_average_intensity(all_intensities[:, None, None],
                           logger=logger,
                           title=f"Intensity of all flat images{title_suffix}",
                           save_name=save_all_flat,
                           plot_graph=show_plots)

    print(f"Raw flat count: {len(flat_files)}")

    # Slice usable files
    usable_files = flat_files[idx[0]:idx[1]]
    print(f"Usable flat count: {len(usable_files)}")

    if logger:
        logger.info(f"--- Computing Master Flat from {len(usable_files)} usable frames ---")

    average_flat, usable_intensities = process_fits_in_batches(usable_files, master_dark=master_dark,
                                                               batch_size=batch_size, logger=logger)

    plot_average_intensity(usable_intensities[:, None, None],
                           logger=logger,
                           title=f"Intensity of usable flat images{title_suffix}",
                           save_name=save_usable_flat,
                           plot_graph=show_plots)

    plot_single_image(average_flat,
                      logger=logger,
                      title=f"Picture of master flat{title_suffix}",
                      save_name=save_master_flat,
                      plot_graph=show_plots)

    # Save the master flat to the specified folder
    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, "master_flat.fits")
        fits.writeto(save_path, average_flat, overwrite=True)
        if logger is not None:
            logger.info(f"Master flat saved to {save_path}")

    return average_flat


def testing_flat(flat_folder, idx=None, master_dark=None, logger=None, show_plots=True, batch_size=50):
    flat_files = get_fits_filepaths(flat_folder)
    if not flat_files:
        raise FileNotFoundError(f"No FITS files found in {flat_folder}")

    title_suffix = " (Dark-Corrected)" if master_dark is not None else ""

    if logger:
        logger.info(f"--- Testing flat intensity across {len(flat_files)} files ---")

    _, all_intensities = process_fits_in_batches(flat_files, master_dark=master_dark, batch_size=batch_size,
                                                 logger=logger)
    plot_average_intensity(all_intensities[:, None, None], logger=logger, title=f"sun_area{title_suffix}",
                           plot_graph=show_plots)
    print(f"Raw flat count: {len(flat_files)}")

    if idx is not None:
        usable_files = flat_files[idx[0]:idx[1]]
        print(f"Usable flat count: {len(usable_files)}")

        average_flat, usable_intensities = process_fits_in_batches(usable_files, master_dark=master_dark,
                                                                   batch_size=batch_size, logger=logger)
        plot_average_intensity(usable_intensities[:, None, None], logger=logger, title=f"sun_area usable{title_suffix}",
                               plot_graph=show_plots)
        plot_single_image(average_flat, logger=logger, title=f"master flat preview{title_suffix}",
                          plot_graph=show_plots)


def save_dark(dark_folder, save_folder=None, logger=None, vmax=1000, show_plots=False, batch_size=50):
    dark_files = get_fits_filepaths(dark_folder)
    if not dark_files:
        raise FileNotFoundError(f"No FITS files found in {dark_folder}")

    if logger:
        logger.info(f"--- Computing Master Dark from {len(dark_files)} files in batches of {batch_size} ---")

    average_dark, _ = process_fits_in_batches(dark_files, master_dark=None, batch_size=batch_size, logger=logger)

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
    # CONFIGURATION PARAMETERS
    # =====================================================================
    FLAT_FOLDER = "./2024-05-10/sun_area/SlitJaw"
    DARK_FOLDER = "./2024-05-10/sun_dark/SlitJaw"
    PARENT_SAVE_FOLDER = "./MASTER_SAVE"

    BATCH_SIZE = 150
    FLAT_IDX = (0, 1200)
    DARK_VMAX = 1000

    CREATE_FLAT_ANIMATION = True
    ANIMATION_FPS = 10
    ANIMATION_VMAX = None

    RUN_FLAT_TESTING = False
    PROCESS_MASTER_DARK = True
    PROCESS_MASTER_FLAT = True
    SHOW_PLOTS = False  # Set to False during heavy runs to save RAM
    # =====================================================================

    logger = setup_logger()

    ref_folder = FLAT_FOLDER if os.path.exists(FLAT_FOLDER) else DARK_FOLDER
    astropy_time_array = compile_directory_timestamps(ref_folder)
    date_suffix = astropy_time_array[0].datetime.strftime("%Y-%m-%d") if len(astropy_time_array) > 0 else "0000-00-00"

    save_folder = os.path.join(PARENT_SAVE_FOLDER, date_suffix)
    os.makedirs(save_folder, exist_ok=True)

    master_dark = None

    # 1. Compute and Save Master Dark
    if PROCESS_MASTER_DARK:
        logger.info("--- Processing Master Dark ---")
        master_dark = save_dark(dark_folder=DARK_FOLDER, save_folder=save_folder, logger=logger, vmax=DARK_VMAX,
                                show_plots=SHOW_PLOTS, batch_size=BATCH_SIZE)
    else:
        existing_dark_path = os.path.join(save_folder, "master_dark.fits")
        if os.path.exists(existing_dark_path):
            logger.info(f"--- Loading existing Master Dark from: {existing_dark_path} ---")
            master_dark = load_fits_img(existing_dark_path)

    # 2. OPTIONAL: Flat Range Testing Mode
    if RUN_FLAT_TESTING:
        logger.info("--- Running Flat Field Slicing Test ---")
        testing_flat(flat_folder=FLAT_FOLDER, idx=FLAT_IDX, master_dark=master_dark, logger=logger,
                     show_plots=SHOW_PLOTS, batch_size=BATCH_SIZE)

    # 3. Compute and Save Master Flat
    if PROCESS_MASTER_FLAT:
        logger.info("--- Processing Dark-Corrected Master Flat ---")
        master_flat = save_data_flat(flat_folder=FLAT_FOLDER, idx=FLAT_IDX, save_folder=save_folder,
                                     master_dark=master_dark, logger=logger, show_plots=SHOW_PLOTS,
                                     batch_size=BATCH_SIZE)

        # Free master_flat array from memory (it's already written to disk)
        del master_flat

    # =====================================================================
    # HARD MEMORY PURGE BEFORE ANIMATION
    # =====================================================================
    logger.info("--- Purging Matplotlib buffers and triggering GC before animation ---")
    plt.close('all')  # Close all open plot figures holding image canvases in RAM
    gc.collect()  # Force immediate garbage collection sweep
    # =====================================================================

    # 4. Generate Flat Sequence Animation
    if CREATE_FLAT_ANIMATION:
        logger.info("--- Generating Flat Sequence Animation ---")
        create_flat_animation(
            flat_folder=FLAT_FOLDER,
            idx=FLAT_IDX,
            save_folder=save_folder,
            master_dark=master_dark,  # Kept in RAM (only ~16 MB)
            fps=ANIMATION_FPS,
            vmax=ANIMATION_VMAX,
            logger=logger
        )