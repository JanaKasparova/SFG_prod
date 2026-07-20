# import gc
# import time
# import os
# import numpy as np
# from data_io import *
# from processing import *
# from analysis import *
# from plotting import *
#
#
# def get_calibrated_data(
#         data_folder: str,
#         xmin: int,
#         xmax: int,
#         ymin: int,
#         ymax: int,
#         batch_size: int,
#         n_images_grid: int,
#         master_flat_path: str,
#         master_dark_path: str,
#         cache_dir: str = "CACHE_DATA",
#         plots_dir: str = "Plots",
#         date_suffix: str = "",
#         logger=None
# ) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Pure pipeline engine. It takes all parameters explicitly from the caller,
#     checks for cached .npy files, or runs the calibration if a cache miss occurs.
#     Saves diagnostics directly to the centralized plots directory using a date suffix.
#     """
#     os.makedirs(cache_dir, exist_ok=True)
#     os.makedirs(plots_dir, exist_ok=True)
#
#     cache_flat_path = os.path.join(cache_dir, "final_flat_corrected_data.npy")
#     cache_dark_path = os.path.join(cache_dir, "cropped_dark.npy")
#
#     # ---- CHECK CACHE ----
#     if os.path.exists(cache_flat_path) and os.path.exists(cache_dark_path):
#         if logger:
#             logger.info("🚀 Found cached data! Loading precomputed arrays instantly...")
#         return np.load(cache_flat_path), np.load(cache_dark_path)
#
#     # ---- COMPUTE DATA (Cache Miss) ----
#     if logger:
#         logger.info("⏳ Cache missing. Executing calibration pipeline...")
#
#     # Load calibration masters from provided paths
#     master_flat = load_fits_img(master_flat_path)
#     master_dark = load_fits_img(master_dark_path)
#
#     # Crop calibration frames
#     cropped_flat = crop_images(master_flat, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, logger=logger)
#     cropped_dark = crop_images(master_dark, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, logger=logger)
#
#     # Load and crop raw science data
#     fits_data = load_fits(data_folder, logger=logger)
#     data_cropped = crop_images(fits_data, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, logger=logger)
#
#     # Base grid save names incorporating the dynamic date suffix
#     suffix_str = f"_{date_suffix}" if date_suffix else ""
#     grid_before_name = os.path.join(plots_dir, f"grid_before_correction{suffix_str}")
#     grid_dark_name = os.path.join(plots_dir, f"grid_after_dark_correction{suffix_str}")
#     grid_flat_name = os.path.join(plots_dir, f"grid_after_flat_correction{suffix_str}")
#
#     plot_image_grid(images=data_cropped, n_images=n_images_grid, save_name=grid_before_name, plot_image=False)
#
#     # Dark correction & alignment
#     dark_cor = correct_dark(data_cropped, cropped_dark, align_dark=True, batch_size=batch_size, logger=logger)
#     plot_image_grid(images=dark_cor, n_images=n_images_grid, save_name=grid_dark_name, plot_image=False)
#
#     # Flat correction & alignment
#     flat_cor = correct_flat(data_cropped, cropped_flat, align_flat=True, batch_size=batch_size, logger=logger)
#     plot_image_grid(images=flat_cor, n_images=n_images_grid, save_name=grid_flat_name, plot_image=False)
#
#     # ---- SAVE BOTH TO CACHE ----
#     np.save(cache_flat_path, flat_cor)
#     np.save(cache_dark_path, cropped_dark)
#
#     if logger:
#         logger.info(f"💾 Data arrays successfully cached inside: {cache_dir}/")
#
#     return flat_cor, cropped_dark
#
#
# # =====================================================================
# # GLOBAL PIPELINE RUNNER
# # =====================================================================
# if __name__ == "__main__":
#     # -----------------------------------------------------------------
#     # CONFIGURATION PARAMETERS (All modifications go here!)
#     # -----------------------------------------------------------------
#     # --- Paths & Directories ---
#     FITS_DATA_FOLDER = "./2024-07-29/sun_area/SlitJaw"
#     FILE_MASTER_FLAT = "MASTER_SAVE/master_flat.fits"
#     FILE_MASTER_DARK = "MASTER_SAVE/master_dark.fits"
#     DIR_CACHE = "CACHE_DATA"
#     DIR_PLOTS = "Plots/2024-07-29/test"  # Centralized plots folder
#
#     # --- Core Calibration Crop Bounds ---
#     CALIB_XMIN, CALIB_XMAX = 500, 1500
#     CALIB_YMIN, CALIB_YMAX = 100, 1000
#     PROC_BATCH_SIZE = 100
#     GRID_PREVIEW_IMG = 4
#
#     # --- Dark Frame Analysis Setup ---
#     DARK_FIGSIZE = (14, 10)
#
#     # --- Reference Box Bounds & Metrics (analyze_and_plot_rect) ---
#     REF_XMIN, REF_XMAX = 50, 150
#     REF_YMIN, REF_YMAX = 450, 550
#     REF_NUM_PLOTS = 4
#     REF_VMAX = 1.5
#
#     # --- Eruption Crop Box Shifts & Parameters ---
#     ERUPTION_POSUN_X = 52
#     ERUPTION_POSUN_Y = 45
#
#     ERUPTION_XMIN = 380 + ERUPTION_POSUN_X
#     ERUPTION_XMAX = 520 + ERUPTION_POSUN_X
#     ERUPTION_YMIN = 280 + ERUPTION_POSUN_Y
#     ERUPTION_YMAX = 420 + ERUPTION_POSUN_Y
#
#     # --- Eruption Circular Mask Region ---
#     ERUPTION_CENTER = (69, 70)  # (xC, yC)
#     ERUPTION_RADIUS = 52  # R
#
#     # --- Downstream Analytics Fine-Tuning ---
#     ERUPTION_NUM_PLOTS = 4
#     ERUPTION_VMAX = 2
#     ERUPTION_SIGMA_LEVEL = 5.0
#     CONTOUR_NUM_PLOTS = 12
#     CONTOUR_MIN_SIGMA = 5
#
#     # --- Global Plot Visibility Toggles ---
#     SHOW_DIAGNOSTIC_PLOTS = False
#     # -----------------------------------------------------------------
#
#     # Initialize logging ecosystem
#     logger = setup_logger()
#
#     # Ensure output directories exist up-front
#     os.makedirs(DIR_PLOTS, exist_ok=True)
#
#     # =====================================================================
#     # STEP 1: LOAD TIMESTAMPS & GENERATE DATE SUFFIX FIRST
#     # =====================================================================
#     astropy_time_array = compile_directory_timestamps(FITS_DATA_FOLDER)
#
#     # Dynamically extract date suffix (e.g., '2024-07-29')
#     date_suffix = astropy_time_array[0].datetime.strftime("%Y-%m-%d") if len(astropy_time_array) > 0 else "0000-00-00"
#
#     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#     print(f"Array Type:  {type(astropy_time_array)}")
#     print(f"Array Shape: {astropy_time_array.shape}")
#     print(f"Data Date:   {date_suffix}")
#     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#
#     # Convert the astropy time array into a Python datetime array using .to_datetime()
#     plot_time_axis = astropy_time_array.to_datetime() if len(astropy_time_array) > 0 else None
#
#     # plot_single_series(plot_time_axis)
#
#     # --- DYNAMIC PLOT OUTPUT SAVE PATHS (All mapped to DIR_PLOTS with Date Suffix) ---
#     SAVE_DARK_HIST = os.path.join(DIR_PLOTS, f"dark_histograms_{date_suffix}")
#     SAVE_HOT_PIX = os.path.join(DIR_PLOTS, f"hot_pixel_map_{date_suffix}")
#     SAVE_DARK_TABLE = os.path.join(DIR_PLOTS, f"dark_analysis_table_{date_suffix}")
#     SAVE_REF_GRID = os.path.join(DIR_PLOTS, f"means_ref_grid_image_{date_suffix}")
#     SAVE_REF_STATS = os.path.join(DIR_PLOTS, f"means_std_ref_plot_{date_suffix}")
#     SAVE_ERUPT_GRID = os.path.join(DIR_PLOTS, f"means_eruption_grid_image_{date_suffix}")
#     SAVE_ERUPT_STATS = os.path.join(DIR_PLOTS, f"means_std_eruption_plot_{date_suffix}")
#     SAVE_INTENSITY = os.path.join(DIR_PLOTS, f"eruption_intensity_plot_{date_suffix}")
#     SAVE_CONTOURS = os.path.join(DIR_PLOTS, f"eruption_contours_grid_{date_suffix}.png")
#
#     # =====================================================================
#     # STEP 2: EXECUTE PRIMARY DATA CALIBRATION PIPELINE
#     # =====================================================================
#     flat_cor, cropped_dark = get_calibrated_data(
#         data_folder=FITS_DATA_FOLDER,
#         xmin=CALIB_XMIN,
#         xmax=CALIB_XMAX,
#         ymin=CALIB_YMIN,
#         ymax=CALIB_YMAX,
#         batch_size=PROC_BATCH_SIZE,
#         n_images_grid=GRID_PREVIEW_IMG,
#         master_flat_path=FILE_MASTER_FLAT,
#         master_dark_path=FILE_MASTER_DARK,
#         cache_dir=DIR_CACHE,
#         plots_dir=DIR_PLOTS,
#         date_suffix=date_suffix,
#         logger=logger
#     )
#
#     logger.info(f"Pipeline finished! Calibration Stack: {flat_cor.shape} | Cropped Dark: {cropped_dark.shape}")
#
#     # =====================================================================
#     # STEP 3: STABILITY AND NOISE ANALYSIS (DARK FRAMES)
#     # =====================================================================
#     dark_stats = analyze_dark_frame(cropped_dark, logger=logger)
#
#     plot_dark_histograms(
#         dark_stats,
#         logger=logger,
#         figsize=DARK_FIGSIZE,
#         save_name=SAVE_DARK_HIST,
#         plot_histograms=SHOW_DIAGNOSTIC_PLOTS
#     )
#
#     plot_hot_pixel_map(
#         dark_stats,
#         logger=logger,
#         save_name=SAVE_HOT_PIX,
#         plot_map=SHOW_DIAGNOSTIC_PLOTS
#     )
#
#     print_dark_analysis_table(
#         dark_stats,
#         logger=logger,
#         plot_table=SHOW_DIAGNOSTIC_PLOTS,
#         save_name=SAVE_DARK_TABLE
#     )
#
#     # =====================================================================
#     # STEP 4: REFERENCE RECTANGLE TRACKING
#     # =====================================================================
#     means_ref, stds_ref = analyze_and_plot_rect(
#         imgs=flat_cor,
#         xmin=REF_XMIN, xmax=REF_XMAX,
#         ymin=REF_YMIN, ymax=REF_YMAX,
#         num_plots=REF_NUM_PLOTS,
#         vmax=REF_VMAX,
#         save_name=SAVE_REF_GRID,
#         plot_graphs=SHOW_DIAGNOSTIC_PLOTS
#     )
#
#     # # 1. Crop the flat-corrected stack to your reference bounding box boundaries
#     # flat_cor_cropped = crop_images(flat_cor, REF_XMIN, REF_YMIN, REF_XMAX, REF_YMAX, logger=logger)
#     #
#     # # 2. Calculate the exact center point inside this new cropped frame
#     # REF_CIRC_XC = (REF_XMAX - REF_XMIN) / 2.0
#     # REF_CIRC_YC = (REF_YMAX - REF_YMIN) / 2.0
#     #
#     # # 3. Guardrail check: Ensure your reference box is large enough to contain the eruption circle
#     # if (REF_XMAX - REF_XMIN) < (2 * ERUPTION_RADIUS) or (REF_YMAX - REF_YMIN) < (2 * ERUPTION_RADIUS):
#     #     print("⚠️ WARNING: Your reference bounding box is smaller than the eruption circle diameter!")
#     #
#     # # 4. Run the circular analysis on the cropped reference region
#     # means_ref, stds_ref = analyze_and_plot_circ(
#     #     imgs=flat_cor_cropped,
#     #     xc=REF_CIRC_XC,
#     #     yc=REF_CIRC_YC,
#     #     r=ERUPTION_RADIUS,  # Uses the exact same circle size as the eruption!
#     #     num_plots=REF_NUM_PLOTS,
#     #     vmax=REF_VMAX,
#     #     save_name=SAVE_REF_GRID,
#     #     plot_graphs=SHOW_DIAGNOSTIC_PLOTS,
#     #     logger=logger
#     # )
#
#     plot_stats(
#         means=means_ref,
#         stds=stds_ref,
#         time_series=plot_time_axis,
#         title_mean="Reference mean",
#         title_std="Reference std",
#         save_name=SAVE_REF_STATS,
#         plot_stats=SHOW_DIAGNOSTIC_PLOTS
#     )
#
#     # =====================================================================
#     # STEP 5: ERUPTION REGION TRACKING & ACTIVE PIXEL RATIOS
#     # =====================================================================
#     eruption_images = crop_images(
#         flat_cor,
#         xmin=ERUPTION_XMIN,
#         xmax=ERUPTION_XMAX,
#         ymin=ERUPTION_YMIN,
#         ymax=ERUPTION_YMAX,
#         logger=logger
#     )
#
#     means_circ, std_circ = analyze_and_plot_circ(
#         imgs=eruption_images,
#         xc=ERUPTION_CENTER[0],
#         yc=ERUPTION_CENTER[1],
#         r=ERUPTION_RADIUS,
#         num_plots=ERUPTION_NUM_PLOTS,
#         vmax=ERUPTION_VMAX,
#         save_name=SAVE_ERUPT_GRID,
#         plot_graphs=SHOW_DIAGNOSTIC_PLOTS
#     )
#
#     plot_stats(
#         means=means_circ,
#         stds=std_circ,
#         time_series=plot_time_axis,
#         title_mean="Eruption region mean",
#         title_std="Eruption region std",
#         save_name=SAVE_ERUPT_STATS,
#         plot_stats=SHOW_DIAGNOSTIC_PLOTS
#     )
#
#     # Extract dynamic threshold active masks
#     erupting_ratios, total_pixels = calculate_erupting_pixels(
#         imgs=eruption_images,
#         xc=ERUPTION_CENTER[0],
#         yc=ERUPTION_CENTER[1],
#         r=ERUPTION_RADIUS,
#         mean_ref=means_ref,
#         std_ref=stds_ref,
#         sigma_level=ERUPTION_SIGMA_LEVEL,
#         logger=logger
#     )
#
#     plot_single_series(
#         data=erupting_ratios,
#         # time_series=plot_time_axis,
#         title="Eruption Intensity Per Frame",
#         y_label="Active Pixel Ratio",
#         logger=logger,
#         save_name=SAVE_INTENSITY
#     )
#
#     plot_eruption_contours(
#         imgs=eruption_images,
#         xc=ERUPTION_CENTER[0],
#         yc=ERUPTION_CENTER[1],
#         r=ERUPTION_RADIUS,
#         mean_ref=means_ref,
#         std_ref=stds_ref,
#         num_plots=CONTOUR_NUM_PLOTS,
#         min_sigma=CONTOUR_MIN_SIGMA,
#         save_name=SAVE_CONTOURS
#     )


import gc
import time
import os
import numpy as np
from data_io import *
from processing import *
from analysis import *
from plotting import *


def get_calibrated_data(
        data_folder: str,
        xmin: int,
        xmax: int,
        ymin: int,
        ymax: int,
        batch_size: int,
        n_images_grid: int,
        master_flat_path: str,
        master_dark_path: str,
        cache_dir: str = "CACHE_DATA",
        plots_dir: str = "Plots",
        date_suffix: str = "",
        logger=None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pure pipeline engine. It takes all parameters explicitly from the caller,
    checks for cached .npy files, or runs the calibration if a cache miss occurs.
    Saves diagnostics directly to the centralized plots directory using a date suffix.
    """
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    cache_flat_path = os.path.join(cache_dir, "final_flat_corrected_data.npy")
    cache_dark_path = os.path.join(cache_dir, "cropped_dark.npy")

    # ---- CHECK CACHE ----
    if os.path.exists(cache_flat_path) and os.path.exists(cache_dark_path):
        if logger:
            logger.info("🚀 Found cached data! Loading precomputed arrays instantly...")
        return np.load(cache_flat_path), np.load(cache_dark_path)

    # ---- COMPUTE DATA (Cache Miss) ----
    if logger:
        logger.info("⏳ Cache missing. Executing calibration pipeline...")

    # Load calibration masters from provided paths
    master_flat = load_fits_img(master_flat_path)
    master_dark = load_fits_img(master_dark_path)

    # Crop calibration frames
    cropped_flat = crop_images(master_flat, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, logger=logger)
    cropped_dark = crop_images(master_dark, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, logger=logger)

    # Load and crop raw science data
    fits_data = load_fits(data_folder, logger=logger)
    data_cropped = crop_images(fits_data, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, logger=logger)

    # Base grid save names incorporating the dynamic date suffix
    suffix_str = f"_{date_suffix}" if date_suffix else ""
    grid_before_name = os.path.join(plots_dir, f"grid_before_correction{suffix_str}")
    grid_dark_name = os.path.join(plots_dir, f"grid_after_dark_correction{suffix_str}")
    grid_flat_name = os.path.join(plots_dir, f"grid_after_flat_correction{suffix_str}")

    plot_image_grid(images=data_cropped, n_images=n_images_grid, save_name=grid_before_name, plot_image=False)

    # Dark correction & alignment
    dark_cor = correct_dark(data_cropped, cropped_dark, align_dark=True, batch_size=batch_size, logger=logger)
    plot_image_grid(images=dark_cor, n_images=n_images_grid, save_name=grid_dark_name, plot_image=False)

    # Flat correction & alignment
    flat_cor = correct_flat(data_cropped, cropped_flat, align_flat=True, batch_size=batch_size, logger=logger)
    plot_image_grid(images=flat_cor, n_images=n_images_grid, save_name=grid_flat_name, plot_image=False)

    # ---- SAVE BOTH TO CACHE ----
    np.save(cache_flat_path, flat_cor)
    np.save(cache_dark_path, cropped_dark)

    if logger:
        logger.info(f"💾 Data arrays successfully cached inside: {cache_dir}/")

    return flat_cor, cropped_dark


# =====================================================================
# GLOBAL PIPELINE RUNNER
# =====================================================================
if __name__ == "__main__":
    # -----------------------------------------------------------------
    # CONFIGURATION PARAMETERS (All modifications go here!)
    # -----------------------------------------------------------------
    # --- Paths & Directories ---
    FITS_DATA_FOLDER = "./2024-07-29/sun_area/SlitJaw"
    FILE_MASTER_FLAT = "MASTER_SAVE/master_flat.fits"
    FILE_MASTER_DARK = "MASTER_SAVE/master_dark.fits"
    DIR_CACHE = "CACHE_DATA"
    DIR_PLOTS = "Plots/2024-07-29/test"  # Centralized plots folder

    # --- Frame Trimming / Cut Settings ---
    CUT_FRONT = 0
    CUT_BACK = 100

    # --- Core Calibration Crop Bounds ---
    CALIB_XMIN, CALIB_XMAX = 500, 1500
    CALIB_YMIN, CALIB_YMAX = 100, 1000
    PROC_BATCH_SIZE = 100
    GRID_PREVIEW_IMG = 4

    # --- Dark Frame Analysis Setup ---
    DARK_FIGSIZE = (14, 10)

    # --- Reference Box Bounds & Metrics (analyze_and_plot_rect) ---
    REF_XMIN, REF_XMAX = 50, 150
    REF_YMIN, REF_YMAX = 450, 550
    REF_NUM_PLOTS = 4
    REF_VMAX = 1.5

    # --- Eruption Crop Box Shifts & Parameters ---
    ERUPTION_POSUN_X = 52
    ERUPTION_POSUN_Y = 45

    ERUPTION_XMIN = 380 + ERUPTION_POSUN_X
    ERUPTION_XMAX = 520 + ERUPTION_POSUN_X
    ERUPTION_YMIN = 280 + ERUPTION_POSUN_Y
    ERUPTION_YMAX = 420 + ERUPTION_POSUN_Y

    # --- Eruption Circular Mask Region ---
    ERUPTION_CENTER = (69, 70)  # (xC, yC)
    ERUPTION_RADIUS = 52  # R

    # --- Downstream Analytics Fine-Tuning ---
    ERUPTION_NUM_PLOTS = 4
    ERUPTION_VMAX = 2
    ERUPTION_SIGMA_LEVEL = 5.0
    CONTOUR_NUM_PLOTS = 12
    CONTOUR_MIN_SIGMA = 5

    # --- Global Plot Visibility Toggles ---
    SHOW_DIAGNOSTIC_PLOTS = False
    # -----------------------------------------------------------------

    # Initialize logging ecosystem
    logger = setup_logger()

    # Ensure output directories exist up-front
    os.makedirs(DIR_PLOTS, exist_ok=True)

    # =====================================================================
    # STEP 1: LOAD TIMESTAMPS & GENERATE DATE SUFFIX FIRST
    # =====================================================================
    astropy_time_array = compile_directory_timestamps(FITS_DATA_FOLDER)

    # Dynamically extract date suffix (e.g., '2024-07-29')
    date_suffix = astropy_time_array[0].datetime.strftime("%Y-%m-%d") if len(astropy_time_array) > 0 else "0000-00-00"

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"Array Type:  {type(astropy_time_array)}")
    print(f"Array Shape: {astropy_time_array.shape}")
    print(f"Data Date:   {date_suffix}")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    # Convert the astropy time array into a Python datetime array using .to_datetime()
    plot_time_axis = astropy_time_array.to_datetime() if len(astropy_time_array) > 0 else None

    # --- DYNAMIC PLOT OUTPUT SAVE PATHS (All mapped to DIR_PLOTS with Date Suffix) ---
    SAVE_DARK_HIST = os.path.join(DIR_PLOTS, f"dark_histograms_{date_suffix}")
    SAVE_HOT_PIX = os.path.join(DIR_PLOTS, f"hot_pixel_map_{date_suffix}")
    SAVE_DARK_TABLE = os.path.join(DIR_PLOTS, f"dark_analysis_table_{date_suffix}")
    SAVE_REF_GRID = os.path.join(DIR_PLOTS, f"means_ref_grid_image_{date_suffix}")
    SAVE_REF_STATS = os.path.join(DIR_PLOTS, f"means_std_ref_plot_{date_suffix}")
    SAVE_ERUPT_GRID = os.path.join(DIR_PLOTS, f"means_eruption_grid_image_{date_suffix}")
    SAVE_ERUPT_STATS = os.path.join(DIR_PLOTS, f"means_std_eruption_plot_{date_suffix}")
    SAVE_INTENSITY = os.path.join(DIR_PLOTS, f"eruption_intensity_plot_{date_suffix}")
    SAVE_CONTOURS = os.path.join(DIR_PLOTS, f"eruption_contours_grid_{date_suffix}.png")

    # =====================================================================
    # STEP 2: EXECUTE PRIMARY DATA CALIBRATION PIPELINE
    # =====================================================================
    flat_cor, cropped_dark = get_calibrated_data(
        data_folder=FITS_DATA_FOLDER,
        xmin=CALIB_XMIN,
        xmax=CALIB_XMAX,
        ymin=CALIB_YMIN,
        ymax=CALIB_YMAX,
        batch_size=PROC_BATCH_SIZE,
        n_images_grid=GRID_PREVIEW_IMG,
        master_flat_path=FILE_MASTER_FLAT,
        master_dark_path=FILE_MASTER_DARK,
        cache_dir=DIR_CACHE,
        plots_dir=DIR_PLOTS,
        date_suffix=date_suffix,
        logger=logger
    )

    logger.info(f"Pipeline finished! Calibration Stack: {flat_cor.shape} | Cropped Dark: {cropped_dark.shape}")

    # --- APPLY SPECIFIED PICTURE CUTS FROM FRONT/BACK AFTER INITIAL ANALYSIS ---
    # Safe guard checking the stack length before indexing
    end_idx = flat_cor.shape[0] - CUT_BACK if CUT_BACK > 0 else flat_cor.shape[0]
    start_idx = CUT_FRONT if CUT_FRONT > 0 else 0

    if start_idx < end_idx:
        flat_cor = flat_cor[start_idx:end_idx]
        if plot_time_axis is not None:
            plot_time_axis = plot_time_axis[start_idx:end_idx]
        if logger:
            logger.info(f"✂️ Scaled dataset: Cut {CUT_FRONT} from front, {CUT_BACK} from back. New Stack: {flat_cor.shape}")
    else:
        if logger:
            logger.warning("⚠️ Cut parameters are larger than total frames available! Skipping cut.")

    # =====================================================================
    # STEP 3: STABILITY AND NOISE ANALYSIS (DARK FRAMES)
    # =====================================================================
    dark_stats = analyze_dark_frame(cropped_dark, logger=logger)

    plot_dark_histograms(
        dark_stats,
        logger=logger,
        figsize=DARK_FIGSIZE,
        save_name=SAVE_DARK_HIST,
        plot_histograms=SHOW_DIAGNOSTIC_PLOTS
    )

    plot_hot_pixel_map(
        dark_stats,
        logger=logger,
        save_name=SAVE_HOT_PIX,
        plot_map=SHOW_DIAGNOSTIC_PLOTS
    )

    print_dark_analysis_table(
        dark_stats,
        logger=logger,
        plot_table=SHOW_DIAGNOSTIC_PLOTS,
        save_name=SAVE_DARK_TABLE
    )

    # =====================================================================
    # STEP 4: REFERENCE RECTANGLE TRACKING
    # =====================================================================
    means_ref, stds_ref = analyze_and_plot_rect(
        imgs=flat_cor,
        xmin=REF_XMIN, xmax=REF_XMAX,
        ymin=REF_YMIN, ymax=REF_YMAX,
        num_plots=REF_NUM_PLOTS,
        vmax=REF_VMAX,
        save_name=SAVE_REF_GRID,
        plot_graphs=SHOW_DIAGNOSTIC_PLOTS
    )

    plot_stats(
        means=means_ref,
        stds=stds_ref,
        time_series=plot_time_axis,
        title_mean="Reference mean",
        title_std="Reference std",
        save_name=SAVE_REF_STATS,
        plot_stats=SHOW_DIAGNOSTIC_PLOTS
    )

    # =====================================================================
    # STEP 5: ERUPTION REGION TRACKING & ACTIVE PIXEL RATIOS
    # =====================================================================
    eruption_images = crop_images(
        flat_cor,
        xmin=ERUPTION_XMIN,
        xmax=ERUPTION_XMAX,
        ymin=ERUPTION_YMIN,
        ymax=ERUPTION_YMAX,
        logger=logger
    )

    means_circ, std_circ = analyze_and_plot_circ(
        imgs=eruption_images,
        xc=ERUPTION_CENTER[0],
        yc=ERUPTION_CENTER[1],
        r=ERUPTION_RADIUS,
        num_plots=ERUPTION_NUM_PLOTS,
        vmax=ERUPTION_VMAX,
        save_name=SAVE_ERUPT_GRID,
        plot_graphs=SHOW_DIAGNOSTIC_PLOTS
    )

    plot_stats(
        means=means_circ,
        stds=std_circ,
        time_series=plot_time_axis,
        title_mean="Eruption region mean",
        title_std="Eruption region std",
        save_name=SAVE_ERUPT_STATS,
        plot_stats=SHOW_DIAGNOSTIC_PLOTS
    )

    # Extract dynamic threshold active masks
    erupting_ratios, total_pixels = calculate_erupting_pixels(
        imgs=eruption_images,
        xc=ERUPTION_CENTER[0],
        yc=ERUPTION_CENTER[1],
        r=ERUPTION_RADIUS,
        mean_ref=means_ref,
        std_ref=stds_ref,
        sigma_level=ERUPTION_SIGMA_LEVEL,
        logger=logger
    )

    plot_single_series(
        data=erupting_ratios,
        time_series=plot_time_axis,
        title="Eruption Intensity Per Frame",
        y_label="Active Pixel Ratio",
        num_ticks=15,  # Sets ~7 ticks on the x-axis
        logger=logger,
        save_name=SAVE_INTENSITY
    )

    plot_eruption_contours(
        imgs=eruption_images,
        xc=ERUPTION_CENTER[0],
        yc=ERUPTION_CENTER[1],
        r=ERUPTION_RADIUS,
        mean_ref=means_ref,
        std_ref=stds_ref,
        num_plots=CONTOUR_NUM_PLOTS,
        min_sigma=CONTOUR_MIN_SIGMA,
        save_name=SAVE_CONTOURS
    )