import gc
import time
import os
import numpy as np
from matplotlib.pyplot import xlabel
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
        logger=None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pure pipeline engine. It takes all parameters explicitly from the caller,
    checks for cached .npy files, or runs the calibration if a cache miss occurs.
    """
    os.makedirs(cache_dir, exist_ok=True)
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

    plot_image_grid(images=data_cropped, n_images=n_images_grid, save_name="grid_before_correction", plot_image=False)

    # Dark correction & alignment
    dark_cor = correct_dark(data_cropped, cropped_dark, align_dark=True, batch_size=batch_size, logger=logger)
    plot_image_grid(images=dark_cor, n_images=n_images_grid, save_name="grid_after_dark_correction", plot_image=False)

    # Flat correction & alignment
    flat_cor = correct_flat(data_cropped, cropped_flat, align_flat=True, batch_size=batch_size, logger=logger)
    plot_image_grid(images=flat_cor, n_images=n_images_grid, save_name="grid_after_flat_correction", plot_image=False)

    # ---- SAVE BOTH TO CACHE ----
    np.save(cache_flat_path, flat_cor)
    np.save(cache_dark_path, cropped_dark)

    if logger:
        logger.info(f"💾 Data arrays successfully cached inside: {cache_dir}/")

    return flat_cor, cropped_dark



if __name__ == "__main__":
    # logger = setup_logger()
    #
    # master_flat = load_fits_img("MASTER_SAVE/master_flat.fits")
    # master_dark = load_fits_img("MASTER_SAVE/master_dark.fits")
    #
    # c_y_min, c_y_max = 100, 1000
    # c_x_min, c_x_max = 500, 1500
    #
    # cropped_flat = crop_images(master_flat,
    #                            xmin=c_x_min,
    #                            xmax=c_x_max,
    #                            ymin=c_y_min,
    #                            ymax=c_y_max,
    #                            logger=logger)
    # cropped_dark = crop_images(master_dark,
    #                            xmin=c_x_min,
    #                            xmax=c_x_max,
    #                            ymin=c_y_min,
    #                            ymax=c_y_max,
    #                            logger=logger)
    #
    # fits_data_folder = "./2024-07-29/sun_area/SlitJaw"
    # fits_data = load_fits(fits_data_folder, logger=logger)
    #
    # data_cropped = crop_images(fits_data,
    #                            xmin=c_x_min,
    #                            xmax=c_x_max,
    #                            ymin=c_y_min,
    #                            ymax=c_y_max,
    #                            logger=logger)
    #
    # plot_image_grid(images=data_cropped, n_images=4,
    #                 save_name="grid_before_correction",
    #                 plot_image=False)
    #
    # dark_cor = correct_dark(data_cropped,
    #                         cropped_dark,
    #                         align_dark=True,
    #                         batch_size=100,  # Processes 100 images at a time
    #                         logger=logger)
    # plot_image_grid(images=dark_cor, n_images=4,
    #                 save_name="grid_after_dark_correction",
    #                 plot_image=False)
    #
    # flat_cor = correct_flat(data_cropped,
    #                         cropped_flat,
    #                         align_flat=True,
    #                         batch_size=100,  # Processes 100 images at a time
    #                         logger=logger)
    # plot_image_grid(images=flat_cor, n_images=4,
    #                 save_name="grid_after_flat_correction",
    #                 plot_image=False)

    # Initialize logger
    logger = setup_logger()

    # --- YOUR DATA CONFIGURATION (All in one place) ---
    fits_data_folder = "./2024-07-29/sun_area/SlitJaw"
    m_flat_file = "MASTER_SAVE/master_flat.fits"
    m_dark_file = "MASTER_SAVE/master_dark.fits"

    # Cropping Parameters
    c_x_min, c_x_max = 500, 1500
    c_y_min, c_y_max = 100, 1000

    # Processing Parameters
    proc_batch_size = 100
    grid_preview_img = 4
    # --------------------------------------------------

    # Fire the calibration passing everything explicitly from here
    flat_cor, cropped_dark = get_calibrated_data(
        data_folder=fits_data_folder,
        xmin=c_x_min,
        xmax=c_x_max,
        ymin=c_y_min,
        ymax=c_y_max,
        batch_size=proc_batch_size,
        n_images_grid=grid_preview_img,
        master_flat_path=m_flat_file,
        master_dark_path=m_dark_file,
        logger=logger
    )

    # extracting time from fits names

    # Compile the array from your directory
    astropy_time_array = compile_directory_timestamps(fits_data_folder)

    # Verify the structure
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"Array Type:  {type(astropy_time_array)}")
    print(f"Array Shape: {astropy_time_array.shape}")
    print(f"Data Type:   {astropy_time_array.dtype}")

    if len(astropy_time_array) > 0:
        print("\nFirst 3 elements in the NumPy array:")
        for t in astropy_time_array[:3]:
            print(f" -> {t}")

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


    # Your data arrays are now fully ready for any downstream processing
    logger.info(f"Pipeline finished! Calibration Stack: {flat_cor.shape} | Cropped Dark: {cropped_dark.shape}")


    # dark analysis
    # 1. Run the math ONCE
    dark_stats = analyze_dark_frame(cropped_dark, logger=logger)

    # # 2. Plot the histograms
    plot_dark_histograms(dark_stats, logger=logger, figsize=(14, 10),
                         save_name="dark_histograms",
                         plot_histograms=False,)
    #
    # # 3. Plot the spatial image maps
    plot_hot_pixel_map(dark_stats, logger=logger,
                       save_name="hot_pixel_map",
                       plot_map=False,)

    print_dark_analysis_table(dark_stats, logger=logger,
                              plot_table=False,
                              save_name="dark_analysis_table")

    xMin = 50
    xMax = 150

    yMin = 450
    yMax = 550

    # Automatically picks and graphs frames [0, 166, 333, 499] with rectangles
    means_ref, stds_ref = analyze_and_plot_rect(
        imgs=flat_cor,
        xmin=xMin, xmax=xMax,
        ymin=yMin, ymax=yMax,
        num_plots=4,
        vmax=1.5,
        save_name="means_ref_grid_image",
        plot_graphs=False,
    )

    plot_stats(
        means=means_ref,
        stds=stds_ref,
        title_mean="Reference mean",
        title_std="Reference std",
        save_name="means_std_ref_plot",
        plot_stats=False,
    )



    posun = 52
    posun2 = 45

    yMin2 = 280 + posun2
    yMax2 = 420 + posun2

    xMin2 = 380 + posun
    xMax2 = 520 + posun

    eruption_images = crop_images(flat_cor,
                                  xmin=xMin2,
                                  xmax=xMax2,
                                  ymin=yMin2,
                                  ymax=yMax2,
                                  logger=logger,
                                  )

    (xC, yC) = (69, 70)
    R = 52

    means_circ, std_circ = analyze_and_plot_circ(
        imgs=eruption_images,
        xc=xC, yc=yC, r=R,
        num_plots=4,
        vmax=2,
        save_name="means_eruption_grid_image",
        plot_graphs=False,
    )

    plot_stats(means=means_circ, stds=std_circ,
               title_mean="Eruption region mean",
               title_std="Eruption region std",
               save_name="means_std_eruption_plot",
               plot_stats=False,
               )

    # Step 2: Feed reference arrays into the circle calculation function
    erupting_ratios, total_pixels = calculate_erupting_pixels(
        imgs=eruption_images,
        xc=xC,
        yc=yC,
        r=R,
        mean_ref=means_ref,
        std_ref=stds_ref,
        sigma_level=5.0,  # Uses 5-sigma default thresholding
        logger=logger
    )

    # # X-axis will automatically scale as 0, 1, 2, 3...
    plot_single_series(
        data=erupting_ratios,
        title="Eruption Intensity Per Frame",
        y_label="Active Pixel Ratio",
        logger=logger,
        save_name="eruption_intenzity_plot",
    )

    plot_eruption_contours(
        imgs=eruption_images,
        xc=xC, yc=yC, r=R,
        mean_ref=means_ref,
        std_ref=stds_ref,
        num_plots=12,
        min_sigma=5,  # Maps contours from 1σ out to 5σ
        save_name="eruption_contours_grid.png"  # Exports the map directly to /Plots
    )
