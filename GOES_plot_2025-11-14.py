# import os
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from astropy.time import Time
# import astropy.units as u
# from data_io import *
# from processing import *
# from analysis import *
# from plotting import *
#
# # =====================================================================
# # GLOBAL PIPELINE RUNNER
# # =====================================================================
# if __name__ == "__main__":
#
#     # -----------------------------------------------------------------
#     # CONFIGURATION PARAMETERS (All modifications go here!)
#     # -----------------------------------------------------------------
#     # --- Directory Paths ---
#     DIR_SLITJAW       = "./sfg/2025-11-14-X4.1/sun_area/SlitJaw"
#     DIR_HDF_DATA      = "./sfg/2025-11-14-X4.1/sun_area"
#     DIR_PLOTS         = "Plots/2025-11-14"
#
#     # --- GOES Satellite Parameters ---
#     GOES_SATELLITE    = 16
#     GOES_BUFFER_HOURS = 1.5
#     GOES_CHANNEL      = "xrsb"
#
#     # --- Reference Box Bounds & Metrics ---
#     REF_XMIN, REF_XMAX = 415, 642
#     REF_YMIN, REF_YMAX = 420, 648
#
#     # --- Eruption Crop Box Shifts & Parameters ---
#     ERUPTION_XMIN = 856
#     ERUPTION_XMAX = 995
#     ERUPTION_YMIN = 429
#     ERUPTION_YMAX = 568
#
#     # --- Eruption Circular Mask Region ---
#     ERUPTION_CENTER = (70, 67)  # (xC, yC)
#     ERUPTION_RADIUS = 53  # R
#
#     # --- Spectrometer H-alpha Parameters ---
#     H_ALPHA_CENTER_IDX = 1379
#     H_ALPHA_HALF_WIDTH = 2
#
#     # --- SlitJaw Eruption Region Crop & Circle ---
#     ERUPTION_BOUNDS    = (ERUPTION_XMIN, ERUPTION_XMAX, ERUPTION_YMIN, ERUPTION_YMAX)
#
#     # --- SlitJaw Reference Region Crop & Circle ---
#     REF_BOUNDS         = (REF_XMIN, REF_XMAX, REF_YMIN, REF_YMAX)
#     REF_CENTER         = ERUPTION_CENTER
#     REF_RADIUS         = ERUPTION_RADIUS
#
#     # =====================================================================
#     # STEP 1: EXTRACT TIMESTAMPS & TIMELINE BOUNDARIES
#     # =====================================================================
#     timestamps = compile_directory_timestamps(DIR_SLITJAW)
#     t_start = timestamps[0]
#     t_end = timestamps[-1]
#
#     # Extract hours and minutes from the start time (e.g., '1247')
#     time_suffix = t_start.datetime.strftime("%H%M")
#     print(f"Extracted Time Boundaries (Suffix: {time_suffix}):\n -> Start: {t_start}\n -> End:   {t_end}\n")
#
#     # --- Output Save Filenames ---
#     SAVE_GOES_PLOT     = os.path.join(DIR_PLOTS, "goes_flux_output.png")
#     SAVE_SPECTRUM_PLOT = os.path.join(DIR_PLOTS, f"spectrum_at_{time_suffix}.png")
#     SAVE_H_ALPHA_PLOT  = os.path.join(DIR_PLOTS, "integrated_h_alpha.png")
#     SAVE_SUMMARY_NAME  = os.path.join(DIR_PLOTS, "flare_summary_profile.png")  # Relative to 'Plots/' inside function
#
#     # --- Plot Display Control Flags ---
#     SHOW_INTERMEDIATE_PLOTS = True
#     SHOW_FINAL_SUMMARY      = True
#     # -----------------------------------------------------------------
#
#
#     # =====================================================================
#     # STEP 2: FETCH GOES SATELLITE DATA & COMPUTE GRADIENT
#     # =====================================================================
#     goes_flare, goes_object = get_goes_flux(
#         t_start=t_start,
#         t_end=t_end,
#         filename=SAVE_GOES_PLOT,
#         satellite=GOES_SATELLITE,
#         buffer_hours=GOES_BUFFER_HOURS
#     )
#
#     gradient_array = calculate_goes_gradient(goes_object, channel=GOES_CHANNEL)
#     print(f"GOES Gradient Trace Array Generated! Shape: {gradient_array.shape}\n")
#
#
#     # =====================================================================
#     # STEP 3: MATCH DATASETS AND LOAD SPECTROMETER HDF5 DATA
#     # =====================================================================
#     path_to_large_fileC, path_to_large_fileD = get_hdf_paths(DIR_HDF_DATA)
#     metadata_list = make_metadata_dict(path_to_large_fileD)
#
#     # Decode target index belonging to current SlitJaw time bounds
#     dataset_index = find_matching_hdf5_index(t_start, t_end, metadata_list)
#     print(f"Matched Target HDF5 Dataset Index: {dataset_index}")
#
#     mC, mD = load_hdf_light(path=DIR_HDF_DATA, idx=dataset_index, logger=None)
#
#
#     # =====================================================================
#     # STEP 4: SPECTRUM LOOKUP ANALYSIS & INTEGRATION
#     # =====================================================================
#     # Check baseline snapshot profile at launch time
#     my_target_time = t_start
#     plot_spectrum_at_time(
#         mC=mC,
#         mD=mD,
#         target_time=my_target_time + u.s * 5,
#         save_filename=SAVE_SPECTRUM_PLOT,
#         show_plot=SHOW_INTERMEDIATE_PLOTS
#     )
#
#     # Compile integration track along your H-alpha core profile
#     timerange, h_alpha_integrated = slice_and_calculate_h_alpha(
#         light_obj=mD,
#         t_start=t_start,
#         t_end=t_end,
#         center_idx=H_ALPHA_CENTER_IDX,
#         half_width=H_ALPHA_HALF_WIDTH
#     )
#
#     plot_single_series(
#         data=h_alpha_integrated,
#         time_series=timerange,
#         title="Integrated H-alpha Intensity",
#         num_ticks=15,  # Sets ~7 ticks on the x-axis
#         xlabel="Time",
#         ylabel="Integrated Intensity",
#         save_filename=SAVE_H_ALPHA_PLOT,
#         plot_graph=SHOW_INTERMEDIATE_PLOTS
#     )
#
#
#     # =====================================================================
#     # STEP 5: LOAD SLITJAW FITS IMAGES & COMPUTE CIRCULAR AREA METRICS
#     # =====================================================================
#     raw_imgs = load_fits(DIR_SLITJAW)
#
#     # Track target eruption area
#     values = sum_circle_values(
#         raw_imgs,
#         crop_bounds=ERUPTION_BOUNDS,
#         circle_center=ERUPTION_CENTER,
#         circle_radius=ERUPTION_RADIUS
#     )
#
#     # Track structural background calibration area
#     reference_values = sum_circle_values(
#         raw_imgs,
#         crop_bounds=REF_BOUNDS,
#         circle_center=REF_CENTER,
#         circle_radius=REF_RADIUS
#     )
#
#     # Render background track stability check
#     plot_single_series(
#         data=reference_values,
#         time_series=timestamps.datetime,
#         num_ticks=15,  # Sets ~7 ticks on the x-axis
#         title="REFERENCE AREA INTENSITY",
#         xlabel="Time",
#         ylabel="Intensity",
#         plot_graph=SHOW_INTERMEDIATE_PLOTS
#     )
#
#
#     # =====================================================================
#     # STEP 6: NORMALIZE TRACKS AND SHOW MULTIPANEL CORRELATION OVERVIEW
#     # =====================================================================
#     intensity_fits = values / reference_values
#
#     plot_flare_summary(
#         goes_flare=goes_object,
#         gradient=gradient_array,
#         num_ticks=15,  # Sets ~7 ticks on the x-axis
#         time_series=timerange,
#         h_alpha_data=h_alpha_integrated,
#         time_array=timestamps,
#         intensity=intensity_fits,
#         save_name=SAVE_SUMMARY_NAME,
#         plot_graph=SHOW_FINAL_SUMMARY
#     )

import gc
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import astropy.units as u
from astropy.io import fits

from data_io import *
from processing import *
from analysis import *
from plotting import *

# =====================================================================
# GLOBAL PIPELINE RUNNER
# =====================================================================
if __name__ == "__main__":

    # -----------------------------------------------------------------
    # CONFIGURATION PARAMETERS (All modifications go here!)
    # -----------------------------------------------------------------
    # --- Directory Paths ---
    DIR_SLITJAW = "./sfg/2025-11-14-X4.1/sun_area/SlitJaw"
    DIR_HDF_DATA = "./sfg/2025-11-14-X4.1/sun_area"
    DIR_PLOTS = "Plots/2025-11-14/test"

    # --- Batch Processing Parameters ---
    BATCH_SIZE = 150  # Number of FITS frames loaded into RAM per batch

    # --- GOES Satellite Parameters ---
    GOES_SATELLITE = 18
    GOES_BUFFER_HOURS = 1.5
    GOES_CHANNEL = "xrsb"

    # --- Reference Box Bounds & Metrics ---
    REF_XMIN, REF_XMAX = 219, 418
    REF_YMIN, REF_YMAX = 387, 582

    # --- Eruption Crop Box Shifts & Parameters ---
    ERUPTION_XMIN = 729
    ERUPTION_XMAX = 865
    ERUPTION_YMIN = 396
    ERUPTION_YMAX = 531

    # --- Eruption Circular Mask Region ---
    ERUPTION_CENTER = (67, 68)  # (xC, yC)
    ERUPTION_RADIUS = 53  # R

    # --- Spectrometer H-alpha Parameters ---
    H_ALPHA_CENTER_IDX = 1379
    H_ALPHA_HALF_WIDTH = 2

    # --- SlitJaw Eruption Region Crop & Circle ---
    ERUPTION_BOUNDS = (ERUPTION_XMIN, ERUPTION_XMAX, ERUPTION_YMIN, ERUPTION_YMAX)

    # --- SlitJaw Reference Region Crop & Circle ---
    REF_BOUNDS = (REF_XMIN, REF_XMAX, REF_YMIN, REF_YMAX)
    REF_CENTER = ERUPTION_CENTER
    REF_RADIUS = ERUPTION_RADIUS

    # Create directory if it doesn't exist
    os.makedirs(DIR_PLOTS, exist_ok=True)
    # =====================================================================
    # STEP 1: EXTRACT TIMESTAMPS & TIMELINE BOUNDARIES
    # =====================================================================
    timestamps = compile_directory_timestamps(DIR_SLITJAW)
    t_start = timestamps[0]
    t_end = timestamps[-1]

    # Extract hours and minutes from the start time (e.g., '1247')
    time_suffix = t_start.datetime.strftime("%H%M")
    print(f"Extracted Time Boundaries (Suffix: {time_suffix}):\n -> Start: {t_start}\n -> End:   {t_end}\n")

    # --- Output Save Filenames ---
    SAVE_GOES_PLOT = os.path.join(DIR_PLOTS, "goes_flux_output.png")
    SAVE_SPECTRUM_PLOT = os.path.join(DIR_PLOTS, f"spectrum_at_{time_suffix}.png")
    SAVE_H_ALPHA_PLOT = os.path.join(DIR_PLOTS, "integrated_h_alpha.png")
    SAVE_SUMMARY_NAME = os.path.join(DIR_PLOTS, "flare_summary_profile.png")

    # --- Diagnostic Region Plot Save Paths ---
    SAVE_DIAG_REF_SINGLE = os.path.join(DIR_PLOTS, f"diag_single_reference_{time_suffix}")
    SAVE_DIAG_REF_CROP = os.path.join(DIR_PLOTS, f"diag_crop_reference_{time_suffix}")
    SAVE_DIAG_ERUP_SINGLE = os.path.join(DIR_PLOTS, f"diag_single_eruption_{time_suffix}")
    SAVE_DIAG_ERUP_CROP = os.path.join(DIR_PLOTS, f"diag_crop_eruption_{time_suffix}")

    # --- Plot Display Control Flags ---
    SHOW_INTERMEDIATE_PLOTS = True
    SHOW_FINAL_SUMMARY = True
    # -----------------------------------------------------------------

    # =====================================================================
    # STEP 2: FETCH GOES SATELLITE DATA & COMPUTE GRADIENT
    # =====================================================================
    goes_flare, goes_object = get_goes_flux(
        t_start=t_start,
        t_end=t_end,
        filename=SAVE_GOES_PLOT,
        satellite=GOES_SATELLITE,
        buffer_hours=GOES_BUFFER_HOURS
    )

    gradient_array = calculate_goes_gradient(goes_object, channel=GOES_CHANNEL)
    print(f"GOES Gradient Trace Array Generated! Shape: {gradient_array.shape}\n")

    # =====================================================================
    # STEP 3: MATCH DATASETS AND LOAD SPECTROMETER HDF5 DATA
    # =====================================================================
    path_to_large_fileC, path_to_large_fileD = get_hdf_paths(DIR_HDF_DATA,CD=True)
    metadata_list = make_metadata_dict(path_to_large_fileD)

    # Decode target index belonging to current SlitJaw time bounds
    dataset_index = find_matching_hdf5_index(t_start, t_end, metadata_list)
    print(f"Matched Target HDF5 Dataset Index: {dataset_index}")

    mC, mD = load_hdf_light(path=DIR_HDF_DATA, idx=dataset_index, logger=None)

    # =====================================================================
    # STEP 4: SPECTRUM LOOKUP ANALYSIS & INTEGRATION
    # =====================================================================
    # Check baseline snapshot profile at launch time
    my_target_time = t_start
    plot_spectrum_at_time(
        mC=mC,
        mD=mD,
        target_time=my_target_time + u.s * 5,
        save_filename=SAVE_SPECTRUM_PLOT,
        show_plot=SHOW_INTERMEDIATE_PLOTS
    )

    # Compile integration track along your H-alpha core profile
    timerange, h_alpha_integrated = slice_and_calculate_h_alpha(
        light_obj=mD,
        t_start=t_start,
        t_end=t_end,
        center_idx=H_ALPHA_CENTER_IDX,
        half_width=H_ALPHA_HALF_WIDTH
    )

    plot_single_series(
        data=h_alpha_integrated,
        time_series=timerange,
        title="Integrated H-alpha Intensity",
        num_ticks=15,  # Sets ~7 ticks on the x-axis
        xlabel="Time",
        ylabel="Integrated Intensity",
        save_filename=SAVE_H_ALPHA_PLOT,
        plot_graph=SHOW_INTERMEDIATE_PLOTS
    )

    # =====================================================================
    # STEP 5: REGION DIAGNOSTICS & BATCH METRIC COMPUTATION
    # =====================================================================
    slitjaw_files = get_fits_filepaths(DIR_SLITJAW)
    total_files = len(slitjaw_files)
    if total_files == 0:
        raise FileNotFoundError(f"No SlitJaw FITS files found in {DIR_SLITJAW}")

    # Load representative first frame for crop/region diagnostic previews
    sample_frame = fits.getdata(slitjaw_files[0]).astype(np.float32)

    # ---- 5A. REFERENCE REGION DIAGNOSTICS ----
    plot_single_image(
        sample_frame,
        title="Raw SlitJaw Reference Frame Overview",
        save_name=SAVE_DIAG_REF_SINGLE,
        plot_graph=SHOW_INTERMEDIATE_PLOTS
    )
    plot_image_with_crop(
        sample_frame,
        xmin=REF_XMIN, xmax=REF_XMAX,
        ymin=REF_YMIN, ymax=REF_YMAX,
        title_full="Full SlitJaw Frame (Reference Bounds)",
        title_crop="Cropped Reference Region",
        save_name=SAVE_DIAG_REF_CROP,
        plot_graph=SHOW_INTERMEDIATE_PLOTS
    )

    # ---- 5B. ERUPTION REGION DIAGNOSTICS ----
    plot_single_image(
        sample_frame,
        title="Raw SlitJaw Eruption Frame Overview",
        save_name=SAVE_DIAG_ERUP_SINGLE,
        plot_graph=SHOW_INTERMEDIATE_PLOTS
    )
    plot_image_with_crop(
        sample_frame,
        xmin=ERUPTION_XMIN, xmax=ERUPTION_XMAX,
        ymin=ERUPTION_YMIN, ymax=ERUPTION_YMAX,
        title_full="Full SlitJaw Frame (Eruption Bounds)",
        title_crop="Cropped Eruption Region",
        save_name=SAVE_DIAG_ERUP_CROP,
        plot_graph=SHOW_INTERMEDIATE_PLOTS
    )

    del sample_frame
    gc.collect()

    # ---- 5C. BATCH COMPUTATION OF AREA INTENSITIES ----
    print(f"\n--- Batch processing {total_files} SlitJaw files (Batch size: {BATCH_SIZE}) ---")

    values_list = []
    reference_values_list = []

    for i in range(0, total_files, BATCH_SIZE):
        batch_paths = slitjaw_files[i:i + BATCH_SIZE]

        # Load batch into RAM
        batch_imgs = np.array([fits.getdata(fpath).astype(np.float32) for fpath in batch_paths])

        # Track target eruption area
        batch_erup = sum_circle_values(
            batch_imgs,
            crop_bounds=ERUPTION_BOUNDS,
            circle_center=ERUPTION_CENTER,
            circle_radius=ERUPTION_RADIUS
        )
        values_list.append(batch_erup)

        # Track structural background calibration area
        batch_ref = sum_circle_values(
            batch_imgs,
            crop_bounds=REF_BOUNDS,
            circle_center=REF_CENTER,
            circle_radius=REF_RADIUS
        )
        reference_values_list.append(batch_ref)

        # Free batch buffer immediately
        del batch_imgs, batch_erup, batch_ref
        gc.collect()

    # Combine batch outputs into unified 1D time series
    values = np.concatenate(values_list)
    reference_values = np.concatenate(reference_values_list)

    # Render background track stability check
    plot_single_series(
        data=reference_values,
        time_series=timestamps.datetime,
        num_ticks=15,  # Sets ~7 ticks on the x-axis
        title="REFERENCE AREA INTENSITY",
        xlabel="Time",
        ylabel="Intensity",
        plot_graph=SHOW_INTERMEDIATE_PLOTS
    )

    # =====================================================================
    # STEP 6: NORMALIZE TRACKS AND SHOW MULTIPANEL CORRELATION OVERVIEW
    # =====================================================================
    intensity_fits = values / reference_values

    plot_flare_summary(
        goes_flare=goes_object,
        gradient=gradient_array,
        num_ticks=15,  # Sets ~7 ticks on the x-axis
        time_series=timerange,
        h_alpha_data=h_alpha_integrated,
        time_array=timestamps,
        intensity=intensity_fits,
        save_name=SAVE_SUMMARY_NAME,
        plot_graph=SHOW_FINAL_SUMMARY
    )