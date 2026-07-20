from data_io import *
from processing import *
from analysis import *
from plotting import *

# --- Example Usage ---
if __name__ == "__main__":
    # Step 1: Run your compiler to get the timestamps from your directory
    directory_path = "./2024-07-29/sun_area/SlitJaw"
    timestamps = compile_directory_timestamps(directory_path)
    t_start = timestamps[0]
    t_end = timestamps[-1]

    # Verify the timestamps look correct
    print(f"Extracted Start: {t_start}")
    print(f"Extracted End: {t_end}")

    # 2. Run the GOES fetcher and specify where to save the image
    output_plot_path = "Plots/goes_flux_output.png"

    # Funkcia teraz vracia DVA objekty (rozbalíme ich pomocou čiarky)
    goes_flare, goes_object = get_goes_flux(
        t_start=t_start,
        t_end=t_end,
        filename=output_plot_path,
        satellite=16,
        buffer_hours=1.5
    )

    # Step 3: Výpočet gradientu zaslaním GOES objektu ako vstupu
    gradient_array = calculate_goes_gradient(goes_object, channel="xrsb")

    # Overenie, že máme čisté numpy pole
    print(f"Gradient calculated from GOES object!")
    print(f"output type: {type(gradient_array)}")
    print(f"Array shape: {gradient_array.shape}")

    import json

    # "./2024-07-29/sun_area/sun_area_2024-07-29_HR4D290.hdf"

    path_hdf = "./2024-07-29/sun_area"
    # 1. Extract the metadata list
    path_to_large_fileC, path_to_large_fileD = get_hdf_paths(path_hdf)  # Replace with your actual path
    metadata_list = make_metadata_dict(path_to_large_fileD)

    # 2. Pretty-print the entire metadata structure
    # print(json.dumps(metadata_list, indent=4, default=str))

    # Step 3: Decode which index belongs to your slitjaw measurement
    dataset_index = find_matching_hdf5_index(t_start, t_end, metadata_list)

    # Now dataset_index has your target index (e.g., 18)
    print(f"Index to read from HDF5 file: {dataset_index}")

    mC, mD = load_hdf_light(path=path_hdf, idx=dataset_index, logger=None)
    # wlc, wld = load_WL_spectrum()

    # 2. Define your desired target timestamp
    my_target_time = t_start

    # # Example A: Plot to screen and don't save
    # plot_spectrum_at_time(
    #     mC=mC,
    #     mD=mD,
    #     target_time=my_target_time,
    #     show_plot=True
    # )

    # Example B: Save to a file and suppress displaying it on screen
    plot_spectrum_at_time(
        mC=mC,
        mD=mD,
        target_time=my_target_time,
        save_filename="Plots/spectrum_at_1247.png",
        show_plot=False
    )

    # 3. Call the slice function
    timerange, h_alpha_integrated = slice_and_calculate_h_alpha(
        light_obj=mD,
        t_start=t_start,
        t_end=t_end,
        center_idx=1379,  # Optional: defaults to 1379
        half_width=2  # Optional: defaults to 2 (range of 5 pixels: 1377-1381)
    )

    print(f"Shape of integrated H-alpha intensity profile: {h_alpha_integrated.shape}")  # e.g., (N_times,)

    plot_single_series(
        data=h_alpha_integrated,
        x_series=timerange,
        title="Integrated H-alpha Intensity",
        xlabel="Time",
        ylabel="Integrated Intensity",
        save_filename="Plots/integrated_h_alpha.png",
        plot_graph=False
    )

    # =====================================================================
    # 1. SETUP PATHS AND LOAD DATA
    # =====================================================================

    # Load raw image stack using your function
    raw_imgs = load_fits(directory_path)

    # =====================================================================
    # 2. CALCULATE INTENSITIES WITH NEW PARAMETERS
    # =====================================================================
    # Eruption values (uses default parameters: crop_bounds=(930, 1070, 430, 570))
    values = sum_circle_values(raw_imgs, (930, 1070, 430, 570),
                               circle_center=(70, 64),
                               circle_radius=54)

    # Reference values (uses specified background crop bounds)
    reference_values = sum_circle_values(raw_imgs, crop_bounds=(1030, 1170, 700, 840),
                                         circle_center=(70, 64),
                                         circle_radius=54)

    # =====================================================================
    # 3. PLOT SINGLE SERIES (FIXING THE TIME CRASH)
    # =====================================================================
    # Astropy Time objects use .datetime to expose standard plottable timestamps
    plot_single_series(
        data=reference_values,
        x_series=timestamps.datetime,  # Converted cleanly to avoid TypeError
        title="REFERENCE AREA INTENSITY",
        xlabel="Time",
        ylabel="Intensity",
        plot_graph=True
    )

    intensity_fits = values / reference_values

    # 2. Pass everything into the function
    plot_flare_summary(
        goes_flare=goes_object,
        gradient=gradient_array,  # Passed as an explicit parameter
        timerange=timerange,
        h_alpha_data=h_alpha_integrated,
        time_array=timestamps,
        intensity=intensity_fits,
        save_name="flare_summary_profile.png",
        plot_graph=True
    )
