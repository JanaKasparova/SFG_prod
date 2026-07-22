from plotting import *
from processing import *
from data_io import *


# =====================================================================
# STANDALONE EXECUTION RUNNER
# =====================================================================
if __name__ == "__main__":
    logger = setup_logger()

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("MUST BE RUN IN INTERACTIVE ENVIRONMENT")
    print("RUNNING FROM COMMAND LINE IS FINE")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    # -----------------------------------------------------------------
    # FILE PATHS & MANUAL OVERRIDES
    # (Set to None to use interactive GUI selection, or enter values to override)
    # -----------------------------------------------------------------
    FITS_DATA_FOLDER = "./sfg/2025-11-14-X4.1/sun_area/SlitJaw"
    FLAT_FOLDER = "./sfg/2025-11-14-X4.1/sun_centre/SlitJaw" # Folder with flat field FITS files

    # Optional manual overrides
    MANUAL_CALIB = None  # e.g., (500, 1500, 100, 1000) -> (xmin, xmax, ymin, ymax)
    MANUAL_REF = None  # e.g., (50, 150, 450, 550)   -> (xmin, xmax, ymin, ymax)
    MANUAL_ERUPTION = None  # e.g., (380, 520, 280, 420)   -> (xmin, xmax, ymin, ymax)
    MANUAL_CIRCLE = None  # e.g., ((69, 70), 52)         -> ((xc, yc), r)

    # -----------------------------------------------------------------
    # STEP 1: Select Core Calibration Bounds on Sampled Flat Fields (50 Frames)
    # -----------------------------------------------------------------
    print("\n--- STEP 1: Select Crop Bounds on Sampled Flat Field Images ---")
    if MANUAL_CALIB:
        cal_xmin, cal_xmax, cal_ymin, cal_ymax = MANUAL_CALIB
    else:
        # Load up to 50 uniformly sampled flat frames for interactive selection
        sampled_flats = load_sampled_fits(FLAT_FOLDER, max_samples=50, logger=logger)

        cal_xmin, cal_xmax, cal_ymin, cal_ymax = select_rectangle_roi(
            sampled_flats, title="Core Calibration Crop (CALIB - Sampled Flats)", var_prefix="CALIB"
        )

        # Immediate memory cleanup
        del sampled_flats
        gc.collect()

    # -----------------------------------------------------------------
    # STEP 2: Uniformly Sample Science Data Stack & Crop to Calibration Region
    # -----------------------------------------------------------------
    print("\n--- STEP 2: Sample Science Data Stack & Crop to Calibration Region ---")

    # Load uniformly sampled subset (max 400 frames) instead of entire dataset
    sampled_stack = load_sampled_fits(FITS_DATA_FOLDER, max_samples=400, logger=logger)

    # Crop sampled stack by primary calibration bounds
    calib_stack = crop_images(
        sampled_stack,
        xmin=cal_xmin, xmax=cal_xmax,
        ymin=cal_ymin, ymax=cal_ymax,
        logger=logger
    )

    # Free raw sampled stack memory
    del sampled_stack
    gc.collect()

    # -----------------------------------------------------------------
    # STEP 3: Select Reference Box (on Calibrated/Cropped Stack)
    # -----------------------------------------------------------------
    print("\n--- STEP 3: Select Reference Region Box ---")
    if MANUAL_REF:
        ref_xmin, ref_xmax, ref_ymin, ref_ymax = MANUAL_REF
    else:
        ref_xmin, ref_xmax, ref_ymin, ref_ymax = select_rectangle_roi(
            calib_stack, title="Reference Region Selector (REF)", var_prefix="REF"
        )

    # -----------------------------------------------------------------
    # STEP 4: Select Eruption Region Box & Crop to Eruption Stack
    # -----------------------------------------------------------------
    print("\n--- STEP 4: Select Eruption Region Crop Box ---")
    if MANUAL_ERUPTION:
        er_xmin, er_xmax, er_ymin, er_ymax = MANUAL_ERUPTION
    else:
        er_xmin, er_xmax, er_ymin, er_ymax = select_rectangle_roi(
            calib_stack, title="Eruption Region Crop Selector (ERUPTION)", var_prefix="ERUPTION"
        )

    print("\n--- Cropping Stack to Eruption Region ---")
    eruption_stack = crop_images(
        calib_stack,
        xmin=er_xmin, xmax=er_xmax,
        ymin=er_ymin, ymax=er_ymax,
        logger=logger
    )

    # -----------------------------------------------------------------
    # STEP 5: Select Circular Eruption Mask (on Eruption-Cropped Stack)
    # -----------------------------------------------------------------
    print("\n--- STEP 5: Select Circular Eruption Mask ---")
    if MANUAL_CIRCLE:
        er_center, er_radius = MANUAL_CIRCLE
    else:
        er_center, er_radius = select_circle_roi(
            eruption_stack,
            title="Eruption Circle Mask Selector"
        )

    # Clean up selection arrays
    del calib_stack, eruption_stack
    gc.collect()

    # =====================================================================
    # FINAL CONSOLIDATED SUMMARY PRINT FOR EASY COPY-PASTING
    # =====================================================================
    print("\n" + "=" * 65)
    print("🚀 COMPLETE PIPELINE CONFIGURATION PARAMETERS 🚀")
    print("=" * 65)
    print("# --- Core Calibration Crop Bounds ---")
    print(f"CALIB_XMIN, CALIB_XMAX = {cal_xmin}, {cal_xmax}")
    print(f"CALIB_YMIN, CALIB_YMAX = {cal_ymin}, {cal_ymax}")
    print()
    print("# --- Reference Box Bounds & Metrics ---")
    print(f"REF_XMIN, REF_XMAX = {ref_xmin}, {ref_xmax}")
    print(f"REF_YMIN, REF_YMAX = {ref_ymin}, {ref_ymax}")
    print()
    print("# --- Eruption Crop Box Shifts & Parameters ---")
    print(f"ERUPTION_XMIN = {er_xmin}")
    print(f"ERUPTION_XMAX = {er_xmax}")
    print(f"ERUPTION_YMIN = {er_ymin}")
    print(f"ERUPTION_YMAX = {er_ymax}")
    print()
    print("# --- Eruption Circular Mask Region ---")
    print(f"ERUPTION_CENTER = ({er_center[0]}, {er_center[1]})  # (xC, yC)")
    print(f"ERUPTION_RADIUS = {er_radius}  # R")
    print("=" * 65 + "\n")