README - How to generate analysis plots
=====================================


## **AI generated summary of the workflow for producing analysis plots from FICUS h-alpha photos (FITS data, flats and darks), spectrometer HDF5 data and GOES x-ray flux.**



Checklist (run in this order)
- Run `find_crop_values_xxx.py` to interactively obtain crop bounds for calibration, reference box and eruption box (and circular mask). Copy the printed values.
- Run `premaking_masters_xxx.py` to create master dark and master flat. Use the included flat "testing" mode to choose which flat frames to include.
- Run `MAKING_PLOTS_xxx.py` (or the date-specific variant) using the crop values from step 1 and the master files from step 2 to produce calibrated data, eruption-reference diagnostics and cache the eruption-area time series.
- Run `GOES_plot_xxx.py` to produce GOES, spectrometer and summary plots that combine the cached eruption-area series with spectrometer and GOES data.

Overview
--------
This repository contains a small, repeatable pipeline for producing analysis plots from SlitJaw FITS data, flats and darks plus spectrometer HDF5 data and GOES. The main scripts you will use are:

- `find_crop_values_xxx.py`  — interactive selection of crop regions (calibration bounds, reference box, eruption box and circular mask)
- `premaking_masters_xxx.py` — compute and save master dark and master flat (includes a `testing_flat` helper to inspect flat intensities)
- `MAKING_PLOTS_xxx.py`      — calibration pipeline, reference/eruption analysis and caching of the eruption-area time series
- `GOES_plot_xxx.py`         — GOES flux, spectrometer H-alpha traces and combined multi-panel flare summary

1) Find crop values (interactive)
---------------------------------
Run the `find_crop_values_xxx.py` script in an environment that supports interactive figure selection (X11 / Jupyter / any GUI-capable environment). The script will:

- Load a sampled subset of flat frames to choose the primary calibration crop (CALIB).
- Load a sampled subset of the science stack and let you choose a `REF` rectangle and an `ERUPTION` rectangle.
- Crop the eruption rectangle and let you select a circular mask within it (center + radius).

When selection completes the script prints a consolidated summary you can copy, for example:

CALIB_XMIN, CALIB_XMAX = 249, 1094
CALIB_YMIN, CALIB_YMAX = 115, 897

REF_XMIN, REF_XMAX = 55, 214
REF_YMIN, REF_YMAX = 108, 259

ERUPTION_XMIN = 488
ERUPTION_XMAX = 619
ERUPTION_YMIN = 261
ERUPTION_YMAX = 391

ERUPTION_CENTER = (65, 66)
ERUPTION_RADIUS = 53

Notes and tips:
- The script prints instructions at start: it expects an interactive environment. If you want to run headless, set the `MANUAL_*` variables at the top of the script with your values.
- For `Making plots` (analysis pipeline) choose a smaller region inside the calibration bounds — this is the `CALIB` area used for the full processing crop.
- For `GOES`-style plots (slitjaw-only diagnostics) choose a larger bounding box so that both the reference area and the eruption area are fully contained. The GOES script expects separate REF and ERUPTION bounding values.

Run (example):

```fish
python3 find_crop_values_xxx.py
```

2) Make master dark and master flat (premaking masters)
---------------------------------------------------
Use `premaking_masters_xxx.py` to build calibration masters used by the main pipeline. Key points:

- Edit the top configuration block to point `FLAT_FOLDER` and `DARK_FOLDER` to your flat/dark folders (or leave as-is if paths are correct).
- `RUN_FLAT_TESTING` toggles running `testing_flat(...)` which will plot per-frame flat intensities to help choose which frames to include.
- `PROCESS_MASTER_DARK` and `PROCESS_MASTER_FLAT` control whether to compute or load existing masters.
- If you need to drop a portion of the flat sequence (e.g. shutter problem in the middle of the run), use `EXCLUDE_FLAT_IDX = (start, end)` to exclude frames before creating the master flat.

Outputs:
- Master files are written into `MASTER_SAVE/<date>/master_flat.fits` and `master_dark.fits`.
- Diagnostic plots and animation (if enabled) are written into the same `MASTER_SAVE/<date>/` folder.

Run (example):

```fish
python3 premaking_masters_xxx.py
```

How to choose flat frames
- Run with `RUN_FLAT_TESTING = True` and inspect the intensity plots printed by `testing_flat`. This shows the raw intensity vs frame index and helps locate bad frames or drifts. Then set `EXCLUDE_FLAT_IDX` or adjust the `FLAT_IDX` tuple accordingly.

3) Produce calibrated data, eruption & reference analysis (MAKING_PLOTS)
----------------------------------------------------------------------
Open `MAKING_PLOTS_xxx.py` and update the calibration and region variables near the top (copy the printed numbers from step 1):

- `CALIB_XMIN, CALIB_XMAX, CALIB_YMIN, CALIB_YMAX` — calibration crop for full pipeline
- `REF_XMIN/REF_XMAX`, `REF_YMIN/REF_YMAX` — reference rectangle
- `ERUPTION_XMIN/ERUPTION_XMAX`, `ERUPTION_YMIN/ERUPTION_YMAX` — eruption bounding box
- `ERUPTION_CENTER` & `ERUPTION_RADIUS` — circular mask used by the eruption analysis

Important script behaviour:
- `MAKING_PLOTS_xxx.py` will look for master files at `MASTER_SAVE/<date>/master_flat.fits` and `master_dark.fits` by default — make sure the date suffix matches the masters created in step 2.
- The script will create a disk-backed cache under `CACHE_DATA/<date>/` containing `final_flat_corrected_data.npy` (memory-mapped calibrated stack) and `cropped_dark.npy`. It will also save `normalized_erupting_pixels.npy` (used by the GOES plot step).
- Note that the eruption bounding box should not be much larger than the reference rectangle, reference value is calculated from the reference rectangle within a circle of the same coordinates `ERUPTION_CENTER` & `ERUPTION_RADIUS` which are defined with respect to `ERUPTION_XMIN/ERUPTION_XMAX`, `ERUPTION_YMIN/ERUPTION_YMAX` but used also with respect to `REF_XMIN/REF_XMAX`, `REF_YMIN/REF_YMAX`, see GOES_plot below.

Run (example):

```fish
python3 MAKING_PLOTS_xxx.py
```

Notes:
- Tweak `PROC_BATCH_SIZE` to fit available RAM. Use smaller values for systems with less memory.
- Set `SHOW_DIAGNOSTIC_PLOTS = False` for headless runs to avoid opening many figures.

4) Run GOES & spectrometer combined summary (GOES_plot)
-----------------------------------------------------
`GOES_plot_xxx.py` expects the cached eruption-area time series produced by `MAKING_PLOTS_xxx.py` in `CACHE_DATA/<date>/normalized_erupting_pixels.npy`.

Primary steps the GOES script performs:
- Fetch GOES flux data for the time-range of the SlitJaw frames.
- Load spectrometer HDF5 data and compute H-alpha core + continuum time series.
- Compute reference and eruption area intensities from the SlitJaw FITS files (the script independently computes these from raw FITS, so ensure REF and ERUPTION bounds are set correctly).
- Normalize spectrometer time series by the SlitJaw reference values and combine everything into a multi-panel `flare_summary_profile.png`.

Run (example):

```fish
python3 GOES_plot_xxx.py
```

Troubleshooting & tips
----------------------
- If a script complains about missing files, double-check the folder variables at the top of each script (e.g., `FITS_DATA_FOLDER`, `FLAT_FOLDER`, `MASTER_SAVE` paths and date suffix).
- Interactive selection requires a GUI backend. If you are on a headless server, either set `MANUAL_*` values in `find_crop_values_2026-07-10.py` or run locally where a display is available.
- If processing is slow or memory errors occur, reduce `BATCH_SIZE`/`PROC_BATCH_SIZE` across the scripts.
- Caches are stored in `CACHE_DATA/<date>/`. Remove these if you want to force recomputation.
- Diagnostic plots are stored under `Plots/<date>/`. Masters are in `MASTER_SAVE/<date>/`.

Automating non-interactive runs
-------------------------------
To run everything non-interactively:

1. Manually edit `find_crop_values_xxx.py` and set `MANUAL_CALIB`, `MANUAL_REF`, `MANUAL_ERUPTION` and `MANUAL_CIRCLE` at the top to the values you want.
2. Run `premaking_masters_xxx.py` (ensure `PROCESS_MASTER_*` flags are set appropriately).
3. Edit `MAKING_PLOTS_xxx.py` and paste in the printed crop values.
4. Run `MAKING_PLOTS_xxx.py` then `GOES_plot_xxx.py`.

Environment & dependencies
--------------------------
Install Python dependencies listed in `requirements.txt` before running:

```fish
python3 -m pip install -r requirements.txt
```

