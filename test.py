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
import logging
from colorlog import ColoredFormatter

def setup_logger():
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()

    formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s | %(levelname)s | %(message)s",
        log_colors={
            "DEBUG": "green",     # inside functions
            "INFO": "blue",       # pipeline-level info
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = setup_logger()

# Define crop bounds
xmn = 900
ymn = 800
x_min, x_max = xmn , xmn +140
y_min, y_max = ymn, ymn +140


# Define circle parameters
circle_center = (70, 64)  # x, y in pixel coordinates of cropped_data
# radius in pixels
circle_radius = 54


# Add rectangle to indicate crop region
rect = Rectangle(
    (x_min, y_min),          # lower-left corner (x, y)
    x_max - x_min,           # width
    y_max - y_min,           # height
    edgecolor='red',
    facecolor='none',
    linewidth=2
)
# flat img setup
# folder = "./2024-07-29/sun_area_flat2/SlitJaw"
folder = "./sun_area_flat/SlitJaw"
fits_files = load_fits(folder, logger=logger)
cropped_images_array = crop_images(fits_files, xmin=x_min, xmax=x_max, ymin=y_min, ymax=y_max, logger=logger)
average_img_val = average_numpy_array(cropped_images_array, axis=(1,2), logger=logger)
images_array_used = fits_files[:80, :, :]
print("Shape of stacked array with usable images:", images_array_used.shape)
average_flat = average_numpy_array(images_array_used, axis=0, logger=logger)
# save_fits_image(average_img, "average_flat.fits", logger=logger)
# img1 = load_fits_img("./2024-07-29/master_flat_full.fits", logger=logger)

c_y_min, c_y_max = 100, 1000
c_x_min, c_x_max = 500, 1500

cropped_average_flat = crop_images(average_flat, xmin=c_x_min, xmax=c_x_max, ymin=c_y_min, ymax=c_y_max, logger=logger)
# cropped_average_flat_img1 = crop_images(img1, xmin=c_x_min, xmax=c_x_max, ymin=c_y_min, ymax=c_y_max, logger=logger)

# # draw rectangle on average image
# fig, ax = plt.subplots(figsize=(15, 12))
#
# ax.imshow(average_flat, origin='lower')
# rect = Rectangle(
#     (c_x_min, c_y_min),          # lower-left corner (x, y)
#     c_x_max - c_x_min,           # width
#     c_y_max - c_y_min,           # height
#     edgecolor='red',
#     facecolor='none',
#     linewidth=2
# )
# ax.add_patch(rect)
# ax.set_title('Average Flat Image with Crop Rectangle')
# plt.show()



#dark img setup
folder2 = "./2024-07-29/sun_area_dark/SlitJaw"

fits_dark = load_fits(folder2, logger=logger)
average_dark = average_numpy_array(fits_dark, axis=0, logger=logger)
cropped_average_dark = crop_images(average_dark, xmin=c_x_min, xmax=c_x_max, ymin=c_y_min, ymax=c_y_max, logger=logger)
# img2 = load_fits_img("./2024-07-29/sun_area_dark/master_dark_slitjaw.fits", logger=logger)

#
# fig, ax = plt.subplots(1, 1, figsize=(10, 8))
# ax.imshow(average_dark, vmax=1000, cmap="gray")
# ax.set_title('Master Dark Image')
# plt.show()
#
# fig, ax = plt.subplots(1, 1, figsize=(10, 8))
# ax.imshow(img2, vmax=1000, cmap="gray")
# ax.set_title('Master Dark Image')
# plt.show()
#

# plot_image_grid(fits_files, n_images=6)
#
# # Call the function
# plot_image_with_crop(
#     image=average_flat,
#     xmin=c_x_min,
#     xmax=c_x_max,
#     ymin=c_y_min,
#     ymax=c_y_max,
#     title_full="Average flat",
#     title_crop="Average cropped flat",
#     logger=logger,cmap="Blues"
# )
wlc, wld = load_WL_spectrum()

m_x = 0
m_y = 0
# data img import
folder3 = "./2024-07-29/sun_area/SlitJaw"
fits_data = load_fits(folder3, logger=logger)
data_cropped = crop_images(fits_data, xmin=c_x_min+m_x, xmax=c_x_max+m_x, ymin=c_y_min+m_y, ymax=c_y_max+m_y, logger=logger)

# plots image selection

plot_image_grid(
    images=data_cropped, n_images=4
)
dark_cor = correct_dark(data_cropped, cropped_average_dark, batch_size=100, align_dark=False, logger=logger)
plot_image_grid(
    images=dark_cor, n_images=4
)
flat_cor = correct_flat(data_cropped, cropped_average_flat, align_flat=True, batch_size=100, logger=logger)
plot_image_grid(
    images= flat_cor, n_images=4
)

# 1. Run the math ONCE
dark_stats = analyze_dark_frame(cropped_average_dark, logger=logger)

# # 2. Plot the histograms
plot_dark_histograms(dark_stats, logger=logger, figsize=(14, 10))
#
# # 3. Plot the spatial image maps
plot_hot_pixel_map(dark_stats, logger=logger)

print_dark_analysis_table(
    dark_stats, logger=logger, plot_table=False)

xMin = 50
xMax = 150

yMin = 450
yMax = 550

# Automatically picks and graphs frames [0, 166, 333, 499] with rectangles
means_ref, stds_ref = analyze_and_plot_rect(
    imgs=flat_cor,
    xmin=xMin, xmax=xMax,
    ymin=yMin, ymax=yMax,
    num_plots=None,
)

plot_stats(
    means=means_ref,
    stds=stds_ref,
    x_label_custom="Elapsed Time (seconds)",

)

posun = 52
posun2 = 45

yMin2 = 280+posun2
yMax2 = 420+posun2

xMin2 = 380+posun
xMax2 = 520+posun

eruption_images = crop_images(flat_cor,
                              xmin=xMin2,
                              xmax=xMax2,
                              ymin=yMin2,
                              ymax=yMax2,
                              logger=logger)

(xC, yC) = (69, 70)
R = 52

means_circ, std_circ = analyze_and_plot_circ(
    imgs=eruption_images,
    xc=xC, yc=yC, r=R,
    num_plots=None
)
#
# plot_stats(
#     means=means_circ,
#     stds=std_circ
# )

# Step 2: Feed reference arrays into the circle calculation function
erupting_ratios, total_pixels = calculate_erupting_pixels(
    imgs=eruption_images,
    xc=69,
    yc=70,
    r=52,
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
    logger=logger
)

# --- 3. Plot Dynamic Spatial Contours (Visualizing Heat Distributions) ---
# This will uniformly grab 4 frames across your timeline and overlay the sigma levels
plot_eruption_contours(
    imgs=eruption_images,
    xc=69, yc=70, r=52,
    mean_ref=means_ref,
    std_ref=stds_ref,
    num_plots=4,                            # Generates a clean 2x2 layout grid
    min_sigma=5,                            # Maps contours from 5σ out to the frame's peak
    # save_name="eruption_contours_grid.png"  # Exports the map directly to /Plots
)

# Pass your stack directly. It slices the last 100 frames out and flattens the rest.
hist_metrics = plot_eruption_histogram(
    imgs=eruption_images,
)
print((hist_metrics))
