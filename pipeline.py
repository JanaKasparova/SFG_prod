import gc
from data_io import *
from processing import *
from analysis import *
from plotting import *
from FICUS.PYTHON.OCAS_lib import Light, Calibration, Measurement
from FICUS.PYTHON.NormalizationModule import Normalization, Linearity
import numpy as np

class Pipeline:
    def __init__(self, fits_dir, hdf_dir, flat_dir, dark_dir, output_dir, logger):
        self.fits_dir = fits_dir
        self.hdf_dir = hdf_dir
        self.flat_dir = flat_dir
        self.dark_dir = dark_dir
        self.output_dir = output_dir
        self.logger = logger
        
        self.raw_data = {}
        self.calibrated_data = {}
        self.results = {}


    def load(self):
        self.logger.info("Loading wavelengths")
        self.wlc = np.load("./FICUS/useful_files/WL_range_C.npy")
        self.wld = np.load("./FICUS/useful_files/WL_range_D.npy")
        self.logger.info("Loading data")
        self.raw_data["fits"] = load_fits(self.fits_dir, logger=self.logger)
        self.raw_data["hdf"] = load_hdf(self.hdf_dir, logger=self.logger)
        self.raw_data["flats"] = load_flats(self.flat_dir, logger=self.logger)
        self.raw_data["darks"] = load_darks(self.dark_dir, logger=self.logger)

    def process(self):
        self.logger.info("Processing data")
        self.calibrated_data = calibrate(
            self.raw_data["fits"],
            self.raw_data["flats"],
            self.raw_data["hdf"],
            self.raw_data["darks"],
        )

        # RAW DATA NO LONGER NEEDED → FREE IT
        self._release_raw_data()

    def analyze(self):
        self.logger.info("Analyzing data")
        self.results = run_analysis(self.calibrated_data)

    def plot(self):
        self.logger.info("Plotting results")
        make_plots(self.calibrated_data, self.results, self.output_dir)

        # CALIBRATED DATA NO LONGER NEEDED
        self._release_calibrated_data()


    # --------------------------------------------------
    # Memory management helpers
    # --------------------------------------------------

    def _release_raw_data(self):
        self.logger.info("Releasing raw data from memory")

        self._close_resources(self.raw_data)
        self.raw_data.clear()

        gc.collect()

    def _release_calibrated_data(self):
        self.logger.info("Releasing calibrated data from memory")

        self.calibrated_data.clear()
        gc.collect()

    def _close_resources(self, data_dict):
        """
        Close any file-backed objects (HDF5, FITS, etc.)
        """
        for obj in data_dict.values():
            try:
                obj.close()
            except Exception:
                pass

    def run(self):
        self.logger.info("Pipeline started")
        self.load()
        plot_image_grid(self.raw_data["fits"])
        self.process()
        self.analyze()
        self.plot()
        self.logger.info("Pipeline finished")
