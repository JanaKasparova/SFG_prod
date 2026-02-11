import gc

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
        from data_io import load_fits, load_hdf, load_flats, load_darks

        self.logger.info("Loading data")
        self.raw_data["fits"] = load_fits(self.fits_dir, logger=self.logger)
        self.raw_data["hdf"] = load_hdf(self.hdf_dir, logger=self.logger)
        self.raw_data["flats"] = load_flats(self.flat_dir, logger=self.logger)
        self.raw_data["darks"] = load_darks(self.dark_dir, logger=self.logger)

    def process(self):
        from processing import calibrate, fits_to_cv
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
        from analysis import run_analysis

        self.logger.info("Analyzing data")
        self.results = run_analysis(self.calibrated_data)

    def plot(self):
        from plotting import make_plots, plot_image_grid

        self.logger.info("Plotting results")
        make_plots(self.calibrated_data, self.results, self.output_dir)

        # CALIBRATED DATA NO LONGER NEEDED
        self._release_calibrated_data()


    def plot_image_grid(
        self,
        key: str,
        indices=None,
        n_images: int = 6,
        suptitle: str | None = None,
        cmap: str = "gray",
    ) -> None:
        """
        Plot images stored in self.raw_data using a uniform grid.

        Parameters
        ----------
        key : str
            Key in self.raw_data dict (e.g. "fits", "calibrated").

        indices : list[int] | slice | None
            Which images to plot. If None, the first n_images are used.

        n_images : int
            Number of images to plot if indices is None.

        suptitle : str, optional
            Figure title.
        """

        if key not in self.raw_data:
            raise KeyError(f"No data stored under key '{key}'")

        images = self.raw_data[key]

        plot_image_grid(
            images=images,
            indices=indices,
            n_images=n_images,
            suptitle=suptitle,
            cmap=cmap,
        )

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
        self.process()
        self.analyze()
        self.plot()
        self.logger.info("Pipeline finished")
