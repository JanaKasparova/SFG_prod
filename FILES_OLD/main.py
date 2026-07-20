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


def main():
    logger = setup_logger()

    pipeline = Pipeline(
        fits_dir="2024-07-29/sun_area/SlitJaw",
        hdf_dir="data/hdf",
        flat_dir="data/flats",
        dark_dir="data/darks",
        output_dir="output",
        logger=logger,
    )

    pipeline.run()

if __name__ == "__main__":
    main()
