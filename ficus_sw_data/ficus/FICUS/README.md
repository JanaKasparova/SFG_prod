

===============================================================================
directory -- PYTHON:
Python3 code written for the FICUS (2020/21)


-> OCAS_lib.py -- the basic module for manipulating the FICUS hdf5 files.
import it for use. The basic class is called "Measurement". The "Light" class
has methods for subtracting dark frames. "Calibration" class is empty, but can
be expanded if dark measurements require different forms of manipulation.

-> adjacent_correlation.py -- plotting the correlation of adjacent pixel values.
can be used to determine:

a) atmospheric conditions
b) the point spread function
c) the gain of the spectrometers

-> asymetric.py -- plotting the W-E hemispheric difference

-> dispersion.py -- finding the dispersion curve, ie. the relation
of px -> wavelength

-> slitjaw.py -- dealing with SlitJaw images.

-> T_profile.ipynb -- a large Jupyter Notebook; analysing drift measurements
of Limb Darkening. 

-> NormalizationMoodule.py -- calibration using the center of the solar disc, using a
method closer described in appended pdf (FICUS.pdf; Czech)
contains classes:
	a) Normalization
	b) Linearity

-> hd_func.py -- examples of loading the HDF Files without using the
OCAS_lib module. The example shown is drawing vertical lines corresponding to
the Fraunhoffer lines.

-> read_fts.py -- a script for loading the FTS altas as a FITS table: may save
time for someone trying to solve the same problem.

===============================================================================

-> EXAMPLE.ipynb -- detailed explanation of usage of modules



===============================================================================

directory -- useful_files

-> WL_range_C.npy, WL_range_D.npy -- lists of 3840 values of wavelenghts (nm)
corresponding to each pixel in given spectrometer.

-> FTS atlas as a .npy file and a .fts file.

-> Fraunhoffer_lines.txt -- the solar fraunhoffer lines.

-> FICUS.pdf -- (Czech) a description of FICUS and of several observations.

-> resampled_C.npy, resampled_D.npy -- FTS atlas resampled for FICUS pixel bins

===============================================================================

CONTACT: marco.souzadejoode@gmail.com

June, 2021.

