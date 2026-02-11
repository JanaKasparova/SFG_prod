#example of usage based on Marco Souza de Joode library and
#EXAMPLE.ipynb
import matplotlib.pyplot as plt
import numpy as np
import sys, inspect
# importing FICUS-specific modules
# OCAS_lib for dealing with hdf files,
# Linearity for the correction of CCD non-lineariry
# and Normalization for the calibration of data using
# the center of the solar disc

sys.path.append('./FICUS/PYTHON')
from OCAS_lib import Light, Calibration, Measurement
from NormalizationModule import Normalization, Linearity

# loading calculated px -> wavelength relations

wlc = np.load("./FICUS/useful_files/WL_range_C.npy")
wld = np.load("./FICUS/useful_files/WL_range_D.npy")

# loading resampled FTS atlases
# aC[0] ... wavelengths
# aC[1] ... I_\nu values

aC = np.load("./FICUS/useful_files/resampled_C.npy")
aD = np.load("./FICUS/useful_files/resampled_D.npy")

plt.plot(aC[0],aC[1])
plt.xlabel('vln.delka [nm]')
plt.ylabel('intensita [erg/s/cm2/ster/Hz]')
plt.show()

mC = Light("./obs/center_2020-09-24_HR4C5177.hdf", -2)
print(mC.data.shape)
plt.plot(mC.data[0])
plt.xlabel('pixel')
plt.ylabel('mereni [data units]')
plt.show()
