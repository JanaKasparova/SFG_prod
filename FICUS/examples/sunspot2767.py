import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/marco/Desktop/solar/OCAS_lib')
from OCAS_lib import Light, Calibration, Measurement
from scipy.signal import savgol_filter, find_peaks
import matplotlib.dates as md
from datetime import datetime
from scipy.optimize import curve_fit


def main():
	m1 = Light("/home/marco/Desktop/solar/DATA/2020-07-30/NOAA2767/NOAA2767B_2020-07-30_HR4C5177.hdf", 2)
	darkfile1 = Calibration("/home/marco/Desktop/solar/DATA/2020-07-30/NOAA2767-dark/NOAA2767-dark_2020-07-30_HR4C5177.hdf", -2)

	m2 = Light("/home/marco/Desktop/solar/DATA/2020-07-30/NOAA2767/NOAA2767B_2020-07-30_HR4D290.hdf",2)
	darkfile2 = Calibration("/home/marco/Desktop/solar/DATA/2020-07-30/NOAA2767-dark/NOAA2767-dark_2020-07-30_HR4D290.hdf",2)

	d1 = darkfile1.master
	d2 = darkfile2.master

	m1.subtract_dark(d1)
	m2.subtract_dark(d2)
	

	trans1 = 4*np.sum(m1.data.T, axis=0)/3840
	trans2 = np.sum(m2.data.T, axis=0)/3840



	#spot1 = np.average(m1.data[50:56], axis=0)
	#back1 = np.average(m1.data[80:200], axis=0)

	spot2 = np.average(m2.data[50:56], axis=0)
	center = np.average(m2.data[125:155], axis=0)
	
	
	for i, row in enumerate(m2.data[10:70]):
		X = row/center
		clr = "k"
		txt = "Quiet Sun"
		if i >= 50 and i <= 56:
			clr = "r"
			txt = "Sunspot"
		plt.scatter(m2.wl_range, X, label=txt, c=clr)
		plt.title("Limb darkening at H-Alpha", fontsize=20)
		plt.xlabel("Wavelength (nm)", fontsize=20)
		plt.ylabel("I / I0", fontsize=20)

		
		plt.xlim(645, 665)
		#plt.ylim(min(X[1350:1400])-0.1, max(X[1350:1400])+0.1)
		plt.ylim(0.75, 0.95)		
		plt.savefig(f"HalphaDrift/halpha_{i}.png")
		plt.clf()
			

	"""
	plt.scatter(m1.row_range, trans1, c="b", s=8, label="4 x C spectrometer")
	plt.scatter(m2.row_range, trans2, c="r", s=8, label="D spectrometer")
	plt.title("Drift over sunspor 2767, 2020-07-30", fontsize=20)
	plt.ylabel("Average pixel counts (ADU)", fontsize=20)
	plt.xlabel("Time (UT)", fontsize=20)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	plt.legend(fontsize=20)
	plt.show()
	"""



if __name__ == "__main__":
	main()
