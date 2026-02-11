import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/marco/Desktop/solar/OCAS_lib')
from OCAS_lib import Light, Calibration, Measurement
from scipy.signal import savgol_filter, find_peaks
import matplotlib.dates as md
from datetime import datetime
from scipy.optimize import curve_fit
from matplotlib import cm
from Normalization import Normalization
import matplotlib.patches as mpatches




"""
Used for plotting the correlation of adjacent pixel values.
can be used to determine:

a) atmospheric conditions
b) the point spread function
c) gain of spectrometers
"""



def hyp(x,a,b):
	return a/x + b


def main(m,d,K,let):
	
	# Dark correction
	masterdark = d.master
	m.subtract_dark(masterdark)

	# data and transposed data
	DAT = m.data
	TR = m.data.T


	


	# stdev along the 0th axis,
	# ie. the stdev of ADU values on a given
	# pixel for the entire time series
	devs = np.std(DAT, axis=0)

	# average temporal value of counts on given pixel
	I = np.average(DAT,axis=0)

	# calculating the GAIN, as sigma**2 / avg
	g = (devs)**2/np.average(DAT,axis=0)

	# What is the relationship between intensity and gain? 
	# should be constant, but it is not.
	params, cov = curve_fit(hyp, I, g)

	print(m.dt)
	print(m.data.shape)
	print(params)
	

	# calculating the correlation of pixel values in window of
	# given size.
	def windcors(center, win):
		cors = []
		A = TR[center] - np.average(TR[center])
		for w in range(-win, win):
			B = TR[center+w] - np.average(TR[center+w])
			cor = np.correlate(A, B) / np.correlate(A,A)
			cors.append(cor)
		return np.array(cors)

	def plotcors(center, win):
		cors = windcors(center, win)
		wls = np.linspace(center-win, center+win, 2*win)
		plt.scatter(wls, cors)
		plt.hlines(0,center-win, center+win)
		plt.show()


	def maxpeakdeltas(center, win):
		cors = windcors(center, win)
		cors[win] = np.nan
		return np.argmax(cors) - win

	
	cormat = []
	DELTAS = []



	# choose a smaller W (window) value for closer
	# inspection of the PSP (W approx 20), and a larger
	# (W > 100) value for inspection of the atmospheric
	# conditions.
	
	W = 500
	for i in range(W+1,3600-W-1, 1):
		cormat.append(windcors(i,W))
	

	cormat = np.array(cormat)
	corav = np.average(cormat, axis=0)
		
	# Displaying the results
	
	plt.title(f"Correlation of adjacent pixel values ({let})", fontsize=20)
	plt.scatter(np.linspace(-W,W,2*W,endpoint=False), np.log10(corav), c="k", s=50)
	plt.plot(np.linspace(-W,W,2*W,endpoint=False), np.log10(corav), color="k",linestyle="dashed", linewidth=1)
	plt.xlabel("Pixels from center value", fontsize=20)
	plt.ylabel("log(Normalised correlation)", fontsize=20)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	plt.show()

			

if __name__ == "__main__":
	m = Light("/home/marco/Desktop/solar/DATA/2020-07-01/stred-h-1/stred-h-1_2020-07-01_HR4C5177.hdf",-2)
	d = Calibration("/home/marco/Desktop/solar/DATA/2020-07-01/stred-dark-h-1/stred-dark-h-1_2020-07-01_HR4C5177.hdf",-2)
	K = "b"
	main(m,d,K,"HR4C5177")
	m = Light("/home/marco/Desktop/solar/DATA/2020-07-01/stred-h-1/stred-h-1_2020-07-01_HR4D290.hdf",-2)
	d = Calibration("/home/marco/Desktop/solar/DATA/2020-07-01/stred-dark-h-1/stred-dark-h-1_2020-07-01_HR4D290.hdf",-2)
	K = "r"
	main(m,d,K,"HR4D290")
	black_patch = mpatches.Patch(color='b', label="HR4C5177")
	red_patch = mpatches.Patch(color='r', label="HR4D290")
	plt.legend(handles=[black_patch, red_patch], fontsize=22)
	
