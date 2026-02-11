import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/marco/Desktop/solar/OCAS_lib')
from OCAS_lib import Light, Calibration, Measurement
from scipy.signal import savgol_filter, find_peaks
import matplotlib.dates as md
from datetime import datetime
from scipy.optimize import curve_fit


# Find center of the solar disc as the median of a sorted list of values


def center(y, N):
	ymaxlist = np.argsort(y)[-N:-1]
	xmax = int(np.median(ymaxlist))
	return xmax
	
def flip(rowrange, xmax):
	new_rowrange = abs(rowrange - xmax)
	return new_trange



# separating the E-W hemispheres


def halves(rowrange, ydata, xmax, align):

	np.pad(ydata, (0, 200), 'constant')
	A = np.argwhere(rowrange < xmax)

	tmp = np.argwhere(rowrange >= xmax)[:align]
	B = np.flip(tmp)


	A = A[:align]
	B = B

	return ydata[A], ydata[B]


# W-E hemispheric difference

def asymmetry(A, B):
	return A - B


def main():
	m2 = Light("/home/marco/Desktop/solar/DATA/2020-09-22/drift_SJ_off/drift_SJ_off_2020-09-22_HR4D290.hdf", 2)
	darkfile2 = Calibration("/home/marco/Desktop/solar/DATA/2020-09-22/darks150/darks150_2020-09-22_HR4D290.hdf", -2)

	m1 = Light("/home/marco/Desktop/solar/DATA/2020-07-30/NOAA2767/NOAA2767B_2020-07-30_HR4D290.hdf",2)
	darkfile1 = Calibration("/home/marco/Desktop/solar/DATA/2020-07-30/NOAA2767-dark/NOAA2767-dark_2020-07-30_HR4D290.hdf",2)

	d1 = darkfile1.master
	d2 = darkfile2.master

	m1.subtract_dark(d1)
	m2.subtract_dark(d2)
	

	trans1 = np.sum(m1.data.T, axis=0)/3840
	trans2 = np.sum(m2.data.T, axis=0)/3840

	xmaxes = []

	for i in range(10, 1000):
		xmax = center(trans2, i)
		xmaxes.append(xmax)

	plt.title("Center search", fontsize=20)
	plt.ylabel("Calculated center frame", fontsize=20)
	plt.xlabel("N", fontsize=20)
	plt.plot(xmaxes, label="Calculated frame of maximum: \n median of frame number \n of N points with \n highest intensity", c="k", linewidth=5)
	plt.legend(fontsize=20)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	plt.show()

	xmax = 509


	plt.plot(trans2)
	plt.vlines(xmax, 0, 1000) 
	plt.show()
	
	

	
	A, B = halves(m2.row_range, trans2, 550, 450)

	print(A-B)


	asyms = []
	for j in range(300, 799):
		A, B = halves(m2.row_range, trans2, j, align=300)
		print(j)
		asm = np.sum(asymmetry(A, B))

		plt.plot(A, c="k")
		plt.plot(B, c="r")
		u = j - 200
		plt.title(f"Integral difference: {int(asm):5} \n given center at frame {u:03}")
		
		plt.ylabel("W-E difference (ADU)")
		plt.xlabel("Frames")
		plt.savefig(f"slider2/Slider_{j:03}.png")
		plt.clf()

		
		asyms.append(asm)

	plt.plot(asyms)
	plt.show()
	

if __name__ == "__main__":
	main()
