import sys
sys.path.append('/home/marco/Desktop/solar/OCAS_lib')
from OCAS_lib import Light, Calibration, Measurement

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

from mpl_toolkits.mplot3d import Axes3D

def main3d():
	rmD = Light("/home/marco/Desktop/solar/DATA/ficus_suncentrenocorr/2020-07-10/Stred-500/Stred-500_2020-07-10_HR4D290.hdf")
	dark = Calibration("/home/marco/Desktop/solar/DATA/2020-07-22/last_dark/last_dark_2020-07-22_HR4C5177.hdf", -2)
	md = dark.master
	rmD.subtract_dark(md)
	print(rmD)
	d = rmD.data
	print(d.shape)
	px = rmD.px_integral(rmD.data)
	d = rebin(rmD.data, (265, 384))
	xx, yy = np.mgrid[0:d.shape[0], 0:d.shape[1]]

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax2 = fig.gca()
	ax.plot_surface(xx, yy, d ,rstride=1, cstride=1, cmap="binary",
        linewidth=0)
	fig.suptitle("Spectrometer D | Drift over sunspot 2020-07-23", fontsize=25)
	ax.set_xlabel("Frame", fontsize=20)
	ax.set_ylabel("Pixels / 10", fontsize=20)
	ax.set_zlabel("Counts (ADU)", fontsize=20)

	#plt.savefig("Drift3dC_s.png", dpi=1000)
	plt.show()
	print("DONE")


def rebin(arr, new_shape):
	shape = (new_shape[0], arr.shape[0] // new_shape[0], new_shape[1], arr.shape[1] // new_shape[1])
	return arr.reshape(shape).mean(-1).mean(1)

def main():
	rmD = Light("/home/marco/Desktop/solar/DATA/2020-07-22/last_drift/last_drift_2020-07-22_HR4D290.hdf",-2)
	dark = Calibration("/home/marco/Desktop/solar/DATA/2020-07-22/last_dark/last_dark_2020-07-22_HR4D290.hdf", -2)

	md = dark.master
	rmD.subtract_dark(md)
	print(rmD)
	print(rmD.data.shape)
	d = rebin(rmD.data, (27, 384))
	print(d)
	
	print(np.argmax(rmD.px_integral(d)))
	

	
	
	f, (ax1, ax2) = plt.subplots(1,2)
	f.suptitle("Spectrometer D | Drift over half of the solar disc", fontsize=25)
	ax1.imshow(d, origin="upper", aspect=2,cmap="hot")
	px = np.array(rmD.px_integral(rmD.data))/3840
	ax2.scatter(px, rmD.t_range, c="k", s=15)
	ax2.plot(px, rmD.t_range, c="k", linewidth=2)
	ax2.invert_yaxis()
	ax1.set_ylabel("Frame", fontsize=20)
	ax2.set_ylabel("Time", fontsize=20)
	ax1.set_xlabel("Wavelength (nm)", fontsize=20)
	ax2.set_xlabel("Average pixel count (ADU)", fontsize=20)
	ax2.tick_params(labelsize=13)
	plt.show()


if __name__ == "__main__":
	main3d()

