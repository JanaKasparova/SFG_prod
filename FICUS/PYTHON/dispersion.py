from OCAS_lib import Measurement, Light, Calibration
import numpy as np
import matplotlib.pyplot as plt
import h5py
import datetime
import pandas as pd







class Dispersion:
	"""used to determine the equation lambda (px) = f(px)
	use this code in case you doubt the quality of the conversion,
	but by default, use the included .npy files containing the precalculated
	conversion (WL_range_C and WL_range_D). Spectral lines (as of August 2020)
	remain on the same pixel values as in 2018."""

	def __init__(self,lines_file, measurement):
		#takes a spectral line file as input. for example, a list of Fraunhoffer lines
		#with (1) the name of the element/line, (2) the width of the line and (3) the central
		#wavelength, separated by \t.
		self.lines(lines_file)
		
		#takes a OCAS measurement object instance.
		self.measurement = measurement

	@property
	def wl_range(self):
		#returns 1D numpy array (np.linspace) of wavelenght range,
		#of size N (3840), based on the polynomial coefficients given
		#on the calibration certificate of the spetrometers.

		if self.measurement.letter == "C":
			_coefs = [349.457516, 0.041200951, -9.86426e-7, -2.30056e-11]
		elif self.measurement.letter == "D":
			_coefs = [476.90386963, 0.1337807626, -2.45278e-6, -5.44474e-11]
		else:
			#exception handling
			print("spectrometer not chosen")
			return None

		#reversing the order of polynomial coefficients for np.polyval function.
		#which is a faster way of generating an array of polynomial calculation results
		_coefs.reverse()
		
		wl_r = np.polyval(_coefs, self.measurement.px_range)
		return wl_r
		
	
	@property
	def new_wl(self):
		if self.measurement.letter == "C":
			_coefs = [-7.86005897e-12,-1.07215664e-06,4.13627789e-02,3.49327688e+02]
		elif self.measurement.letter == "D":
			_coefs = [1.07179498e-10,-2.70267014e-06,1.33766093e-01,4.76810734e+02]
		else:
			print("spectrometer not chosen")
			return None
		wl_r = np.polyval(_coefs, self.measurement.px_range)
		return wl_r	


	def lines(self, lines_fn):
		rf = np.genfromtxt(lines_fn, delimiter="\t", dtype=None)
		wavel, w, element = [], [], []
		for a, b, c in rf:
			wavel.append(a)
			w.append(b)
			element.append(c.decode("utf-8"))

		self.line_wls = wavel
		self.line_widths = w
		self.line_names = element
	

	def find_nearest(self, array, value):
		import math
		idx = np.searchsorted(array, value, side="left")
		if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
			return (array[idx-1], idx-1)
		else:
			return (array[idx], idx)

	def onclick(self,event, lwl):
		X = event.xdata
		nearest = self.find_nearest(self.wl_range, X)
		pxs.append(nearest[1])
		

	def close(self, event):
		# closes matplotlib window on pressing "x"
		if event.key == "x":
			plt.close()

	def selector(self):
		plt.plot(self.new_wl, self.measurement.master)
		plt.show()

		from scipy.signal import savgol_filter

		print(self.line_wls)
		for i, line_wl in enumerate(self.line_wls):
			print(line_wl)

			global pxs
			pxs = []

			inp = input(f"")
			

			window = self.line_widths[i]
			fig, ax = plt.subplots()
			clck = lambda event: self.onclick(event, lwl = line_wl)
			cid = fig.canvas.mpl_connect('button_press_event', clck)
			cid2 = fig.canvas.mpl_connect('key_press_event', self.close)


			x, y = self.new_wl, self.measurement.master

			bot, top = line_wl - 50*window - 1, line_wl + 50*window + 1
			mask = (x >= bot) & (x <= top)
			points = np.sum(mask) - 3

			if points % 2 == 0: points += 1

			ax.scatter(x[mask], y[mask], color="k")
			ax.plot(x[mask], y[mask], color="b")
			y_savgol = savgol_filter(y[mask],points,2)
			ax.plot(x[mask], y_savgol, color="b")

			ax.set_xlim(bot, top)
			ax.axvline(line_wl, color="r")
			ax.axvline(line_wl - window, color="g")
			ax.axvline(line_wl + window, color="g")
			plt.show()
			
			
			

		
			if inp == "" or inp == "y" or inp == "Y":
				plt.show()
				avg = sum(pxs)/len(pxs)
				print(f"{self.line_names[i]}\t{line_wl}\t{avg}")
			elif inp == "n" or "N":
				break
			else:
				pass
	def plotter(self):
		plt.plot(self.new_wl, self.measurement.master)
		plt.show()
			

def main():
	fraunhofer = "/home/marco/Desktop/solar/center_disk/Fraunhofer_lines.txt"
	m1 = Light("/home/marco/Desktop/solar/center_disk/mm20/stred-test-7_2020-06-22_HR4C5177.hdf",2)
	m2 = Light("/home/marco/Desktop/solar/center_disk/mm13/stred-test-clona2_2020-06-22_HR4D290.hdf",2)

	d1 = np.load("/home/marco/Desktop/solar/center_disk/mm20/0002_0050ms_DARK_C.npy")
	d2 = np.load("/home/marco/Desktop/solar/center_disk/mm13/0002_0050ms_DARK_D.npy")
	m1.subtract_dark(d1)
	m2.subtract_dark(d2)

	disp = Dispersion(fraunhofer, m2)
	disp.selector()


if __name__ == "__main__":
	main()
