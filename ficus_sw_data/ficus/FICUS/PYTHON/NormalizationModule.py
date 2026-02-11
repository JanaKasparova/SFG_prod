import sys
sys.path.append('/home/marco/Desktop/solar/OCAS_lib')
from OCAS_lib import Light, Calibration, Measurement

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.signal import savgol_filter
from scipy.optimize import curve_fit


class Normalization:
	def __init__(self, letter, wl):
		self.letter = letter
		self.wl = wl
		self.N = 3840
		self.px = np.arange(self.N)

	def first_crop(self, k):

		if self.letter == "C":
			_av = np.nanmean(k)
			_A = (self.wl < 480)		# mean
			_B = abs(k - _av) < 10*_av 	# dead pixels
			_C = k > 0 			# dead pixels in master dark
			_D = ~np.isnan(k)		# division by zero
			mask = _A & _B & _C & _D	# boolean conjunction

		elif self.letter == "D":
			_A = (480 < self.wl) & (self.wl < 890) & (self.px > 19)	# borders
			_B = (self.wl < 758) | (self.wl > 768)			# O2 band (fraunhofer A)
			_C = (self.wl < 715) | (self.wl > 735)			# H20 band
			_D = (self.wl < 812) | (self.wl > 837)			# H20 band
			mask = _A & _B & _C & _D				# boolean conjuction

		return mask
	
	def first_trend(self, k, wl_window):
		#the window of the running average (=wl_window) is given in nm, 
		mask = self.first_crop(k)

		if self.letter == "C":
			#initialize the trend array -- the output array with a smoothed line
			trend = np.zeros(self.N)
			trend[:] = np.nan

			#find an aproximate number of elements in the wl array that correspond to the window

			avwin = np.average(np.diff(self.wl))

			win = wl_window // avwin

			#make it an even number
			if win % 2 == 1:
				win += 1
			
			#looping over the k range - data with holes
			for i, c in enumerate(k):
				tr = []
				#check if we should include point in weighted sum

				if i + win < self.N:
					pass
				else:
					win = self.N - i

				for w in range(int(win)):
					if mask[i+w]:
						tr.append(k[i+w])
				tr = np.array(tr)

				if mask[int(i+w/2)]:
					try:
						trend[int(i + win/2)] = np.average(tr)
					except:
						pass
				else:
					trend[int(i+win//2)] = None
				

		if self.letter == "D":

			#initialize the trend array -- the output array with a smoothed line
			trend = np.zeros(self.N)
			trend[:] = np.nan

			#find an aproximate number of elements in the wl array that correspond to the window

			avwin = np.average(np.diff(self.wl))

			win = wl_window // avwin

			#make it an even number
			if win % 2 == 1:
				win += 1
			
			#looping over the k range - data with holes
			for i, c in enumerate(k):
				tr = []
				#check if we should include point in weighted sum

				if i + win < self.N:
					pass
				else:
					win = self.N - i

				for w in range(int(win)):
					if mask[i+w]:
						tr.append(k[i+w])
				tr = np.array(tr)

				if mask[int(i+w/2)]:
					try:
						trend[int(i + win/2)] = np.average(tr)
					except:
						pass
				else:
					trend[i] = None

		#returns array of size 3840 always.

		return np.array(trend)

	def dev_trend(self, k):
		#find a polynomial (or linear, constant) trendline for residuals of running average
		
		if self.letter == "C":
			ft = self.first_trend(k, wl_window=4) 		# window approx. 4 nm. ft is the running average. 
			ftmask = ~np.isnan(ft)				# masking if value = NaN

			res = k - ft					# residuals = k - runnning average
			avres = np.sqrt((res)**2)
			rm = ~np.isnan(res)
			pc_res = np.polyfit(self.wl[rm], res[rm], 4)
			l_res = np.polyval(pc_res, self.wl)		# fitting residuals with polynomial of deg=4
			
		elif self.letter == "D":
			ft = self.first_trend(k, wl_window=15)
			ftmask = ~np.isnan(ft)
			res = k - ft
			avres = np.sqrt((res)**2)
			rm = ~np.isnan(res)
			pc_res = np.polyfit(self.wl[rm], res[rm], 3)

			l_res = np.polyval(pc_res, self.wl)
			pass

		return (res, l_res, avres)

	def cor_dev_trend(self, k):
		#typically, this correction is much biger for C than for D
		R = self.dev_trend(k)[1] + self.first_trend(k, 5)
		return R

	def second_dev(self, k):
		#10 works quite well
		parts = 10

		#difference in y value: k - shifted moving average
		sigma = abs(k - self.cor_dev_trend(k))
		
		#window of integer size ( // operator is int division)
		win = sigma.shape[0] // parts

		#np array init.
		stepped = np.zeros(sigma.shape[0])

		for i in range(parts):
			#sigma range
			sgrg = sigma[i*win:(i+1)*win]

			#mean in given window
			av = np.nanmean(sgrg)

			#asigning the mean value to a stepped function later used for cutoff
			stepped[i*win:(i+1)*win] = av
		
		#boolean mask
		mask = sigma < stepped

		return (sigma, stepped, mask)

	def filtered_trend(self, k):
		if self.letter == "C":
			good_points = k[self.second_dev(k)[2]]
			cs_fit = np.polyfit(self.wl[self.second_dev(k)[2]], k[self.second_dev(k)[2]], 11)
			fit = np.polyval(cs_fit, self.wl)
			return fit

		if self.letter == "D":
			good_points = k[self.second_dev(k)[2]]
			cs_fit = np.polyfit(self.wl[self.second_dev(k)[2]], k[self.second_dev(k)[2]], 50)
			fit = np.polyval(cs_fit, self.wl)
			return fit


	def yshift(self, k, a, mstr):

		if self.letter == "C":
			D = (a - self.filtered_trend(k)*mstr)
			mask = ~np.isnan(D) & self.first_crop(k)
			fit = np.polyfit(self.wl[mask], D[mask], 3)

			fline = np.polyval(fit, self.wl)
			rD = np.median(D[mask] - fline[mask])
			X = fline + rD
			
			avX = np.median(X)

			mask2 = np.sqrt((D - X)**2) < 4*rD
			

			cf2 = np.polyfit(self.wl[mask2], D[mask2], 3)
			cor = np.polyval(cf2, self.wl)

			relative_D = D/a
			relative_fline = fline/a
			return (relative_D, relative_fline, X, mask2, cor)


		if self.letter == "D":
			D = a - self.filtered_trend(k)*mstr
			mask = ~np.isnan(D) & self.first_crop(k)
			fit = np.polyfit(self.wl[mask], D[mask], 0)

			fline = np.polyval(fit, self.wl)
			rD = np.median(D[mask] - fline[mask])
			X = fline + rD

			avX = np.median(X)

			mask2 = np.sqrt((D - X)**2) < 2*rD



			cf2 = np.polyfit(self.wl[mask2], D[mask2], 0)
			cor = np.polyval(cf2, self.wl)

			relative_D = D/a
			relative_fline = fline/a
			return (relative_D, relative_fline, X, mask2, cor)



	def filtered_2nd(self, k, a, mstr):
		#filt_tr = self.filtered_trend(k) + self.yshift(k, a)[4]
		filt_tr = self.yshift(k, a, mstr)[4]
		return filt_tr


class Linearity:
	def __init__(self, x, let):
		self.x = x
		self.let = let

	def linearity_ccd(self):
		if self.let == "C":
			A = [0.952836,1.320268e-5,4.91468e-10,-7.415e-13,1.46495e-16,-1.40293e-20,6.72033e-25,-1.288e-29]
		if self.let == "D":
			A = [0.963663,6.6087e-6,5.63616e-9,-2.81251e-12,5.51632e-16,-5.48221e-20,2.71142e-24,-5.30629e-29]
		S = 0
		for i, a in enumerate(A):
			S += a*self.x**i
		return S

