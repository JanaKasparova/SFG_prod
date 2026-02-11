from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2 as cv
from PIL.TiffTags import TAGS
from datetime import datetime
import sys
sys.path.append('/home/marco/Desktop/solar/OCAS_lib')
from OCAS_lib import Light, Calibration, Measurement

pa = "/home/marco/Desktop/solar/DATA/ficus_suncentrenocorr/2020-07-10/Stred-SJ/SlitJaw/Stred-SJ_20200710_091019.tif"

class SlitJaw:
	# requires PIL.Image

	def __init__(self, filenames):

		# requires a sorted list of strings, which are the paths to individual tif images.
		self.filenames = filenames

		# making an attribute out of t_range
		# possibly better than @property decorator?
		self.t_range = self.t_range_calc()

		# opening the zeroth image to retrieve width and height
		_im = Image.open(filenames[0])
		_a = np.asarray(_im)
		
		# width and height as an int and as a np.arange
		self.H, self.W = _a.shape
		self.h_range = np.arange(self.H)
		self.w_range = np.arange(self.W)

		# circular slit parameters
		# hard coded coordinates and radii
		self.X = 485 # in height
		self.Y = 945 # in width
		self.r_outer = 100
		self.r_inner = 65
		


	def t_range_calc(self):
		T = []
		for l in self.filenames:
			
			# hardcoded string format for SlitJaw images
			# takes time from filename
			timestr = l[-19:-4]
			
			# uses datetime.datetime to convert string to datetime object
			dto = datetime.strptime(timestr, '%Y%m%d_%H%M%S')

			# build up T 
			T.append(dto)

		return T


	def rectangular_mask(self, win, X=None, Y=None):
		if X is None:
			X = self.X
		if Y is None:
			Y = self.Y

		# making a square cutout from a frame
		# using center coordinates (X,Y) and a sidelength = win
		hmask = ((X - win < self.h_range) & (self.h_range < X + win)) 
		wmask = ((Y - win < self.w_range) & (self.w_range < Y + win))
		
		# masking an array of shape (self.H, self.W) using
		# the rectangular selection
		mask = wmask[np.newaxis, :] & hmask[:, np.newaxis]
		return mask
		

	def circular_mask(self, radius, X=None, Y=None):

		if X is None:
			X = self.X
		if Y is None:
			Y = self.Y

		# making a np open grid
		gridX, gridY = np.ogrid[:self.H, :self.W]

		# pythagorean distance from (X,Y) matrix
		dist_from_center = np.sqrt((gridX - X)**2 + (gridY - Y)**2)

		# masking everything inside given radius
		mask = dist_from_center <= radius
		return mask

	def ring_mask(self, X=None, Y=None, r_inner=None, r_outer=None):

		if X is None:
			X = self.X
		if Y is None:
			Y = self.Y
		if r_inner is None:
			r_inner = self.r_inner
		if r_outer is None:
			r_outer = self.r_outer
	
		# two circular masks centered at same point
		# with diferent radii
		outer_mask = self.circular_mask(radius=r_outer, X=X, Y=Y)
		inner_mask = self.circular_mask(radius=r_inner, X=X, Y=Y)

		# XOR of both masks results in ring
		mask = np.logical_xor(outer_mask, inner_mask)
		return mask

	def masked_values(self, fn, mask):
		
		# load image using PIL and convert to np array
		im = Image.open(fn)
		a = np.asarray(im)

		# use selected mask
		selected = np.ma.masked_array(a, ~mask)
		return selected

	def average_values(self, mask):
		# return average value of masked region on frame
		avgs = []
		for fn in self.filenames:
			masked_vals = self.masked_values(fn, mask)
			avg = np.average(masked_vals)
			avgs.append(avg)
		return avgs
			


def main():
	
	# load SJ tif frames
	L = glob.glob("/home/marco/Desktop/solar/DATA/2020-09-22/SJ_drift/SlitJaw/*.tif")
	
	# sort the list of strings and 
	L = sorted(L)

	print(len(L))

	SJ = SlitJaw(L)
	
	



	fig = plt.figure(figsize=(10,8))
	a1 = fig.add_subplot(331)
	a2 = fig.add_subplot(332)
	a3 = fig.add_subplot(333)
	a4 = fig.add_subplot(334)
	a5 = fig.add_subplot(335)
	a6 = fig.add_subplot(336)
	a7 = fig.add_subplot(337)
	a8 = fig.add_subplot(338)
	a9 = fig.add_subplot(339)

	As = [a1,a2,a3,a4,a5,a6,a7,a8,a9]

	for i, a in enumerate(As):
		_im = Image.open(L[i*9 + 3])
		S = f"{L[9*i + 3][-10:-4]}"
		a.set_title(f"{S[1:2]}:{S[2:4]}:{S[4:6]} UTC")
		print(L[9*i + 3])
		_a = np.asarray(_im)
		a.imshow(_a)
	plt.show()
	



def main2():
	# load SJ tif frames
	L = glob.glob("/home/marco/Desktop/solar/DATA/2020-09-22/SJ_drift/SlitJaw/*.tif")
	_im = Image.open(L[0])
	_a = np.asarray(_im)
	plt.imshow(_a)
	plt.show()

if __name__ == "__main__":
	main()


