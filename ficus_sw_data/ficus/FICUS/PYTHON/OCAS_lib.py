#a module for FICUS measurements
#marco souza de joode

import numpy as np
import matplotlib.pyplot as plt
import h5py
import datetime
import pandas as pd
from operator import itemgetter

#using python 3.6.9
#necessary to install using pip
#using terminal command
#pip install numpy
#pip install matplotlib
#pip install h5py
#pip install DateTime
#pip install pandas
#pip install simple-term-menu

#if this does not work, try
#pip3 install numpy,
#pip3 install matplotlib...



class Measurement:
	"""the parent class for all measurements
	each specific measurement (ie. lights from C spectrometer, darks from D spectrometer...)
	are treated as instances of child classes of Measurement (Light, Calibration)."""

	def __init__(self, filename, dataset_index=None):
		#loading the hdf file
		with h5py.File(filename, "r") as self.of:

			#dataset_keys is the list of names of datasets in open hdf (self.of)
			self.dataset_keys = list(self.of.keys())

			#choosing the dataset in the file. if not given, a menu opens (by self.menu())
			if dataset_index is not None:
				self.dataset = self.of[self.dataset_keys[dataset_index]]
			else:
				print("Menu: ")
				self.dataset = self.of[self.menu()[1]]

			#raw_data is the contained data in the dataset, without calibration. Raw_data is accesible even after correction.
			self.raw_data = np.array(self.dataset)
			
			#metadata is a dict containing additional info (as t0, dt). Dispersion coefs. are wrong.
			self.metadata = {k:v for (k, v) in self.dataset.attrs.items()}

			#print(self.metadata)
			#number of rows = number of frames taken by spectrometer.
			#N is the number of pixels in each frame (if nothing changes, N=3840).
			self.rows = self.raw_data.shape[0]
			self.row_range = np.linspace(0, self.rows-1, self.rows)
			self.N = self.raw_data.shape[1]
						
			#np.array() of pixel numbers from 0 to 3840
			self.px_range = np.linspace(0, self.N-1, self.N)

			#parsing the datetime of the first frame (DATE_ROW0) from metadata as a datetime.datetime object,
			#converting ROW_DELTA from metadata to datetime.timedelta object,
			#making a np.array() of the times (of dimension self.rows)
			try:
				self.t_row0 = datetime.datetime.strptime(self.metadata["DATE_ROW0"].decode("utf-8"),"%Y-%m-%d %H:%M:%S.%f")
			except KeyError:
				self.t_row0 = datetime.datetime.strptime(self.metadata["DATE_CREATE"].decode("utf-8"),"%Y-%m-%d %H:%M:%S.%f")

			self.dt = datetime.timedelta(milliseconds=int(self.metadata["ROW_DELTA"]))
			self.t_range = np.array([self.t_row0 + k*self.dt for k in range(self.rows)])

			#by deafult, data = raw_data, but this is toggeled when subtracting dark from light
			#self.data is the principal attribute used for analysis
			self.data = self.raw_data

			#selected frames is a dictionary. It is used to store selected frame ranges, given by the add_frame_selection() method.
			self.frame_selections = {}
			
			#remove this part, if problems with memory
			self.frame_selections["DATA"] = self.data

			#self.letter is the used spectrometer. 
			# C == HR4C5177
			# D == HR4D290
			if filename[-5] == "7":
				self.letter = "C"
			elif filename[-5] == "0":
				self.letter = "D"
			else:
				self.letter = None
	
	@property			
	def wl_range(self):
		#change path to wl_range arrays as needed!
		if self.letter == "C":
			_rng = np.load("/home/marco/Desktop/solar/OCAS_lib/WL_range_C.npy")
		elif self.letter == "D":
			_rng = np.load("/home/marco/Desktop/solar/OCAS_lib/WL_range_D.npy")
		else:
			print("Error: self.letter not defined.")
		return _rng 
		
	def reasign_data(self):
		#a method for asigning self.data
		#pythonic "ask for forgiveness not permission"
		try:
			self.data = self.calibrated_data
		except AttributeError:
			self.data = self.raw_data

	def menu(self):
		#returns index of chosen dataset and its name in a tuple
		#when dataset number not given in the __init__ method

		#imports an external module 
		from simple_term_menu import TerminalMenu

		#underscore variables are "private"
		_options = [x for i, x in enumerate(self.dataset_keys)]
		_terminal_menu = TerminalMenu(_options)
		_local_index = _terminal_menu.show()
		return (_local_index, self.dataset_keys[_local_index])

	

	
	def px_integral(self, selection):
		#return a "photmetric" value, the integral of the whole spectral range
		#due to the @property decorator, px_integral can be accessed as an attribute
		pxinteg = [np.sum(fr) for fr in selection]
		return pxinteg

	@property
	def master(self):

		#master frame of the entire self.data object
		_mstr = np.sum(self.data, axis=0)/self.rows
		
		self.frame_selections["MASTER"] = _mstr
		return _mstr

	def add_frame_selection(self, selection_name, selection):
		#selection is a np.array or indexes
		selection.flatten()	
		self.frame_selections[selection_name] = np.array(itemgetter(*selection)(self.data))

	def selection_master(self, selection_name):
		s = self.frame_selections[selection_name]
		r = s.shape[0]
		#self.frame_selections[f"{selection_name}_master"] = np.sum(s, axis=0)/r
		
		
	

		
class Light (Measurement):

	# Child class of Measurement, for measurements of the solar disc

	def subtract_dark(self, masterdark):
		#masterdark_fn is a 1D np array
		self.dark = masterdark
		self.calibrated_data = self.raw_data - self.dark
		self.reasign_data()

	def master_light(self, frame):

		#master for light frames works differently.
		#it allows for small shifts up and down due to atmosphere etc.
		means = np.sum(frame, axis=0)/self.N
		return means
		


class Calibration (Measurement):
	# Child class of Measurement, for calibration series (dark, bias)
	pass


