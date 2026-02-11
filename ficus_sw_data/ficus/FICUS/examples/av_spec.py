import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

def main():
	from hd_func import menu, wl_range, time_stamp, hot_pixels

	N = 3840
	files = [sys.argv[1], sys.argv[2]]
	
	with h5py.File(files[0], "r") as my_hdf:
		with h5py.File(files[1], "r") as my_hdf2:
			with open("diameter_variation_darks.txt", "a") as f:
				index, dataset_name = menu(my_hdf)
				first_dataset = my_hdf.get(dataset_name)
				#list of keys in second file (of second spectrometer)
				ls2 = list(my_hdf2.keys())
				second_dataset = my_hdf2.get(ls2[index])
				#list of two wavelength ranges
				wl_rs = [wl_range(first_dataset,N), wl_range(second_dataset,N)]
				#print(f"Hot pixels : {hot_pixels(first_dataset)}")
				#print(f"Hot pixels : {hot_pixels(second_dataset)}")
				#time_stamp(first_dataset, 0)[0][1] + "_master_dark_C"
				C = master_dark(first_dataset, f"{dataset_name}_C")
				D = master_dark(second_dataset,f"{dataset_name}_D")
				f.write(f"{files[0]}\t{dataset_name}\tC\t{average_dark(C)}\n")
				f.write(f"{files[1]}\t{dataset_name}\tD\t{average_dark(D)}\n")


def master_dark(dataset, name):
	L = dataset.shape[0]
	S = np.sum(dataset, axis=0)
	avs = np.array(S/L)
	av_val = np.sum(avs, axis=0)/dataset.shape[1]
	STD = np.std(dataset)
	np.save(name, avs)
	print("the array has been saved")
	return(avs)


def average_dark(dark):
	L = dark.shape[0]
	S = np.sum(dark)
	avs = (S/L)
	STD = np.std(dark)
	return(f"{avs}\t{STD}")

def subtract_dark(dataset, dark):
	data = np.array(dataset)
	return data - dark

if __name__ == "__main__":
	main()
