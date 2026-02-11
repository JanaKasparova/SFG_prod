import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
from limb_darkening import ld_simple, centrally_symmetric
import matplotlib.cm as cm



def wl_range(Dataset, N):
	# returns 1D numpy array (np.linspace) of wavelenght range
	
	dataset_metadata = {k:v for (k, v) in Dataset.attrs.items()}
	### BE EXTREMELY CAREFUL HERE!! CHANGES IN WL_STEP WERE MADE
	wl_start = float(dataset_metadata["COL_VAL0"])
	wl_step =float(dataset_metadata["COL_DELTA"])
	wl_stop = wl_start + N*wl_step
	wl_range = np.linspace(0,wl_stop,N)
	return wl_range


def custom_wl_range(Dataset, N, letter):
	# returns 1D numpy array (np.linspace) of wavelenght range,
	# according to Stanek, 2019
	if letter == "C":
		wl_start = 349.472
		quad = -109.1e-8
		wl_step = 0.0412810
	elif letter == "D":
		wl_start = 476.671
		quad = -27.878e-7
		wl_step = 0.134204
	else:
		print("spectrometer not chosen")
		return None
	wl_range = []
	for px in range(N):
		wl_range.append(wl_start + px*wl_step + px**2*quad)
	return np.array(wl_range)

def time_stamp(Dataset, row_number):
	dataset_metadata = {k:v for (k, v) in Dataset.attrs.items()}
	t0 = str(dataset_metadata["DATE_ROW0"])[2:-1].split()
	dt = float(dataset_metadata["ROW_DELTA"])
	return t0, dt



def rows(Dataset):
	return Dataset.shape[0]

def menu(hdf_obj):
	# returns index of chosen dataset and its name in a tuple
	from simple_term_menu import TerminalMenu
	ls = list(hdf_obj.keys())
	options = [x for i,x in enumerate(ls)]
	terminal_menu = TerminalMenu(options)
	local_index = terminal_menu.show()
	return (local_index, ls[local_index])


def plot_spectrum_ld(wl_r_a, wl_r_b, Dataset_a, Dataset_b, row_numbers, mu):
	#wavelenght range a (np.linespace array), wavelenght range b (np.linespace array)
	#2x dataset objects - corresponding, 
	# row number (time equivalent)
	colors = cm.rainbow(np.linspace(0, 1, len(row_numbers)))
	for i, row_number in enumerate(row_numbers):
		data_a, data_b = np.array(Dataset_a)[row_number], np.array(Dataset_b)[row_number]
		print(data_a.shape)
		print(data_b.shape)
		v_b = [max(wl_r_a), min(wl_r_b)]
		h_b = [max(max(data_a), max(data_b)), min(min(data_a), min(data_b))]
		plt.vlines(v_b[0],h_b[0],h_b[1], linewidth=3, color="k")
		plt.vlines(v_b[1],h_b[0],h_b[1], linewidth=3, color="k")
		plt.plot(wl_r_a, data_a,linewidth=1, color=colors[i])
		plt.plot(wl_r_b, data_b,linewidth=1, color=colors[i])
	
	if len(row_numbers) > 0:
		#dumd condition
		#limb darkened data
		row_number = row_numbers[0]
		data_a, data_b = np.array(Dataset_a)[row_number], np.array(Dataset_b)[row_number]
		ld_a = ld_simple(wl_r_a, data_a, mu)
		ld_b = ld_simple(wl_r_b, data_b, mu)
		plt.plot(wl_r_a, ld_a, c="r")
		plt.plot(wl_r_b, ld_b, c="r")
	
	row_number = row_numbers[1]
	data_a, data_b = np.array(Dataset_a)[row_number], np.array(Dataset_b)[row_number]
	plt.plot(wl_r_a, data_a,linewidth=1, color="k")
	plt.plot(wl_r_b, data_b,linewidth=1, color="k")

	plt.show()



def rainbow_plot(dataset, k):
	T = dataset.shape[0]
	time = np.linspace(0,T,T)
	colors = cm.rainbow(np.linspace(0, 1, len(k)))
	for i, frame in enumerate(k):
		Is = np.array(dataset).T[frame]
		plt.plot(time,Is, c=colors[i])
		np.save(f"{frame}_rainbow_plot", Is)
	plt.show()
	
	


def main():
	N = 3840
	pix_range = np.linspace(0,N,N)
	files = [sys.argv[1], sys.argv[2]]
	master_darks = ["/home/marco/Desktop/solar/prvni_tyden/hdfs/darks/16-06-2020/lamp_master_darkC.npy", "/home/marco/Desktop/solar/prvni_tyden/hdfs/darks/16-06-2020/lamp_master_darkD.npy"]
	dark_c, dark_d = np.load(master_darks[0]), np.load(master_darks[1])
	c_lines = [48,211,344.5,628,837,1068,1095.5,1185,1453.2,1834.5,2754.5,3296,3434,3553.5]
	d_lines = [70,114,313,513,748.5,988.5,1086.2,1200,1249.5,1324.8,1378.2,1730,1824,1888,1954,2260,2497,2718.5,2938,3000,3103]
	d_lines_wl = [486.13,492.05,518.36,544.69,575.47,606.55,619.16,633.68,640,649.5,656.28,700.59,712.22,720.22,728.28,765.76,794.59,820.77,846.84,854.21,866.22]
	c_lines_wl = [351.51,358.12,363.15,374.95,383.23,392.29,393.37,396.85,407.17,421.55,454.95,473.68,478.34,482.35]
	c_line_names = ["Ni I","Fe I","Fe I","Fe I","Mg I","Fe I","Ca IIK","Ca IIH","Fe I","Sr II","Fe II","Fe I","Mn I","Mn I"]
	d_line_names = ["H-beta","Fe I","Mg I","Fe I","Ni I","Fe I","Fe I","Fe I","Fe I","Fe I","H-alpha","Si I","Ni I","Ca I","Si I","Mg I","Fe I","Fe I","Fe I","Ca II","Ca II"]
	with h5py.File(files[0], "r") as my_hdf:
		with h5py.File(files[1], "r") as my_hdf2:
			index, dataset_name = menu(my_hdf)
			first_dataset = my_hdf.get(dataset_name)
			#list of keys in second file (of second spectrometer)
			ls2 = list(my_hdf2.keys())
			second_dataset = my_hdf2.get(ls2[index])
			#list of two wavelength ranges
			wl_rs_roman = [wl_range(first_dataset,N), wl_range(second_dataset,N)]
			wl_rs_stanek = [custom_wl_range(first_dataset,N, "C"), custom_wl_range(second_dataset,N, "D")]
			hp_c = hot_pixels_master(dark_c)
			hp_d = hot_pixels_master(dark_d)
			data_c = subtract_dark(first_dataset, dark_c)
			data_d = subtract_dark(second_dataset, dark_d)
			plot_simple(pix_range, data_c[150], c_lines, c_line_names)
			plot_simple(pix_range, data_d[150], d_lines, d_line_names)
			#plot_simple(wl_rs_stanek[0], data_c[150], c_lines_wl,  c_line_names)
			#plot_simple(wl_rs_stanek[1], data_d[150], d_lines_wl,  d_line_names)
			heat(data_c, data_d)
			
			

			
def heat(dataA, dataB):
	data = np.append(dataA, dataB, axis=1)
	plt.imshow(data, cmap="hot", aspect=15)
	plt.vlines(3840, 0, 318)
	plt.show()



def plot_simple(xax, data, lines, line_labels):
	#input direct, data is 1D array, not T*N array
	N = 3840
	ax = plt.gca()
	A, B = min(data)-100, max(data)
	ax.plot(xax, data, color="k")
	for i, line in enumerate(lines):
		ax.vlines(line, A, B, color="r")
		ax.annotate(line_labels[i], (line, A+70), fontsize=15, c="r")
		#ax.annotate(str(lines), (line, A+30))
	plt.show()


def dark(dataset):
	L = dataset.shape[0]
	S = np.sum(dataset, axis=0)
	avs = S/L
	av_val = np.sum(avs, axis=0)/dataset.shape[1]
	STD = np.std(dataset)
	hot_pixels = [i for i, x in enumerate(avs) if x > (av_val + 3*STD)]
	print(hot_pixels)
	print(f"STD of dark: {STD}")
	print(f"Average value  of dark : {av_val}")
	return(avs)


def hot_pixels(dark_dataset):
	L = dark_dataset.shape[0]
	#[0] or [1]? need to find out
	S = np.sum(dark_dataset, axis=0)
	avs = S/L
	av_val = np.sum(avs, axis=0)/dark_dataset.shape[1]
	STD = np.std(dark_dataset)
	hps = [i for i, x in enumerate(avs) if x > (av_val + 3*STD)]
	return hps
	

def hot_pixels_master(dark_dataset):
	L = dark_dataset.shape[0]
	S = np.sum(dark_dataset)
	av = S/L
	STD = np.std(dark_dataset)
	hps = [i for i, x in enumerate(dark_dataset) if abs(x-av) > (3*STD)]
	return hps

def plot_spectrum_cs(wl_r_a, wl_r_b, Dataset_a, Dataset_b, row_number):
	#wavelenght range a (np.linespace array), wavelenght range b (np.linespace array)
	#2x dataset objects - corresponding, 
	# row number (time equivalent)
	data_a, data_b = np.array(Dataset_a)[row_number], np.array(Dataset_b)[row_number]
	print(data_a.shape)
	print(data_b.shape)
	v_b = [max(wl_r_a), min(wl_r_b)]
	h_b = [max(max(data_a), max(data_b)), min(min(data_a), min(data_b))]
	plt.plot(wl_r_a, data_a)
	plt.plot(wl_r_b, data_b)

	#centrally symmetric problem
	cs_a = centrally_symmetric(wl_r_a, data_a, 0.6, 0.61)
	cs_b = centrally_symmetric(wl_r_b, data_b, 0.6, 0.61)
	plt.plot(wl_r_a, 1000*data_a/cs_a)
	plt.plot(wl_r_b, 1000*data_b/cs_b)

	plt.vlines(v_b[0],h_b[0],h_b[1], linewidth=3, color="k")
	plt.vlines(v_b[1],h_b[0],h_b[1], linewidth=3, color="k")
	plt.show()


def subtract_dark(dataset, dark):
	#subtract dark and remove hot pixels by averaging around
	data = np.array(dataset)
	
	return data - dark


if __name__ == "__main__":
	main()
