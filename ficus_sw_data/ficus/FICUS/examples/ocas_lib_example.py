#Exaple of usage


def main():
	#create instance of the Calibration child class
	#call the masterframe() method
	Dark_C = Calibration(filename="/home/marco/Desktop/solar/prvni_tyden/hdfs/darks1/darks1_2020-06-17_HR4C5177.hdf", dataset_index=2)
	Dark_C.master

	#two Light child class instances
	m2 = Light(filename="/home/marco/Desktop/solar/sun_out/sun_area_centre_out_2020-06-25_HR4C5177.hdf",dataset_index=6)
	m1 = Light(filename="/home/marco/Desktop/solar/sun_out/sun_area_centre_out_2020-06-25_HR4D290.hdf",dataset_index=6)


	DC = np.load("/home/marco/Desktop/solar/sun_out/0002_0500ms_DARK_C.npy")
	DD = np.load("/home/marco/Desktop/solar/sun_out/0002_0500ms_DARK_D.npy")

	# for plotting, the command is
	# plt.plot(X,Y)
	# plt.show()
	# where X,Y are arrays (or array-like objects) of the same dimension

	
	#the plotting module is called matplotlib and there is plenty of documentation online


	m2.subtract_dark(DD)
	plt.plot(m2.px_integral)
	plt.show()
	m2.add_frame_selection("selection_1", np.arange(0,20))
	m2.add_frame_selection("selection_2", np.arange(140, 160))
	print(m2.frame_selections)
	print(m2.rows)
	m2.selection_master("selection_1")
	m2.selection_master("selection_2")
	print(m2.frame_selections)
	f, (ax1, ax2) = plt.subplots(2,1,sharex=True)
	ax1.plot(m2.frame_selections["selection_1_master"])
	ax1.plot(m2.frame_selections["selection_2_master"])
	ax2.plot(m2.frame_selections["selection_2_master"]/m2.frame_selections["selection_1_master"])
	plt.show()

	plt.plot(m1.t_range, m1.px_integral)
	plt.plot(m2.t_range, m2.px_integral)
	plt.show()

	plt.plot(m1.px_range, m1.data[150])
	plt.show()

