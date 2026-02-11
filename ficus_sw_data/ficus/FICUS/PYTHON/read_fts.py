from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

hdulist = fits.open('fts_atlas.fts')
atlas = hdulist[0].data.T


# Opening and plotting the FTS atlas as a FITS file using the astropy module,
# with displayed Fraunhoffer lines.



# Strongest (Fraunhoffer) lines in the C and D ranges.


d_lines_wl = [486.13,492.05,518.36,544.69,575.47,606.55,619.16,633.68,640,649.5,656.28,700.59,712.22,720.22,728.28,765.76,794.59,820.77,846.84,854.21,866.22]
c_lines_wl = [351.51,358.12,363.15,374.95,383.23,392.29,393.37,396.85,407.17,421.55,454.95,473.68,478.34,482.35]
c_line_names = ["Ni I","Fe I","Fe I","Fe I","Mg I","Fe I","Ca IIK","Ca IIH","Fe I","Sr II","Fe II","Fe I","Mn I","Mn I"]
d_line_names = ["H-beta","Fe I","Mg I","Fe I","Ni I","Fe I","Fe I","Fe I","Fe I","Fe I","H-alpha","Si I","Ni I","Ca I","Si I","Mg I","Fe I","Fe I","Fe I","Ca II","Ca II"]



fig, ax = plt.subplots()


a = atlas[1]/10
b = atlas[0]

ar = np.array([a,b])

np.save("FTS_atlas",ar)



ax.scatter(atlas[1]/10, atlas[0], marker="o", s=1, color="k")
plt.xlabel("Wavelenght (nm)", fontsize=30)
plt.ylabel("Intensity", fontsize=30)


A, B = min(atlas[0]), max(atlas[0])

def anot(lines, line_labels, A, B):
	for i, line in enumerate(lines):
			ax.vlines(line, A, B, color="r", linewidth=3)
			ax.annotate(line_labels[i], (line, B), fontsize=18, color="r")


anot(c_lines_wl, c_line_names, A, B)
anot(d_lines_wl, d_line_names, A, B)


plt.title("FTS Atlas", fontsize=30)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.show()
