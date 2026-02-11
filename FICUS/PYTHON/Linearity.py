# linearity correction coefficients

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
