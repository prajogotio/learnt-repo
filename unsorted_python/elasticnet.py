from numpy import *
from pylab import *


class LinearRegression:
	def __init__(self, alpha, regulator):
		self.alpha = alpha
		self.regulator = regulator
		self.convergenceRate = 0.001

	def setSample(self, data, target):
		self.data = data
		self.target = target
		dim = shape(data)[1]
		self.weight = zeros((dim, 1))

	def updateError(self):
		(size, dim) = shape(self.data)
		self.error = 0
		for i in xrange(size):
			t = 0
			for j in xrange(dim):
				t += self.weight[j] * self.data[i][j]
			dy = (self.target[i] - t)
			self.error += dy * dy


	def train(self):
		(size, dim) = shape(self.data)
		converged = False
		j = 0
		self.updateError()
		while not converged:
			Sj = 0
			Mj = 0
			prevError = self.error
			for i in xrange(size):
				t = 0
				for k in xrange(dim):
					if k == j:
						continue
					t += self.weight[k] * self.data[i][k]
				Sj += self.data[i][j] * (self.target[i] - t)
				Mj += self.data[i][j] * self.data[i][j]
			r = Mj + self.regulator * (1 - self.alpha)
			u = self.regulator * self.alpha
			if Sj > u:
				self.weight[j] = (Sj - u) / r
			elif Sj < -u:
				self.weight[j] = (Sj + u) / r
			else:
				self.weight[j] = 0

			self.updateError()
			curError = self.error
			if abs(curError - prevError) < self.convergenceRate:
				converged = True
			j = (j+1) % dim

	def regress(self, x):
		dim = shape(self.weight)[0]
		t = 0
		for i in xrange(dim):
			t += self.weight[i] *  x[i]
		return t



