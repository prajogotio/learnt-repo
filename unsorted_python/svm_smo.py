# Implementation of Support Vector Machine with RBF Kernel
# Making use of Sequential Minimal Optimization

from numpy import *
from pylab import *
from numpy import random

class SVM:
	EPS = 1e-12
	def __init__(self):
		self.C = 1e100
		self.error = 1e-3
		self.nonBoundedList = []
		self.kernel = RbfKernel();

	def setTrainingData(self, inputData, targetData):
		self.inputData = inputData
		self.targetData = targetData
		self.numOfData, self.inputDimension = shape(inputData)

	def setKernel(self, kernel):
		self.kernel = kernel

	def setConstraint(C):
		self.C = C

	def setTolerance(error):
		self.error = error

	def train(self):
		self.alpha = zeros((self.numOfData, 1))
		self.E = zeros((self.numOfData, 1))
		self.isCached = [False for i in xrange(self.numOfData)]
		self.bias = 0
		# Heuristic for choosing first example
		self.readAll = True
		self.kktViolated = False
		while self.kktViolated or self.readAll:
			self.kktViolated = False
			if self.readAll:
				for i in xrange(self.numOfData):
					if self.examineExample(i):
						self.kktViolated = True
			else:
				for i in xrange(self.numOfData):
					if (self.isUnbounded(self.alpha[i])):
						if self.examineExample(i):
							self.kktViolated = True
			print "readAll: ", self.readAll, " kktviolated: ", self.kktViolated
			if self.readAll:
				self.readAll = False
			elif not self.kktViolated:
				self.readAll = True

	def examineExample(self, i):
		E = self.computeError(i);
		r = E*self.targetData[i];
		if (r < -self.error and self.alpha[i] < self.C) or (r > self.error and self.alpha[i] > 0):
			# Heuristic for choosing second example

			# First heuristic: find the maximum |E1 - E2|
			j = -1
			dE = 0
			for k in xrange(self.numOfData):
				if fabs(self.computeError(k) - self.computeError(i)) > dE and i != k:
					dE = fabs(self.computeError(k) - self.computeError(i))
					j = k
			if j != -1:
				if self.takeStep(i, j):
					return True

			# Second heuristic: amongst unbounded, try taking step
			# randomize
			index = [k for k in xrange(self.numOfData)]
			random.shuffle(index)
			for k in index:
				if self.isUnbounded(self.alpha[k]):
					if self.takeStep(i, k):
						return True

			random.shuffle(index)
			for k in index:
				if self.takeStep(i, k):
					return True
		return False

	def computeError(self, i):
		error = self.E[i] if self.isCached[i] else self.computeMargin(self.inputData[i]) - self.targetData[i]
		self.E[i] = error
		self.isCached[i] = True
		return error
	
	def isUnbounded(self, alpha):
		return 0 < alpha and alpha < self.C

	def takeStep(self, i, j):
		if i == j:
			return False

		s = self.targetData[i] * self.targetData[j]
		L = max(0, self.alpha[j] + s * self.alpha[i] - (1+s)/2 * self.C)
		H = min(self.C, s * self.alpha[i] + self.alpha[j] + (1-s)/2 * self.C)
		if L == H:
			return False
		E1 = self.computeError(i)
		E2 = self.computeError(j)
		if fabs(E1-E2) < SVM.EPS:
			return False
		
		k11 = self.kernel.compute(self.inputData[i], self.inputData[i])
		k22 = self.kernel.compute(self.inputData[j], self.inputData[j])
		k12 = self.kernel.compute(self.inputData[i], self.inputData[j])
		eta = 2 * k12 - k11 - k22

		if (eta < 0):
			a2 = (E2 - E1) * self.targetData[j] / eta + self.alpha[j]
		else:
			c1 = eta/2
			c2 = self.targetData[j] * (E1 - E2) - eta * self.alpha[j]
			Lobj = c1 * L * L + c2 * L
			Hobj = c1 * H * H + c2 * H
			a2 = L if Lobj > Hobj + SVM.EPS else self.alpha[j]
			a2 = H if Hobj > Lobj + SVM.EPS else a2

		a2 = a2 if a2 > L else L
		a2 = a2 if a2 < H else H
		a1 = self.alpha[i] + s * self.alpha[j] - s * a2

		if fabs(self.alpha[j]) < SVM.EPS:
			self.alpha[j] = 0

		if fabs(self.alpha[j] - self.C) < SVM.EPS:
			self.alpha[j] = self.C

		if fabs(a2 - self.alpha[j]) < SVM.EPS * (self.alpha[j] + a2 + SVM.EPS):
			return False

		b1 = E1 + self.targetData[i] * (a1 - self.alpha[i]) * k11 + self.targetData[j] * (a2 - self.alpha[j]) * k12 + self.bias;
		b2 = E2 + self.targetData[i] * (a1 - self.alpha[i]) * k12 + self.targetData[j] * (a2 - self.alpha[j]) * k22 + self.bias;
		newBias = (b1+b2)/2
		self.E[i] = self.E[i] + self.targetData[i] * (a1 - self.alpha[i]) * k11 + self.targetData[j] * (a2 - self.alpha[j]) * k12 + self.bias - newBias
		self.E[j] = self.E[j] + self.targetData[i] * (a1 - self.alpha[i]) * k12 + self.targetData[j] * (a2 - self.alpha[j]) * k22 + self.bias - newBias
		self.bias = newBias
		self.alpha[i] = a1
		self.alpha[j] = a2
		return True

	def computeMargin(self, input):
		ret = 0
		for i in xrange(self.numOfData):
			prod = self.kernel.compute(self.inputData[i], input)
			ret += self.alpha[i] * self.targetData[i] * prod
		return ret

	def classify(self, input):
		ret = self.computeMargin(input)
		return 1 if ret - self.bias >= 0 else -1


	def getNumberOfSV(self):
		ret = 0
		for i in xrange(self.numOfData):
			ret += 1 if (self.isUnbounded(self.alpha[i])) else 0
		return ret


class RbfKernel:
	def __init__(self):
		self.gamma = 1
		pass

	def setGamma(self, gamma):
		self.gamma = gamma

	def compute(self, x1, x2):
		return exp(-linalg.norm(x1-x2) * self.gamma)