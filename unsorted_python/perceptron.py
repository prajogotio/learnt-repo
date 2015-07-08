from numpy import *
from pylab import *
from numpy import random

def train(data, target, iteration=1500, learning=0.25):
	inputSize, inputDimension = shape(data)
	outputSize, outputDimension = shape(target)
	assert (inputSize == outputSize)
	data = concatenate((-ones((inputSize, 1)), data), axis=1)
	inputDimension += 1
	weights = random.rand(inputDimension, outputDimension)*0.1 - 0.05
	iter = 0
	while iter < iteration:
		for i in xrange(inputSize):
			y = dot(data[i], weights)
			y = where(y > 0, 1, 0)
			weights = weights + learning * dot(data[i].reshape(inputDimension,1), (target[i] - y)).reshape(inputDimension, outputDimension)
		iter += 1
	return weights

def error_rate(data, target, weights):
	inputSize, inputDimension = shape(data)
	data = concatenate((-ones((inputSize, 1)), data), axis=1)
	res = dot(data, weights)
	res = where(res > 0, 1, 0)
	delta = res - target
	wrong = 0
	for row in delta:
		correct = True
		for x in row:
			if x != 0:
				correct = False
				break
		if not correct:
			wrong += 1
	return 1.0 * wrong / inputSize
