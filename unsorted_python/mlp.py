from numpy import *
from pylab import *
from numpy import random

def g(x, type):
	if type == 'linear':
		return x
	elif type == 'tanh':
		return  ( exp(x) - exp(-x) ) / ( exp(x) + exp(-x) )
	elif type == 'softmax':
		if (x >= 700):
			return 1e300
		return exp(x)

def dg(x, type):
	if type == 'linear':
		return 1
	elif type == 'tanh':
		t = g(x, type)
		return 1 - t*t
	elif type == 'softmax':
		return 1

def train(data, target, layers, activation='tanh', iter=4000, eta=0.2, valid_input=None, valid_output=None, error_margin=0.001, print_error=False):
	(data_size, input_dimension) = shape(data)
	(target_size, target_dimension) = shape(target)
	L = len(layers)
	assert (data_size == target_size)
	f = vectorize(g)
	df = vectorize(dg)
	w = []
	h = []
	s = []
	x = []
	x.append(zeros((input_dimension, 1)))
	h.append(zeros((1, 1)))
	w.append(zeros((1, 1)))
	s.append(zeros((1, 1)))
	for i in xrange(len(layers)):
		(prev_row, prev_col) = shape(x[i])
		cur_dim = layers[i]
		w.append(random.rand(prev_row, cur_dim) * 0.1 - 0.05)
		s.append(zeros((cur_dim, 1)))
		h.append(zeros((cur_dim, 1)))
		x.append(zeros((cur_dim+1, 1)))
		x[i+1][cur_dim] = 1;				# initializing the bias value
	
	seed = list(range(data_size))

	validated = False
	prev_error_rate = 1e300
	trail_error_rate = 2e300
	if valid_input is not None:
		iter = 50000000		# something big
		validated = True

	# stochastic version of MLP
	for iteration in xrange(iter):
		random.shuffle(seed)
		data = data[seed, :]
		target = target[seed, :]
		for k in xrange(data_size):
			# initialize input nodes
			x[0] = data[k].reshape(input_dimension, 1)

			# forward pass
			for layer in xrange(L):
				next = layer+1
				dnext, dump = shape(x[next])
				h[next] = dot(transpose(x[layer]), w[next]).reshape(dnext-1, 1)
				if activation == 'softmax':
					normalizer = sum(exp(x[next]))
					x[next] = append(f(h[next], activation) / normalizer, array([[1]]), axis=0)
				else:	
					x[next] = append(f(h[next], activation), array([[1]]), axis=0)

			# backward propagation
			# initialize s of last layer
			for i in xrange(target_dimension):
				s[L][i] = (target[k][i] - x[L][i]) * dg(h[L][i], activation)
			for i in xrange(layers[L-2]+1):
				for j in xrange(target_dimension):
					w[L][i][j] += eta * s[L][j] * x[L-1][i]
			for layer in reversed(xrange(L)):
				for j in xrange(layers[layer-1]):
					s[layer][j] = 0
					for k in xrange(layers[layer]):
						s[layer][j] += s[layer+1][k] * w[layer+1][j][k] * dg(h[layer][j], activation)
				if layer == 1:
					break
				for i in xrange(layers[layer-1]+1):
					for j in xrange(layers[layer]):
						w[layer][i][j] += eta * s[layer][j] * x[layer-1][i]
			
			for i in xrange(input_dimension):
				for j in xrange(layers[0]):
					w[1][i][j] += eta * s[1][j] * x[0][i]
		if validated:
			current_error_rate = validate(valid_input, valid_output, w, layers, activation=activation)
			if (print_error):
				print current_error_rate
			if ((prev_error_rate - current_error_rate) < error_margin) and ((trail_error_rate - prev_error_rate) < error_margin):
				break
			trail_error_rate = prev_error_rate
			prev_error_rate = current_error_rate
	return w

def compute(input, w, layers, activation='tanh'):
	(data_size, input_dimension) = shape(input)
	f = vectorize(g)
	L = len(layers)
	x = []
	x.append(zeros((input_dimension, 1)))
	for i in xrange(len(layers)):
		curDim = layers[i]
		x.append(zeros((curDim+1, 1)))
		x[i+1][curDim] = 1

	x[0] = input.reshape(input_dimension, 1)

	for layer in xrange(L):
		next = layer+1
		dnext, dump = shape(x[next])
		h = dot(transpose(x[layer]), w[next]).reshape(dnext-1, 1)
		if activation == 'softmax':
			normalizer = sum(exp(x[next]))
			x[next] = append(f(h, activation) / normalizer, array([[1]]), axis=0)
		else:	
			x[next] = append(f(h, activation), array([[1]]), axis=0)
	return x[L][:-1]

def validate(valid_input, valid_output, w, layers, activation='tanh'):
	error = 0
	size, dim = shape(valid_input)
	output_dim = shape(valid_output)[1]
	for i in xrange(size):
		y = compute(valid_input[i].reshape(1, dim), w, layers, activation)
		for k in xrange(output_dim):
			if activation == 'softmax':
				error += -1.0 * valid_output[i][k] * log(y[k])
			else:
				error += 0.5*((y[k] - valid_output[i][k])**2)
	return error

def confusion_matrix(test, test_target, w, layers, activation='tanh'):
	output_dimension = shape(test_target)[1]
	matrix = zeros((output_dimension, output_dimension))
	size, input_dimension = shape(test)
	percentage_error = 0;
	for i in xrange(size):
		y = compute(test[i].reshape(1, input_dimension), w, layers, activation=activation)
		max_val = -1e300
		index = -1
		correct = -1
		for j in xrange(output_dimension):
			if y[j] > max_val:
				max_val = y[j]
				index = j
			if test_target[i][j] == 1:
				correct = j
		matrix[correct][index] += 1;
		if correct != index:
			percentage_error += 1
	percentage_error = 1.0 * percentage_error / size * 100
	print matrix
	print 'percentage error: ',percentage_error, '%'
	return matrix