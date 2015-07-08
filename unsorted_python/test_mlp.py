from mlp import *

# testing on XOR function

# data = array([[0,0], [0,1], [1,0], [1,1]])
# target = array([[0], [1], [1], [0]])
# layers = [3,1]
# w = train(data, target, layers, eta=0.25, iter=3000)
# print 'MLP training on XOR function with using one hidden layer having 3 weights'
# print 'weights results:'
# for weight in w:
# 	print weight
# print 'testing for in sample error:'
# for i in xrange(4):
# 	val = compute(data[i].reshape(1,2), w, layers)
# 	print val, ' againts ', target[i]


print '\nMLP regression training'
x = linspace(0,1,40).reshape(40,1)
t = sin(2*pi*x) + cos(4*pi*x) + (random.randn(40)*0.2).reshape(40,1)

traindata = x[0::2,:]
test = x[1::4,:]
valid = x[3::4,:]
traintarget = t[0::2,:]
testtarget = t[1::4,:]
validtarget = t[3::4,:]

#for k in xrange(10):
k = 10
layer = [k+1,1]
w = train(traindata, traintarget, layer, activation='linear', eta=0.025, valid_input=valid, valid_output=validtarget, error_margin=0.0001, print_error=True)
print 'result of training using ', k+1,' nodes in the hidden layer:'
L = 1
for weight in w:
	print 'layer ', L, ':'
	print weight
	L += 1
error = validate(test, testtarget, w, layer, activation='linear')
print "out-of-sample error rate: ", error

