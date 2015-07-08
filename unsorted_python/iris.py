from mlp import *

def preprocessIris(infile, outfile):
	stext1 = 'Iris-setosa'
	stext2 = 'Iris-versicolor'
	stext3 = 'Iris-virginica'
	rtext1 = '0'
	rtext2 = '1'
	rtext3 = '2'

	fid = open(infile, "r")
	oid = open(outfile, "w")

	for s in fid:
		if s.find(stext1)>-1:
			oid.write(s.replace(stext1,rtext1))
		elif s.find(stext2)>-1:
			oid.write(s.replace(stext2,rtext2))
		elif s.find(stext3)>-1:
			oid.write(s.replace(stext3,rtext3))
	fid.close()
	oid.close()

preprocessIris('iris.data', 'iris_proc.data')
iris = loadtxt('iris_proc.data', delimiter=',')
iris[:, :4] = iris[:, :4] - iris[:, :4].mean(axis=0)
imax = iris.max(axis=0).reshape(1,5)
iris[:, :4] = iris[:,:4] / imax[:,:4]

target = zeros((shape(iris)[0], 3))
indices = where(iris[:,4]==0)
target[indices,0] = 1
indices = where(iris[:,4]==1)
target[indices,1] = 1
indices = where(iris[:,4]==2)
target[indices,2] = 1
order = range(shape(iris)[0])
random.shuffle(order)
iris = iris[order, :]
target = target[order, :]

traindata = iris[::2, 0:4]
traintarget = target[::2]
valid = iris[1::4, 0:4]
validtarget = target[1::4]
test = iris[3::4,0:4]
testtarget = target[3::4]

layers = [5,3]

w = train(traindata, traintarget, layers, eta=0.01, error_margin=0.005, activation='softmax', valid_input=valid, valid_output=validtarget, print_error=True)
m = confusion_matrix(test, testtarget, w, layers, activation='softmax')
