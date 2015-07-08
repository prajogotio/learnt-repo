from svm_smo import *
from numpy import random
from matplotlib import *
inputData = 5*random.randn(100,2)
targetData = []
for i in xrange(100):
	(x, y) = inputData[i]
	r = sqrt((x) * (x)+ (y) * (y))
	t = 1 if r > 5.2 else -1
	targetData.append(t)

targetData = array(targetData).reshape(100,1)

# scatter(inputData[:,0], inputData[:,1],c=targetData)
# show()



print "svm training start"

svm = SVM()
svm.kernel.setGamma(0.2)
svm.setTrainingData(inputData, targetData)
svm.train()

print "done"

print "confusion matrix"

cm = zeros((2,2))

for i in xrange(100):
	t = svm.classify(inputData[i])
	j = 0 if targetData[i] > 0 else 1
	k = 0 if t > 0 else 1
	cm[j][k] += 1

print cm

accuracy = (cm[0][0] + cm[1][1])

print "accuracy : ", accuracy

