from elasticnet import *
from numpy import random

plot = []
label = []

data = []
target = []
for i in xrange(50):
	y = -3 * i + 5 + random.random() * 100
	data.append([1, i])
	target.append([y])
	plot.append([i, y])
	label.append([0])

data = array(data)
target = array(target)
print target

linreg = LinearRegression(0.5, 0.6)
linreg.setSample(data, target)
linreg.train()

print linreg.weight

ans = []

for i in xrange(50):
	y = linreg.regress([1, i])
	ans.append([y])
	plot.append([i, y])
	label.append([1])

plot = array(plot)
label = array(label)
scatter(plot[:,0], plot[:,1], c=label);
show()



