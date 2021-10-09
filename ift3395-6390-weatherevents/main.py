import numpy as np

train = np.genfromtxt('train.csv', delimiter=',')
test = np.genfromtxt('test.csv', delimiter=',')

print(train)
print(test)