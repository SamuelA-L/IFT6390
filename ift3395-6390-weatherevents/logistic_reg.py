import numpy as np
import pandas as pd
from scipy import sparse

'''
this code was written with the help of :
-https://machinelearningmastery.com/implement-logistic-regression-stochastic-gradient-descent-scratch-python/
-https://medium.com/@martinpella/logistic-regression-from-scratch-in-python-124c5636b8ac
-https://towardsdatascience.com/logistic-regression-from-scratch-in-python-ec66603592e2
'''

class MyLogReg :
    def __init__(self):
        self.weights = None
        self.bias = None

    @staticmethod
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    # TODO : see difference with < -np.mean(y*(np.log(y_hat)) - (1-y)*np.log(1-y_hat)) >
    @staticmethod
    def crossentropy(self, t, y):
        return np.mean(- (t * np.log(y) + (1-t) * np.log(1-y)))

    @staticmethod
    def gradient(self, x, t, y):
        nb_examples = len(x)

        weight_gradient = np.dot(x.T, (y-t)) / nb_examples
        bias_gradient = np.sum((y-t)) / nb_examples

        return weight_gradient, bias_gradient

    @staticmethod
    def get_t(self, x):
        return self.sigmoid(np.dot(x, self.weights) + self.bias)


    def train(self, x, y, learning_rate=0.01, epochs=500, min_loss=0):
        nb_examples, nb_features = x.shape
        self.bias = 0
        self.weights = np.zeros((nb_features))


        for i in range(epochs) :
            t = self.sigmoid(np.dot(x, self.weights) + self.bias)
            weight_gradient, bias_gradient = self.gradient(x, t, y)

            self.weights = learning_rate * weight_gradient
            self.bias = learning_rate * bias_gradient

            t = self.get_t(x)
            loss = self.crossentropy(t, t, y)

            if min_loss >= loss:
                break

            print('epcoh : ', i,  ' ------> Loss : ', loss)


    def predict(self, x, treshold):
        predictions = np.empty((len(x)))
        t = self.get_t(x)

        for i, pred in enumerate(t):
            predictions[i] = 1 if (pred >= treshold) else 0

        return predictions






