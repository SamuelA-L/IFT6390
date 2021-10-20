import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
'''
this code was written with the help of :
-https://machinelearningmastery.com/implement-logistic-regression-stochastic-gradient-descent-scratch-python/
-https://medium.com/@martinpella/logistic-regression-from-scratch-in-python-124c5636b8ac
-https://towardsdatascience.com/logistic-regression-from-scratch-in-python-ec66603592e2
'''

class MyLogClassifier :
    def __init__(self):
        self.weights = None
        self.bias = None

    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    @staticmethod
    def crossentropy(t, y):
        return (-np.sum(t * np.log(y) + (1-t) * np.log(1-y))) / len(y)# TODO sum or mean ?
        # return np.mean(- (t * np.log(y) + (1-t) * np.log(1-y)))
        # return -np.mean(t*(np.log(y)) - (1-t)*np.log(1-y))

    @staticmethod
    def gradient(x, t, y):
        nb_examples = len(x)

        weight_gradient = np.dot(x.T, (y-t)) / nb_examples
        bias_gradient = np.sum((y-t)) / nb_examples

        return weight_gradient, bias_gradient


    def get_t(self, x):
        return self.sigmoid(np.dot(x, self.weights) + self.bias)

    def train(self, x, y, learning_rate=0.01, epochs=500, min_loss=0):
        nb_examples, nb_features = x.shape
        self.bias = 0
        self.weights = np.zeros(nb_features)

        for i in range(epochs):
            t = self.get_t(x)
            weight_gradient, bias_gradient = self.gradient(x, t, y)

            self.weights = learning_rate * weight_gradient
            self.bias = learning_rate * bias_gradient

            t = self.get_t(x)
            loss = self.crossentropy(y, t)

            if min_loss >= loss:
                break

            print('epoch : ', i,  ' ------> Loss : ', loss)

    def predict(self, x, treshold=0.5):
        predictions = np.empty((len(x)))
        t = self.get_t(x)

        for i, pred in enumerate(t):
            predictions[i] = 1 if (pred >= treshold) else 0

        return predictions


from sklearn.datasets import make_moons
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


X, y = make_moons(n_samples=100, noise=0.1)

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=8)

log_classifier = MyLogClassifier()
log_classifier.train(x_train, y_train, epochs=100, learning_rate=0.005)
predictions = log_classifier.predict(x_val)
print('My log reg : \n', classification_report(y_val, predictions, zero_division=0))
print(confusion_matrix(y_val, predictions))


scikit_classifier = LogisticRegression(random_state=8, max_iter=500)
scikit_classifier.fit(x_train, y_train)
predictions_scikit = scikit_classifier.predict(x_val)
print('scickit log reg : \n', classification_report(y_val, predictions_scikit, zero_division=0))
print(confusion_matrix(y_val, predictions_scikit))



