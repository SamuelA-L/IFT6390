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


    def get_y(self, x):
        return self.sigmoid(np.dot(x, self.weights) + self.bias)

    def train(self, x, t, learning_rate=0.01, epochs=500, min_loss=0):
        nb_examples, nb_features = x.shape
        n_classes = len(np.unique(t))
        self.bias = np.zeros(n_classes)
        self.weights = np.zeros((nb_features, n_classes))

        for i in range(epochs):
            y = self.get_y(x)
            # for i in range(y.shape[1])
            weight_gradient_0, bias_gradient_0 = self.gradient(x=x, t=t, y=y[:, 0])
            weight_gradient_1, bias_gradient_1 = self.gradient(x=x, t=t, y=y[:, 1])
            # weight_gradient_2, bias_gradient_2 = self.gradient(x=x, t=t, y=y[:,2]))

            self.weights[0] -= learning_rate * weight_gradient_0
            self.weights[1] -= learning_rate * weight_gradient_1
            self.bias[0] -= learning_rate * bias_gradient_0
            self.bias[1] -= learning_rate * bias_gradient_1

            y = self.get_y(x)
            loss_0 = self.crossentropy(t=t, y=y[:, 0])
            loss_1 = self.crossentropy(t=t, y=y[:, 1])
            loss = (loss_1 + loss_0) / 2

            if min_loss >= loss:
                break

            print('epoch : ', i,  ' ------> Loss : ', loss)

    def predict(self, x, treshold=0.5):
        predictions = np.empty((len(x)))
        y = self.get_y(x)

        for i, pred in enumerate(y):
            best_class = np.argmax(pred)
            predictions[i] = 1 if (pred[best_class] >= treshold) else 0

        return predictions


from sklearn.datasets import make_moons
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


X, y = make_moons(n_samples=500, noise=0.1)

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=8)

log_classifier = MyLogClassifier()
log_classifier.train(x=x_train, t=y_train, epochs=1000, learning_rate=0.5)
predictions = log_classifier.predict(x_val)
print('My log reg : \n', classification_report(y_val, predictions, zero_division=0))
print(confusion_matrix(y_val, predictions))


scikit_classifier = LogisticRegression(random_state=8, max_iter=500)
scikit_classifier.fit(x_train, y_train)
predictions_scikit = scikit_classifier.predict(x_val)
print('scickit log reg : \n', classification_report(y_val, predictions_scikit, zero_division=0))
print(confusion_matrix(y_val, predictions_scikit))



