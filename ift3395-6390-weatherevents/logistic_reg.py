import numpy as np

class MyLogClassifier :
    def __init__(self):
        self.weights = None
        self.bias = None

    @staticmethod
    def sigmoid(x):

        return 1/(1+np.exp(-x))

    @staticmethod
    def binary_crossentropy(t, y):

        return (-np.sum(t * np.log(y) + (1-t) * np.log(1-y))) / len(y)# TODO sum or mean ?


    @staticmethod
    def crossentropy(t, y):
        loss = 0
        n_classes = y.shape[1]
        for i, t_i in enumerate(t):
            loss -= (np.log(y[i, t_i]))

        loss = loss / len(t)
        return loss


    @staticmethod
    def gradient(x, t, y):

        nb_examples = len(x)
        weight_gradient = np.dot(x.T, (y-t)) / nb_examples
        bias_gradient = np.sum((y-t)) / nb_examples

        return weight_gradient, bias_gradient


    def get_y(self, x):

        return self.sigmoid(np.dot(x, self.weights.T) + self.bias)


    def standardize(self,x):
        x = np.asarray(x)
        m, n = x.shape
        new_x = np.empty((m, n))

        for i in range(n):
            mean = np.mean(x[:, i])
            sigma = np.std(x[:, i])
            for j in range(m):
                new_x[j][i] = (x[j][i] - mean) / sigma

        return new_x

    def train(self, x, t, learning_rate=0.01, epochs=500, min_loss=0):

        x = self.standardize(x)
        nb_examples, nb_features = x.shape
        n_classes = len(np.unique(t))
        self.bias = np.random.randn(n_classes)
        self.weights = np.random.rand(n_classes, nb_features)
        t_c = np.empty((n_classes, len(t)))

        for i in range(n_classes):
            for j, t_j in enumerate(t):
                t_c[i][j] = 1 if t_j == i else 0

        for epoch in range(epochs):
            y = self.get_y(x)

            for i in range(n_classes):
                w_gradient, b_gradient = self.gradient(x=x, t=t_c[i], y=y[:, i])
                self.weights[i] -= learning_rate * w_gradient
                self.bias[i] -= learning_rate * b_gradient

            y = self.get_y(x)
            loss = 0

            for i in range(n_classes):
                loss += self.binary_crossentropy(t=t_c[i], y=y[:, i])

            loss = loss / n_classes

            if min_loss >= loss:
                break
            print('epoch : ', epoch,  ' ------> Loss : ', loss)

    def predict(self, x):

        x = self.standardize(x)
        predictions = np.empty((len(x)), dtype=int)
        y = self.get_y(x)

        for i, pred in enumerate(y):
            predictions[i] = np.argmax(pred)

        return predictions


# functions to test binary and multiclass logistic regression on smaller and simpler datasets (debugging)

from sklearn.datasets import make_moons, load_iris
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


X, y = make_moons(n_samples=10000, noise=0.1, random_state=8)
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=8)

data = load_iris()
X = data['data']
y = data['target']
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=8)

def mine():

    log_classifier = MyLogClassifier()
    log_classifier.train(x=x_train, t=y_train, epochs=5000, learning_rate=0.3)
    predictions = log_classifier.predict(x_val)
    print('My log reg : \n', classification_report(y_val, predictions, zero_division=0))
    print(confusion_matrix(y_val, predictions))
    print(predictions)

def scikit():
    from sklearn.linear_model import LogisticRegression
    scikit_classifier = LogisticRegression(random_state=8, max_iter=1000)
    scikit_classifier.fit(x_train, y_train)
    predictions_scikit = scikit_classifier.predict(x_val)
    print('scickit log reg : \n', classification_report(y_val, predictions_scikit, zero_division=0))
    print(confusion_matrix(y_val, predictions_scikit))


    print(predictions_scikit)


# mine()
# scikit()