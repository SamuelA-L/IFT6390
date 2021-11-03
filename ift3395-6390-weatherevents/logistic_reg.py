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

        return (-np.sum(t * np.log(y) + (1-t) * np.log(1-y))) / len(y)

    @staticmethod
    def crossentropy(t, y):
        loss = 0
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

    def standardize(self, x):
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
