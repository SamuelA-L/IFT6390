import numpy as np
iris_ds = np.genfromtxt('iris.txt')

######## DO NOT MODIFY THIS FUNCTION ########
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)
#############################################


class Q1:

    def feature_means(self, iris): 
        
        return np.mean(iris[:, 0:4], axis=0)

    def covariance_matrix(self, iris):

        return np.cov(iris[:,0:4], rowvar=False)

    def feature_means_class_1(self, iris):
        class1 = np.array([element for element in iris if element[4] == 1])
        
        return self.feature_means(class1)


    def covariance_matrix_class_1(self, iris):
        class1 = np.array([element for element in iris if element[4] == 1])

        return np.cov(class1[:, 0:4], rowvar=False)

class HardParzen:
    def __init__(self, h):
        self.h = h

    def euclidean_dist(self, a, b):

        return np.linalg.norm(np.array(a)-np.array(b))

    def train(self, train_inputs, train_labels):
        self.label_list = np.unique(train_labels)
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.n_classes = len(self.label_list)
        self.n_examples = len(train_inputs)

    def compute_predictions(self, test_data):

        predictions = np.zeros((len(test_data)))
        
        for (i, point) in enumerate(test_data):

            onehot_count = np.zeros((self.n_classes))
            dists = np.array([self.euclidean_dist(a, point) for a in self.train_inputs])
            neighbours = []
            neighbours = [point_index for (point_index, distance) in enumerate(dists) if distance < self.h]

            if len(neighbours) == 0:
                predictions[i] = draw_rand_label(point, self.label_list)
            else:
                for k in neighbours:
                    label = int(self.train_labels[k])
                    onehot_count[label - 1] += 1

                predictions[i] = np.argmax(onehot_count) + 1

        return predictions.astype(int)


class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma = sigma

    def train(self, train_inputs, train_labels):
        self.label_list = np.unique(train_labels)
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.n_classes = len(self.label_list)
        self.n_examples = len(train_inputs)
        self.n_features = len(train_inputs[0])
        
    def euclidean_dist(self, a, b):

        return np.linalg.norm(np.array(a)-np.array(b))

    def rbf(self, a, b):

        D = self.n_features
        o = self.sigma

        return (1/((2*np.pi)**(D/2))*o**D)*np.exp((-(1/2))*((self.euclidean_dist(a, b)**2)/(o**2)))


    def compute_predictions(self, test_data):

        predictions = np.zeros((len(test_data)))
        
        for (i, point) in enumerate(test_data):

            class_scores = np.zeros((self.n_classes))
            scores = np.array([self.rbf(point, a) for a in self.train_inputs])

            for (index, score) in enumerate(scores):
                label = int(self.train_labels[index])
                class_scores[label - 1 ] += score

            predictions[i] = int(np.argmax(class_scores) + 1)

        return predictions.astype(int)


def split_dataset(iris):

    training_set = np.array([sample for (index, sample) in enumerate(iris) if (index % 5) in [0,1,2]])
    validation_set = np.array([sample for (index, sample) in enumerate(iris) if (index % 5) == 3])
    test_set = np.array([sample for (index, sample) in enumerate(iris) if (index % 5) == 4])

    return training_set, validation_set, test_set

def x_y_from_split(iris):
    train, val, test = split_dataset(iris)
    x_train = train[:, 0:-1]
    y_train = train[:, -1]
    x_val = val[:, 0:-1]
    y_val = val[:, -1]
    x_test = test[:, 0:-1]
    y_test = test[:, -1]

    return x_train, y_train, x_val, y_val, x_test, y_test


class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def hard_parzen(self, h):
        classifier = HardParzen(h)
        classifier.train(self.x_train, self.y_train)
        predictions = classifier.compute_predictions(self.x_val)
        total = len(self.y_val)
        good_predictions = np.sum(predictions == self.y_val)

        return 1 - (good_predictions / total)

    def soft_parzen(self, sigma):
        classifier = SoftRBFParzen(sigma)
        classifier.train(self.x_train, self.y_train)
        predictions = classifier.compute_predictions(self.x_val)
        total = len(self.y_val)
        good_predictions = np.sum(predictions == self.y_val)

        return 1 - (good_predictions / total)


def select_params(h_values, o_values, iris):

    x_train, y_train, x_val, y_val, x_test, y_test = x_y_from_split(iris)

    er = ErrorRate(x_train, y_train, x_val, y_val)
    hard_parzen_error = [er.hard_parzen(h) for h in h_values]
    soft_parzen_error = [er.soft_parzen(o) for o in o_values]

    return h_values[np.argmin(hard_parzen_error)],\
           o_values[np.argmin(soft_parzen_error)],\
           hard_parzen_error,\
           soft_parzen_error


def get_test_errors(iris):

    h_values = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]
    o_values = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]

    h_star, o_star, hp_error, sp_error = select_params(h_values, o_values, iris)
    x_train, y_train, x_val, y_val, x_test, y_test = x_y_from_split(iris)
    error = ErrorRate(x_train, y_train, x_test, y_test)
    hard_parzen_test_error = error.hard_parzen(h_star)
    soft_parzen_test_error = error.soft_parzen(o_star)

    return hard_parzen_test_error, soft_parzen_test_error


def random_projections(X, A):

    X_proj = [(1/np.sqrt(2)) * np.matmul(x, A) for x in X]

    return np.array(X_proj)


#report questions

# import matplotlib.pyplot as plt

# Q5 a et b
# h_values = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]
# o_values = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]
# h_star, o_star, hard_parzen_error, soft_parzen_error = select_params(h_values, o_values, iris_ds)
# print(hard_parzen_error)
# print(soft_parzen_error)

# plt.plot(h_values, hard_parzen_error, label="hard parzen")
# plt.plot(o_values, soft_parzen_error, label="soft parzen")
# plt.xlabel("valeurs pour h et sigma")
# plt.ylabel("ratio d'erreur")
# plt.title("Erreurs pour différents paramètres")
# plt.legend()
# plt.show()

# Q9

# x_train, y_train, x_val, y_val, x_test, y_test = x_y_from_split(iris_ds)

# projections = np.empty((500, 90, 2))
# hard_parzen_proj_errors = np.empty((500, 10))
# soft_parzen_proj_errors = np.empty((500, 10))
# for i in range(500):
#     A = np.random.normal(0, 1, 8).reshape(4, 2)
#     train_projection = random_projections(x_train, A)
#     val_projections = random_projections(x_val, A)
#     for j in range(10) :
#         error = ErrorRate(train_projection, y_train, val_projections, y_val)
#         hard_parzen_proj_errors[i][j] = error.hard_parzen(h_values[j])
#         soft_parzen_proj_errors[i][j] = error.soft_parzen(o_values[j])


# hard_parzen_proj_errors_means = np.mean(hard_parzen_proj_errors, axis=0)
# soft_parzen_proj_errors_means = np.mean(soft_parzen_proj_errors, axis=0)

# hard_parzen_proj_errors_std = np.std(hard_parzen_proj_errors, axis=0)
# soft_parzen_proj_errors_std = np.std(soft_parzen_proj_errors, axis=0)

# print(hard_parzen_proj_errors_means)
# print(soft_parzen_proj_errors_means)
# print(hard_parzen_proj_errors_std)
# print(soft_parzen_proj_errors_std)



# [0.65993333 0.41366667 0.23813333 0.17586667 0.15513333 0.14893333
#  0.1702     0.3756     0.66566667 0.66666667]

# [0.16126667 0.1252     0.12553333 0.13093333 0.1374     0.14186667
#  0.16193333 0.17766667 0.18       0.18033333]

# [8.18752167e-02 1.24694738e-01 1.14305944e-01 1.01542678e-01
#  9.70354116e-02 1.00116466e-01 1.12144569e-01 1.84270863e-01
#  1.37072568e-02 1.33226763e-15]

# [0.10110367 0.09720805 0.09528463 0.09703846 0.09896754 0.0997467
#  0.10478463 0.10961093 0.11025224 0.11024165]

# plt.errorbar(o_values, soft_parzen_proj_errors_means, yerr=0.2*soft_parzen_proj_errors_std, label="hard parzen")
# plt.errorbar(h_values, hard_parzen_proj_errors_means, yerr=0.2*hard_parzen_proj_errors_std, label="soft parzen")
# plt.xlabel("valeurs pour h et sigma")
# plt.ylabel("ratio d'erreur")
# plt.title("Erreurs pour différents paramètres")
# plt.legend()
# plt.show()
