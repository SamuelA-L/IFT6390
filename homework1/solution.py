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
        
        return np.mean(iris[:,0:4], axis=0)

    def covariance_matrix(self, iris):

        return np.cov(iris[:,0:4], rowvar=False)

    def feature_means_class_1(self, iris):
        class1 = np.array([element for element in iris if element[4] == 1])
        
        return self.feature_means(class1)


    def covariance_matrix_class_1(self, iris):
        class1 = np.array([element for element in iris if element[4] == 1])

        return np.cov(class1[:,0:4], rowvar=False)

class HardParzen:
    def __init__(self, h):
        self.h = h

    def euclidean_dist(self, a, b) :

        return np.linalg.norm(np.array(a)-np.array(b))

    def train(self, train_inputs, train_labels):
        self.label_list = np.unique(train_labels)
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.n_classes = len(self.label_list)
        self.n_examples = len(train_inputs)

    def compute_predictions(self, test_data):

        predictions = np.empty((len(test_data)))
        
        for (i, point) in enumerate(test_data) :

            onehot_count = np.zeros((self.n_classes))
            dists = np.array([self.euclidean_dist(a,point) for a in self.train_inputs ])
            neighbours = []
            neighbours = [ point_index for (point_index, distance) in enumerate(dists) if distance < self.h ]

            if len(neighbours) == 0 :
                predictions[i] = draw_rand_label(point, self.label_list)
            else :
                for k in neighbours :
                    onehot_count[self.train_labels[k]-1] += 1
                
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
        
    def euclidean_dist(self, a, b) :

        return np.linalg.norm(np.array(a)-np.array(b))

    def rbf(self, a, b) :

        D = self.n_features
        o = self.sigma

        return (1/((2*np.pi)**(D/2))*o**D)*np.exp((-(1/2))*((self.euclidean_dist(a, b)**2)/(o**2)))


    def compute_predictions(self, test_data):

        predictions = np.empty((len(test_data)))
        
        for (i, point) in enumerate(test_data) :

            # onehot_count = np.zeros((self.n_classes))
            scores = np.array([self.rbf(point, a) for a in self.train_inputs ])
            # neighbours = []
            # neighbours = [ point_index for (point_index, distance) in enumerate(dists) if distance < self.h ]

            # if len(neighbours) == 0 :
            #     predictions[i] = draw_rand_label(point, self.label_list)
            # else :
            #     for k in neighbours :
            #         onehot_count[self.train_labels[k]-1] += 1
                
            predictions[i] = self.train_labels[np.argmax(scores)]

        return predictions.astype(int)

# train = [[1,1,1], [2,2,2], [3,3,3], [4,4,4]]
# labels = [4,3,2,1]

# s_par = SoftRBFParzen(2)
# s_par.train(train, labels)
# pred = s_par.compute_predictions(train)
# print(pred)


def split_dataset(iris):

    training_set = np.array( [sample for (index, sample) in enumerate(iris) if (index % 5) in [0,1,2]])
    validation_set = np.array( [sample for (index, sample) in enumerate(iris) if (index % 5) == 3])
    test_set = np.array( [sample for (index, sample) in enumerate(iris) if (index % 5) == 4])

    return training_set, validation_set, test_set


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


def get_test_errors(iris):
    pass


def random_projections(X, A):
    pass




h_values = [0.01,0.1,0.2,0.3,0.4,0.5,1.0,3.0,10.0,20.0]
o_values = [0.01,0.1,0.2,0.3,0.4,0.5,1.0,3.0,10.0,20.0]

train, val, test = split_dataset(iris_ds)
x_train = train[:,0:-1]
y_train = train[:,-1]
x_val = val[:,0:-1]
y_val = val[:,-1]

er = ErrorRate(x_train, y_train, x_val ,y_val )
hard_parzen_error = np.array( [ er.hard_parzen(h) for h in h_values ] )
# soft_parzen_error =  np.array( [ er.soft_parzen(o) for o in o_values ] )