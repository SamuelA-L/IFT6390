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
                
                predictions[i] = int(np.argmax(onehot_count) + 1) 

        return predictions


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

        return (1/((2*np.pi)**(self.n_features/2))*self.sigma**self.n_features)*np.exp((-1/2)*((self.euclidean_dist(a, b)**2)/(self.sigma**2)))


    def compute_predictions(self, test_data):
        predictions = np.empty((len(test_data)))
        
        for (i, point) in enumerate(test_data) :

            onehot_count = np.zeros((self.n_classes))
            scores = np.array([self.rbf(a,point) for a in self.train_inputs ])
            print(scores)
            # neighbours = []
            # neighbours = [ point_index for (point_index, distance) in enumerate(dists) if distance < self.h ]

            # if len(neighbours) == 0 :
            #     predictions[i] = draw_rand_label(point, self.label_list)
            # else :
            #     for k in neighbours :
            #         onehot_count[self.train_labels[k]-1] += 1
                
            predictions[i] = int(self.train_labels[np.argmax(scores)]) 

        return predictions

train = [[1,1], [2,2], [3,3], [4,4]]
labels = [1,2,3,4]

s_par = SoftRBFParzen(2)
s_par.train(train, labels)
pred = s_par.compute_predictions(train)
print(pred)

def split_dataset(iris):
    pass


class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def hard_parzen(self, h):
        pass

    def soft_parzen(self, sigma):
        pass


def get_test_errors(iris):
    pass


def random_projections(X, A):
    pass


