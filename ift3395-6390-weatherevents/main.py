import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, scale, normalize
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from tensorflow import keras


def create_submission_csv(predictions_df, name):
    submission_file = open("./" + name + ".csv", "w")
    predictions_df.to_csv(submission_file, index=True)
    submission_file.close()


def scale(data):
    scaler = StandardScaler()
    scaler.fit(data)
    scaled_data = scaler.transform(data)

    return scaled_data


def train_PCA(data):
    scaled_data = scale(data)
    pca_object = PCA(n_components='mle')
    pca_object.fit(scaled_data)

    return pca_object


def apply_pca(trained_pca : PCA, data):
    scaled_data = scale(data)
    pc_data = trained_pca.transform(scaled_data)

    return pc_data


def train_gauss_naive_bayes(x_train,y_train):
    classifier = GaussianNB()
    classifier.fit(x_train, y_train)

    return classifier


def train_logistic_regression(x_train, y_train, max_iter=100) :
    classifier = LogisticRegression(random_state=8, max_iter=max_iter)
    classifier.fit(x_train, y_train)

    return classifier

# data

target_names = ['Standard background conditions', 'Tropical cyclone', 'Atmospheric river']
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

x_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1]

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=8)

train_len = len(y_train)
val_len = len(y_val)
value_train, count_train = np.unique(y_train, return_counts=True)
value_val, count_val = np.unique(y_val, return_counts=True)
print("train : ")
for i in range(count_train.shape[0]):
    print(value_train[i], " : ", count_train[i]/train_len, " %")

print("val : ")
for i in range(count_val.shape[0]):
    print(value_val[i], " : ", count_val[i]/val_len, " %")



pca_object = train_PCA(x_train)
x_train_pca = apply_pca(pca_object, x_train)
x_val_pca = apply_pca(pca_object, x_val)
test_pca = apply_pca(pca_object, test)

'''
#---gaussian naive bayes predictions---

gauss_nb_classifier = train_gauss_naive_bayes(x_train_pca, y_train)
predictions = gauss_nb_classifier.predict(x_val_pca)

print(classification_report(y_val, predictions, target_names=target_names))

test_predictions = gauss_nb_classifier.predict(test_pca)
test_predictions_df = pd.DataFrame(test_predictions)
create_submission_csv(test_predictions_df, 'predictions_val')

'''
'''
#---scikit logistic regression---

logistic_classifier = train_logistic_regression(x_train_pca, y_train, max_iter=300)
predictions = logistic_classifier.predict(x_val_pca)

print(classification_report(y_val, predictions, target_names=target_names))

test_predictions = logistic_classifier.predict(test_pca)
test_predictions_df = pd.DataFrame(test_predictions)
create_submission_csv(test_predictions_df, 'predictions_')
'''

'''
#---decision tree---

decision_tree_classifier = DecisionTreeClassifier(random_state=8)
decision_tree_classifier.fit(x_train, y_train)
predictions = decision_tree_classifier.predict(x_val)
print('Decision Tree :\n', classification_report(y_val, predictions, target_names=target_names))
test_predictions = decision_tree_classifier.predict(test)
test_predictions_df = pd.DataFrame(test_predictions)
create_submission_csv(test_predictions_df, 'predictions')
'''


#---random forest---

random_forest_classifier = RandomForestClassifier(max_depth=15, random_state=8, criterion='entropy')
random_forest_classifier.fit(x_train_pca, y_train)
predictions = random_forest_classifier.predict(x_val_pca)
print('Random Forest : \n', classification_report(y_val, predictions, target_names=target_names, zero_division=1))
test_predictions = random_forest_classifier.predict(test_pca)
test_predictions_df = pd.DataFrame(test_predictions)
create_submission_csv(test_predictions_df, 'predictions')



'''
from numpy.random import seed
seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)

n_features = x_train_pca.shape[1]

dnn = keras.Sequential()
dnn.add(keras.Input(shape=(n_features,)))
dnn.add(keras.layers.Dense(128, activation='relu'))
dnn.add(keras.layers.Dense(256, activation='relu'))
dnn.add(keras.layers.Dense(1, activation='sigmoid'))
dnn.summary()

adam_optimizer = keras.optimizers.Adam(learning_rate=0.001)
sgd_optimizer = keras.optimizers.SGD(learning_rate=0.001)


dnn.compile(
    optimizer=adam_optimizer,
    loss='categorical_crossentropy',
    metrics=['Precision', 'Accuracy']
)

# callback = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.001, patience=15,  restore_best_weights=True, mode='auto')

dnn.fit(x_train_pca, y_train, epochs=50, batch_size=16)#, callbacks=[callback])
predictions = dnn.predict(x_val_pca)
print('Dnn : \n', classification_report(y_val, predictions, target_names=target_names, zero_division=1))
'''

