import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, scale, normalize
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from tensorflow import keras
from sklearn.feature_selection import SelectKBest, chi2
from logistic_reg import MyLogClassifier


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


def train_gauss_naive_bayes(x_train,y_train, priors=None):
    classifier = GaussianNB(priors=priors)
    classifier.fit(x_train, y_train)

    return classifier


def train_logistic_regression(x_train, y_train, max_iter=100) :
    classifier = LogisticRegression(random_state=8, max_iter=max_iter)
    classifier.fit(x_train, y_train)

    return classifier

# data

target_names = ['Standard background conditions', 'Tropical cyclone', 'Atmospheric river']
train = pd.read_csv('train.csv', index_col="S.No")
test = pd.read_csv('test.csv', index_col="S.No")

#-----------------------------------------for 2 classes-----------------------------------------------------------
# train.drop(train[train['LABELS'] == 2].index, inplace=True)


train.drop_duplicates(inplace=True)

x_all = train.iloc[:, :-1].to_numpy()
y_all = train.iloc[:, -1].to_numpy()

x_train, x_val, y_train, y_val = train_test_split(x_all, y_all, test_size=0.2, random_state=8)

x_train = scale(x_train)
x_val = scale(x_val)



train_len = len(y_train)
val_len = len(y_val)
value_train, count_train = np.unique(y_train, return_counts=True)
value_val, count_val = np.unique(y_val, return_counts=True)
print("train : ")
for i in range(count_train.shape[0]):
    print(value_train[i], " : ", count_train[i]/train_len, " % (", count_train[i], ")")

print("val : ")
for i in range(count_val.shape[0]):
    print(value_val[i], " : ", count_val[i]/val_len, " % (", count_val[i], ")")


pca_object = train_PCA(x_train)
x_train_pca = apply_pca(pca_object, x_train)
x_val_pca = apply_pca(pca_object, x_val)
test_pca = apply_pca(pca_object, test)

# '''
#<<<----- My log reg ------>>>
log_classifier = MyLogClassifier()
log_classifier.train(x=x_train, t=y_train, epochs=5000, learning_rate=0.05)
predictions = log_classifier.predict(x_val)
print('My log reg : \n', classification_report(y_val, predictions, zero_division=0))
print(confusion_matrix(y_val, predictions))

test_predictions = log_classifier.predict(test)
test_predictions_df = pd.DataFrame(test_predictions)
create_submission_csv(test_predictions_df, 'predictions')

# '''


'''
#---gaussian naive bayes predictions---

gauss_nb_classifier = train_gauss_naive_bayes(x_train_pca, y_train, count_train/train_len)
predictions = gauss_nb_classifier.predict(x_val_pca)

print(classification_report(y_val, predictions, target_names=target_names))

test_predictions = gauss_nb_classifier.predict(test_pca)
test_predictions_df = pd.DataFrame(test_predictions)
create_submission_csv(test_predictions_df, 'predictions_val')
# '''

# '''
# ---scikit logistic regression---

logistic_classifier = train_logistic_regression(x_train_pca, y_train, max_iter=500)
predictions = logistic_classifier.predict(x_val_pca)

print(classification_report(y_val, predictions))#, target_names=target_names))

# test_predictions = logistic_classifier.predict(test_pca)
# test_predictions_df = pd.DataFrame(test_predictions)
# create_submission_csv(test_predictions_df, 'predictions_')
# '''

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

'''
# ---random forest---

random_forest_classifier = RandomForestClassifier(max_depth=11, random_state=8)
random_forest_classifier.fit(x_train_pca, y_train)
predictions = random_forest_classifier.predict(x_val_pca)
print('Random Forest : \n', classification_report(y_val, predictions, target_names=target_names, zero_division=1))
test_predictions = random_forest_classifier.predict(test_pca)
test_predictions_df = pd.DataFrame(test_predictions)
create_submission_csv(test_predictions_df, 'predictions')
'''


'''
from numpy.random import seed
seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)

n_features = x_train_pca.shape[1]

dnn = keras.Sequential()
dnn.add(keras.layers.Dense(128, activation='relu', input_dim=n_features))
dnn.add(keras.layers.Dense(128, activation='relu'))
dnn.add(keras.layers.Dense(1024, activation='relu'))
dnn.add(keras.layers.Dense(2048, activation='relu'))
dnn.add(keras.layers.Dense(2048, activation='relu'))
dnn.add(keras.layers.Dense(1024, activation='relu'))
dnn.add(keras.layers.Dense(128, activation='relu'))
dnn.add(keras.layers.Dense(1, activation='sigmoid'))

dnn.summary()

learning_rate = 0.005
adam_optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
sgd_optimizer = keras.optimizers.SGD(learning_rate=learning_rate)


dnn.compile(
    optimizer=adam_optimizer,
    loss='categorical_crossentropy',
    metrics=['Precision', 'Accuracy']
)

# callback = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.001, patience=15,  restore_best_weights=True, mode='auto')

dnn.fit(x_train_pca, y_train, epochs=5, batch_size=128)#, callbacks=[callback])
predictions = dnn.predict(x_val_pca)
print(np.unique(predictions))

print('Dnn : \n', classification_report(y_val, predictions, target_names=target_names, zero_division=0))
print(confusion_matrix(y_val, predictions))

test_predictions = dnn.predict(test_pca)
test_predictions_df = pd.DataFrame(test_predictions)
create_submission_csv(test_predictions_df, 'predictions')
'''

