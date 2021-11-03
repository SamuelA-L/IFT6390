import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, scale
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from tensorflow import keras
from logistic_reg import MyLogClassifier
from sklearn.svm import SVC


def create_submission_csv(predictions_df, name):
    submission_file = open("./" + name + ".csv", "w")
    predictions_df.index.name = 'S.No'
    predictions_df.columns = ['LABELS']
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



# data

target_names = ['Standard background conditions', 'Tropical cyclone', 'Atmospheric river']
train = pd.read_csv('train.csv', index_col="S.No")
test = pd.read_csv('test.csv', index_col="S.No")

'''
# train.drop_duplicates(inplace=True)
'''

x_all = train.iloc[:, :-1].to_numpy()
y_all = train.iloc[:, -1].to_numpy()

x_train, x_val, y_train, y_val = train_test_split(x_all, y_all, test_size=0.2, random_state=8)


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


def my_logistic_reg():

    log_classifier = MyLogClassifier()
    log_classifier.train(x=x_train, t=y_train, epochs=10000, learning_rate=0.99)
    predictions = log_classifier.predict(x_val)
    print('My log reg : \n', classification_report(y_val, predictions, zero_division=0))
    print(confusion_matrix(y_val, predictions))

    test_predictions = log_classifier.predict(test)
    test_predictions_df = pd.DataFrame(test_predictions)
    create_submission_csv(test_predictions_df, 'predictions')


def gaussian_naive_bayes():

    gauss_nb_classifier = train_gauss_naive_bayes(scale(x_train), y_train, count_train/train_len)
    predictions = gauss_nb_classifier.predict(scale(x_val))
    print(classification_report(y_val, predictions, target_names=target_names))

    test_predictions_gnb = gauss_nb_classifier.predict(scale(test))
    test_predictions_df = pd.DataFrame(test_predictions_gnb)
    create_submission_csv(test_predictions_df, 'predictions_val')

    return test_predictions_gnb


def scikit_logistic_reg():

    logistic_classifier = LogisticRegression(random_state=8, max_iter=500)
    logistic_classifier.fit(scale(x_train), y_train)
    predictions = logistic_classifier.predict(scale(x_val))
    print(classification_report(y_val, predictions))

    test_predictions_lc = logistic_classifier.predict(scale(test))
    test_predictions_df = pd.DataFrame(test_predictions_lc)
    create_submission_csv(test_predictions_df, 'predictions')
    print(confusion_matrix(y_val, predictions))

    return test_predictions_lc


def decision_tree():

    decision_tree_classifier = DecisionTreeClassifier(random_state=8)
    decision_tree_classifier.fit(x_train, y_train)
    predictions = decision_tree_classifier.predict(x_val)
    print('Decision Tree :\n', classification_report(y_val, predictions, target_names=target_names))

    test_predictions = decision_tree_classifier.predict(test)
    test_predictions_df = pd.DataFrame(test_predictions)
    create_submission_csv(test_predictions_df, 'predictions')

    return test_predictions


def random_forest():

    random_forest_classifier = RandomForestClassifier(max_depth=12, random_state=8, n_estimators=100, min_samples_leaf=100)
    random_forest_classifier.fit(scale(x_train), y_train)
    predictions = random_forest_classifier.predict(scale(x_val))
    print('Random Forest : \n', classification_report(y_val, predictions, target_names=target_names, zero_division=1))

    test_predictions_rf = random_forest_classifier.predict(scale(test))
    test_predictions_df = pd.DataFrame(test_predictions_rf)
    create_submission_csv(test_predictions_df, 'predictions')

    return test_predictions_rf


def gradient_boosting() :

    grad_boost_classifier = GradientBoostingClassifier(max_depth=8, random_state=8, n_estimators=300, learning_rate=0.1, min_samples_leaf=5)
    grad_boost_classifier.fit(x_train, y_train)
    predictions = grad_boost_classifier.predict(x_val)
    print('gradient boosting : \n', classification_report(y_val, predictions, target_names=target_names, zero_division=1))

    test_predictions_gb = grad_boost_classifier.predict(test)
    test_predictions_gb = pd.DataFrame(test_predictions_gb)
    create_submission_csv(test_predictions_gb, 'predictions')

    return test_predictions_gb


def dnn() :
    from numpy.random import seed
    seed(1)
    # from tensorflow import set_random_seed
    # set_random_seed(2)

    n_features = x_train_pca.shape[1]

    dnn = keras.Sequential()
    dnn.add(keras.layers.Dense(128, activation='relu', input_dim=n_features))
    dnn.add(keras.layers.Dense(1024, activation='relu'))
    dnn.add(keras.layers.Dense(2048, activation='relu'))
    # dnn.add(keras.layers.Dense(2048, activation='relu'))
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


def svm():

    svm_classifier = SVC(decision_function_shape='ovo')
    svm_classifier.fit(scale(x_train), y_train)
    predictions = svm_classifier.predict(scale(x_val))
    print('SVM : \n', classification_report(y_val, predictions, zero_division=0))
    print(confusion_matrix(y_val, predictions))

    test_predictions_svm = svm_classifier.predict(scale(test))
    test_predictions_df = pd.DataFrame(test_predictions_svm)
    create_submission_csv(test_predictions_df, 'predictions')

    return test_predictions_svm


def combine_predictions():

    test_predictions_gnb = gaussian_naive_bayes()
    test_predictions_lc = scikit_logistic_reg()
    test_predictions_rf = random_forest()
    test_predictions_svm = svm()
    comb_pred = np.zeros((len(test_predictions_gnb), 3))
    for i in range(len(test_predictions_gnb)):
        comb_pred[i][test_predictions_gnb[i]] += 1
        comb_pred[i][test_predictions_lc[i]] += 1
        comb_pred[i][test_predictions_rf[i]] += 1
        comb_pred[i][test_predictions_svm[i]] += 1

    predictions = np.argmax(comb_pred, axis=1)
    test_predictions_df = pd.DataFrame(predictions)
    create_submission_csv(test_predictions_df, 'predictions')
