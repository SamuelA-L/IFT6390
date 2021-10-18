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


target_names = ['Standard background conditions', 'Tropical cyclone', 'Atmospheric river']
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

x_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1]

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=8)

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

#---decision tree---
decision_tree_classifier = DecisionTreeClassifier(random_state=8)
decision_tree_classifier.fit(x_train, y_train)
predictions = decision_tree_classifier.predict(x_val)
print('Decision Tree :\n', classification_report(y_val, predictions, target_names=target_names))


#---random forest---

random_forest_classifier = RandomForestClassifier(max_depth=12, random_state=8)
random_forest_classifier.fit(x_train, y_train)
predictions = random_forest_classifier.predict(x_val)
print('Random Forest : \n', classification_report(y_val, predictions, target_names=target_names, zero_division=1))
test_predictions = random_forest_classifier.predict(test)
test_predictions_df = pd.DataFrame(test_predictions)
create_submission_csv(test_predictions_df, 'predictions')

