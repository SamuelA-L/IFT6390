import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, scale
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline



def create_submission_file(predictions, name='predictions'):
    predictions_df = pd.DataFrame(predictions)
    submission_file = open("./" + name + ".csv", "w")
    predictions_df.index.name = 'S.No'
    predictions_df.columns = ['LABELS']
    predictions_df.to_csv(submission_file, index=True)
    submission_file.close()

def train_and_eval(model, x_train, y_train, x_val, y_val) :
    model.fit(x_train, y_train)
    predictions = model.predict(x_val)
    print(classification_report(y_true=y_val, y_pred=predictions, target_names=target_names))

def make_pca_pipeline(model):
    return make_pipeline(StandardScaler(), PCA(n_components='mle'), model)

target_names = ['Non-crop land', 'Crop land']
train = pd.read_csv('train.csv', index_col="Unnamed: 0")
test = pd.read_csv('test_nolabels.csv', index_col="S.No")

x_test = test.to_numpy()
x = train.iloc[:, :-1].to_numpy()
y = train.iloc[:, -1].to_numpy()
counts = np.unique(y, return_counts=True)
print('class distribution on training data \n', int(counts[0][0]), ' : ', counts[1][0], '  |  ', int(counts[0][1]), ' : ', counts[1][1])
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)



# boost = GradientBoostingClassifier()
# boost_pca_pipeline = make_pca_pipeline(boost)
# train_and_eval(boost_pca_pipeline, x_train, y_train, x_val, y_val)
# create_submission_file(boost_pca_pipeline.predict(x_test).astype(int))


forest = RandomForestClassifier()
train_and_eval(forest, x_train, y_train, x_val, y_val)
create_submission_file(forest.predict(x_test).astype(int))
