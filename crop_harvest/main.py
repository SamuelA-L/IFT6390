import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, scale
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.pipeline import make_pipeline
import xgboost as xgb


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


# boost = GradientBoostingClassifier(random_state=1)
# train_and_eval(boost, x_train, y_train, x_val, y_val)
# # boost_pca_pipeline = make_pca_pipeline(boost)
# # train_and_eval(boost_pca_pipeline, x_train, y_train, x_val, y_val)
# # create_submission_file(boost_pca_pipeline.predict(x_test).astype(int))

# #find best hyperparams
# for i in [None, 50]:
#     for j in [50, 100, 200]:
#         for k in [1, 10, 50]:
#             print('max_dept : ', i, " n_estimators : ", j, " min sample leaf : ", k)
#             forest = RandomForestClassifier(random_state=1, max_depth=i, n_estimators=j, min_samples_leaf=k)
#             train_and_eval(forest, x_train, y_train, x_val, y_val)

# best found none 200 1
# best found 50 200 1
forest = RandomForestClassifier(random_state=1, max_depth=None, n_estimators=200, min_samples_leaf=1)
train_and_eval(forest, x_train, y_train, x_val, y_val)
# create_submission_file(forest.predict(x_test).astype(int))


# ada_boost = AdaBoostClassifier(random_state=1, n_estimators=250, learning_rate=1.5)
# train_and_eval(ada_boost, x_train, y_train, x_val, y_val)
# create_submission_file(ada_boost.predict(x_test).astype(int))


'''
# find best hyper params
for i in [2, 4, 6, 8, 10]:
    for j in [50, 100, 150, 200]:
        print('max_dept : ', i , " n_estimators : ", j)
        xg_boost = xgb.XGBClassifier(random_state=1, eval_metric='logloss', use_label_encoder=False, n_estimators=j, max_depth=i)
        train_and_eval(xg_boost, x_train, y_train, x_val, y_val)

# best hyperparams found : 8 max_dept, 200 estimators

xg_boost = xgb.XGBClassifier(random_state=1, eval_metric='logloss', use_label_encoder=False, n_estimators=200, max_depth=8)
train_and_eval(xg_boost, x_train, y_train, x_val, y_val)
create_submission_file(xg_boost.predict(x_test).astype(int))
'''
