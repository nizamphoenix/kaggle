# LightGBM--Cross Validation implementation

from sklearn.model_selection import KFold, train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
import gc

import lightgbm as lgb


from tensorflow.keras.utils import Sequence
fnc_df = pd.read_csv("../input/trends-assessment-prediction/fnc.csv")
loading_df = pd.read_csv("../input/trends-assessment-prediction/loading.csv")
labels_df = pd.read_csv("../input/trends-assessment-prediction/train_scores.csv")

fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])
df = fnc_df.merge(loading_df, on="Id")
labels_df["is_train"] = True

df = df.merge(labels_df, on="Id", how="left")

target_cols = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']

#imputing missing values in targets
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors = 5, weights="distance")
df[target_cols] = pd.DataFrame(imputer.fit_transform(df[target_cols]), columns = target_cols)

test_df = df[df["is_train"] != True].copy()
train_df = df[df["is_train"] == True].copy()

#y_train_df = train_df[target_cols]
train_df = train_df.drop(['is_train'], axis=1)
#train_df = train_df.drop(target_cols + ['is_train'], axis=1)
test_df = test_df.drop(target_cols+['is_train'], axis=1)

FNC_SCALE = 1/500
test_df[fnc_features] *= FNC_SCALE
train_df[fnc_features] *= FNC_SCALE
print(train_df.shape,test_df.shape)
print("Train and test dataframes contain Id columns,too!!")


def my_metric(y_true, y_pred):
    return np.mean(np.sum(np.abs(y_true - y_pred), axis=0)/np.sum(y_true, axis=0))

params = {'lambda_l1': 2.7318340828772203e-05,
 'lambda_l2': 0.04505123541570556,
 'num_leaves': 77,
 'feature_fraction': 0.8999999999999999,
 'bagging_fraction': 0.9508895705450257,
 'bagging_freq': 5,
 'min_child_samples': 20,
 'objective': 'gamma',
 'metric': 'l1',
 'boosting_type': 'gbdt',
 'learning_rate': 0.01,
 'max_depth': -1,
 'num_threads': 8,
 'seed': 0}


NFOLDS = 7
from sklearn.model_selection import KFold
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=0)

targets=['age','domain1_var1','domain1_var2', 'domain2_var1','domain2_var2']
features=list(set(train_df.columns)-set(targets)-set(['Id']))
overal_score = 0.0
for target,w in tqdm([('age',0.3),('domain1_var1',0.175),('domain1_var2',0.175),('domain2_var1',0.175),('domain2_var2',0.175)]):
    y_oof = np.zeros(train_df.shape[0])
    y_test = np.zeros((test_df.shape[0], NFOLDS))
    print('*'*20,target,'*'*20)
    for i,(train_index, valid_index) in enumerate(kf.split(train_df, train_df)):
        print('>'*20,'Fold-',i+1)
        train,val = train_df.iloc[train_index],train_df.iloc[valid_index]
        X_train = train[features]
        y_train = train[target]
        X_val = val[features]
        y_val = val[target]
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        #create model
        model = lgb.train(params, 
                          train_data, 
                          num_boost_round=100, 
                          early_stopping_rounds=20, 
                          valid_sets=[val_data], 
                          learning_rates=lambda iter: 0.05 * (0.99 ** iter),
                          verbose_eval=10)
    
        val_pred = model.predict(X_val)
        test_pred = model.predict(test_df[features])
        y_oof[valid_index] = val_pred
        y_test[:, i] = test_pred

    train_df["pred_{}".format(target)] = y_oof
    test_df[target] = y_test.mean(axis=1)

    score = my_metric(train_df[train_df[target].notnull()][target].values, train_df[train_df[target].notnull()]["pred_{}".format(target)].values)
    print("="*20,target, np.round(score, 5))
    print()
    overal_score += w*score
        
print("Overal score:", np.round(overal_score, 5))
