import optuna.integration.lightgbm as lgb

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split

from tqdm.notebook import tqdm
import gc



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


train_df[features]=train_df[features].pow(2)
train_df[fnc_features]=train_df[fnc_features].mul(1/100)
train_df[fnc_features]=train_df[fnc_features].pow(2)

test_df[features]=test_df[features].pow(2)
test_df[fnc_features]=test_df[fnc_features].mul(1/100)
test_df[fnc_features]=test_df[fnc_features].pow(2)

print(train_df.shape,test_df.shape)
print("Train and test dataframes contain Id columns,too!!")



targets=['age','domain1_var1','domain1_var2', 'domain2_var1','domain2_var2']
features=list(set(train_df.columns)-set(targets)-set(['Id']))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_df[features] = scaler.fit_transform(train_df[features],train_df[targets])
test_df[features] = scaler.transform(test_df[features])


X_train = train_df[features]
X_test = test_df[features]
y_train = train_df[targets]
print(X_train.shape,X_test.shape)

def my_metric(y_pred,train_data):
    y_true = train_data.get_label()
    print(len(y_true),len(y_pred))
    return np.mean(np.sum(np.abs(y_true - y_pred), axis=0)/np.sum(y_true, axis=0))

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True, random_state=20)
train_data = lgb.Dataset(X_tr, label=y_tr['domain2_var1'])
val_data = lgb.Dataset(X_val, label=y_val['domain2_var1'])
params = {
        'objective':'fair',
        'metric':'l1',
        'boosting_type':'gbdt',
        'learning_rate':0.003,
        'tree_learner':'feature_parallel',
        'num_threads':4,
        'seed':0
        }

best_params, tuning_history = dict(), list()

model = lgb.train(params, 
                  train_data, 
                  num_boost_round=100, 
                  early_stopping_rounds=20, 
                  valid_sets=[train_data,val_data], 
                  verbose_eval=20,
                  learning_rates=lambda it: 0.01 * (0.8 ** it),
                  best_params=best_params,
                 tuning_history=tuning_history)
 
print("Best Params", best_params)
