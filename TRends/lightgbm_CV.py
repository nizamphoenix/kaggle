# LightGBM--Cross Validation implementation
params={
            'age':{
         'num_leaves': 100,
         'feature_fraction': 0.90,
         'bagging_fraction': 1.0,
         'bagging_freq': 0,
         'min_child_samples': 30,
         'objective': 'rmse',
         'subsample': 0.7678,
            'subsample_freq': 1,
            'max_depth': 50,
            'lambda_l1': 0,  
            'lambda_l2': 10,
         'metric': 'l1',
         'boosting_type': 'gbdt',
         'learning_rate': 0.003,
         'tree_learner': 'feature_parallel',
         'num_threads': 4,
         'seed': 0}
        ,

        'domain1_var1':{'lambda_l1': 0.0,
     'lambda_l2': 0.0,
     'num_leaves': 30,
     'feature_fraction': 0.95,
     'bagging_fraction': 0.9765733975192812,
     'bagging_freq': 1,
     'min_child_samples': 10,
     'objective': 'huber',
     'metric': 'l1',
     'boosting_type': 'gbdt',
     'learning_rate': 0.003,
     'tree_learner': 'feature_parallel',
     'num_threads': 4,
     'seed': 0}  
        ,
        'domain1_var2':{'lambda_l1': 7.733581684659643e-05,
     'lambda_l2': 1.1878841440097718,
     'num_leaves': 31,
     'feature_fraction': 1.0,
     'bagging_fraction': 1.0,
     'bagging_freq': 0,
     'min_child_samples': 25,
     'objective': 'huber',
     'metric': 'l1',
     'boosting_type': 'gbdt',
     'learning_rate': 0.003,
     'tree_learner': 'feature_parallel',
     'num_threads': 4,
     'seed': 0}
        ,
        'domain2_var1':{'lambda_l1': 1,
     'lambda_l2': 2,
     'num_leaves': 105,
     'feature_fraction': 0.9,
     'bagging_fraction': 0.8,
     'bagging_freq': 4,
     'min_child_samples': 10,
     'objective': 'huber',
     'metric': 'l1',
     'boosting_type': 'gbdt',
     'learning_rate': 0.003,
     'max_depth': -1,
     'tree_learner': 'feature_parallel',
     'num_threads': 4,
     'seed': 0}
        ,
        'domain2_var2':{'lambda_l1': 1,
     'lambda_l2': 2,
     'num_leaves': 80,
     'feature_fraction': 1.0,
     'bagging_fraction': 1.0,
     'bagging_freq': 0,
     'min_child_samples': 20,
     'objective': 'huber',
     'metric': 'l1',
     'boosting_type': 'gbdt',
     'learning_rate': 0.003,
     'max_depth': -1,
     'tree_learner': 'feature_parallel',
     'num_threads': 4,
     'seed': 0}
    }
from sklearn.model_selection import KFold

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
import gc

import lightgbm as lgb


fnc_df = pd.read_csv("../input/trends-assessment-prediction/fnc.csv")
loading_df = pd.read_csv("../input/trends-assessment-prediction/loading.csv")
labels_df = pd.read_csv("../input/trends-assessment-prediction/train_scores.csv")

fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])
df = fnc_df.merge(loading_df, on="Id")
labels_df["is_train"] = True

df = df.merge(labels_df, on="Id", how="left")

target_cols = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']

#imputing missing values in targets
# from sklearn.impute import KNNImputer
# imputer = KNNImputer(n_neighbors = 5, weights="distance")
# df[target_cols] = pd.DataFrame(imputer.fit_transform(df[target_cols]), columns = target_cols)

test_df = df[df["is_train"] != True].copy()
train_df = df[df["is_train"] == True].copy()

train_df = train_df.drop(['is_train'], axis=1)
test_df = test_df.drop(target_cols+['is_train'], axis=1)


targets=['age','domain1_var1','domain1_var2', 'domain2_var1','domain2_var2']
features=list(set(train_df.columns)-set(targets)-set(['Id']))


train_df[fnc_features]=train_df[fnc_features].mul(1/600)
#train_df[fnc_features]=train_df[fnc_features].pow(2)

test_df[fnc_features]=test_df[fnc_features].mul(1/600)
#test_df[fnc_features]=test_df[fnc_features].pow(2)




#-------Normalizing------------------------
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_df[features] = scaler.fit_transform(train_df[features],train_df[targets])
test_df[features] = scaler.transform(test_df[features])
#----------------------------------------------------
to_drop=['IC_20',
         'IC_02',
         'IC_05',
         'IC_16',
         'IC_10',
         'IC_18'
        ]
# train_df = train_df.drop(to_drop, axis=1)
# test_df = test_df.drop(to_drop, axis=1)
print(train_df.shape,test_df.shape)
print("Train and test dataframes contain Id column!!")


def my_metric(y_true, y_pred):
    return np.mean(np.sum(np.abs(y_true - y_pred), axis=0)/np.sum(y_true, axis=0))


NFOLDS = 7
from sklearn.model_selection import KFold
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=0)

targets=['age','domain2_var1','domain2_var2', 'domain1_var1','domain1_var2']
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
        model = lgb.train(params[target], 
                          train_data, 
                          num_boost_round=10000, 
                          early_stopping_rounds=60, 
                          valid_sets=[train_data,val_data], 
                          verbose_eval=50)
    
        val_pred = model.predict(X_val)
        test_pred = model.predict(test_df[features])
        y_oof[valid_index] = val_pred
        y_test[:, i] = test_pred

    train_df["pred_{}".format(target)] = y_oof
    test_df[target] = y_test.mean(axis=1)

    score = my_metric(train_df[train_df[target].notnull()][target].values, train_df[train_df[target].notnull()]["pred_{}".format(target)].values)
    print("="*20,target, np.round(score, 5))
    print("-"*100)
    overal_score += w*score
        
print("Overal score:", np.round(overal_score, 5))
