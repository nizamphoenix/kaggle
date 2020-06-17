import numpy as np
from cuml import SVR

def my_metric(y_true, y_pred):
    return np.mean(np.sum(np.abs(y_true - y_pred), axis=0)/np.sum(y_true, axis=0))

def cv_train_predict(df,test_df,features):
    '''
    training with k-fold cross-validation
    '''
    NUM_FOLDS = 5
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)
    overal_score = 0
    for target, w in [("age",0.3), ("domain1_var1",0.175), ("domain1_var2", 0.175), ("domain2_var1", 0.175), ("domain2_var2", 0.175)]:    
        y_oof = np.zeros(df.shape[0])
        y_test = np.zeros((test_df.shape[0], NUM_FOLDS))

        for f, (train_ind, val_ind) in enumerate(kf.split(df)):
            train_df, val_df = df.iloc[train_ind], df.iloc[val_ind]
            model = RandomForestRegressor(n_estimators=200,split_criterion=3,accuracy_metric=my_metric,bootstrap=True,seed=0)
            #training
            X_train = np.array(train_df[features].to_gpu_matrix(),dtype=np.float32)
            y_train = np.array(train_df[[target]].to_gpu_matrix(),dtype=np.float32)
            model.fit(X_train,y_train)
            #validation: prediction on validation data for the target of interest
            X_val = np.array(val_df[features].to_gpu_matrix(),dtype=np.float32)
            y_val = np.array(val_df[[target]].to_gpu_matrix(),dtype=np.float32)
            y_oof[val_ind] = model.predict(val_df[features])
            #prediction on test data NOT validation data for the target of interest
            X_test = np.array(test_df[features].to_gpu_matrix(),dtype=np.float32)
            y_test[:, f] = model.predict(X_test)

        df["pred_{}".format(target)] = y_oof
        test_df[target] = y_test.mean(axis=1)

        score = my_metric(df[df[target].notnull()][target].values, df[df[target].notnull()]["pred_{}".format(target)].values)
        overal_score += w*score
        print(target, np.round(score, 5))
        print()
    print("Overal score:", np.round(overal_score, 5))
