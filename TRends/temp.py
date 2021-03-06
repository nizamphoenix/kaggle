import numpy as np
from cuml import SVR
from cuml import RandomForestRegressor
from cuml import NearestNeighbors,KMeans,UMAP,Ridge,ElasticNet
import cupy as cp
from sklearn.model_selection import KFold


def my_metric(y_true, y_pred):
    return np.mean(np.sum(np.abs(y_true - y_pred), axis=0)/np.sum(y_true, axis=0))

def cv_train_predict(df,test_df,features):
    '''
    training with k-fold cross-validation
    '''
    weights={}#Weights & other hyperparameters
    #[target,SVR_penalty(C),score_wieght,ElasticNet_weight,RandomForest_weight]
    weights['age']=["age",100,0.3,0.15,0.3]
    weights['domain1_var1']=["domain1_var1",12,0.175,0.2,0.4]
    weights['domain1_var2']=["domain1_var2", 8,0.175,0.2,0.4]
    weights['domain2_var1']=["domain2_var1",10,0.175,0.3,0.2]
    weights['domain2_var2']=["domain2_var2",12,0.175,0.22,0.4]
    NUM_FOLDS = 5
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)
    overal_score = 0
    for target,c,w,el,rf in [weights['age'], weights['domain1_var1'], weights['domain1_var2'], weights['domain2_var1'], weights['domain2_var2']]:    
        y_oof = np.zeros(df.shape[0])
        y_test = np.zeros((test_df.shape[0], NUM_FOLDS))

        for f, (train_ind, val_ind) in enumerate(kf.split(df)):
            train_df, val_df = df.iloc[train_ind], df.iloc[val_ind]
          
            #-------training,val,test data preparation for RandomForestRegressor since it operates on float32
            X_train = np.array(train_df[features].to_gpu_matrix(),dtype=np.float32)
            y_train = np.array(train_df[[target]].to_gpu_matrix(),dtype=np.float32)
            X_val = np.array(val_df[features].to_gpu_matrix(),dtype=np.float32)
            y_val = np.array(val_df[[target]].to_gpu_matrix(),dtype=np.float32)
            X_test = np.array(test_df[features].to_gpu_matrix(),dtype=np.float32)
            #---------------------------------------------------------------------------------------
            
            
            model = RandomForestRegressor(n_estimators=100,split_criterion=3,accuracy_metric=my_metric,bootstrap=True,seed=0)
            model.fit(X_train,y_train)
            model_1 = SVR(C=c, cache_size=3000.0)
            model_1.fit(train_df[features].values, train_df[target].values)
            model_2 = ElasticNet(alpha = 1,l1_ratio=0.2)
            model_2.fit(train_df[features].values, train_df[target].values)
            
            val_pred_rf   = model.predict(X_val)
            val_pred_1 = model_1.predict(val_df[features])
            val_pred_2 = model_2.predict(val_df[features])

            test_pred_rf   = model.predict(X_test)
            test_pred_1 = model_1.predict(test_df[features])
            test_pred_2 = model_2.predict(test_df[features])
            #pred    = Blended prediction(RandomForest + Blended prediction(ElasticNet & SVR))
            
            
            val_pred = rf*val_pred_rf + cp.asnumpy((1-rf)*((1-el)*val_pred_1+el*val_pred_2))
            #val_pred = cp.asnumpy(val_pred.values.flatten())
            
            test_pred = rf*test_pred_rf + cp.asnumpy((1-rf)*((1-el)*test_pred_1+el*test_pred_2))
            #test_pred = cp.asnumpy(test_pred.values.flatten())

            y_oof[val_ind] = val_pred
            y_test[:, f] = test_pred

        df["pred_{}".format(target)] = y_oof
        test_df[target] = y_test.mean(axis=1)

        score = my_metric(df[df[target].notnull()][target].values, df[df[target].notnull()]["pred_{}".format(target)].values)
        overal_score += w*score
        print(target, np.round(score, 5))
        print()
    print("Overal score:", np.round(overal_score, 5))
