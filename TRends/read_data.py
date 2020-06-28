import os
import pandas as pd
import cudf
import gc
def get_train_test(fnc_file,loadings_file,lablels_file):
    '''
    function to get training and test data sets
    Works with Rapids.ai ONLY
    
    '''
    path = "../input/trends-assessment-prediction/"
    fnc_df = pd.read_csv(os.path.join(path,fnc_file))
    loading_df = pd.read_csv(os.path.join(path,loadings_file))
    fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])
    df = fnc_df.merge(loading_df, on="Id")
    labels_df = pd.read_csv(os.path.join(path,lablels_file))
    labels_df["is_train"] = True
    df = df.merge(labels_df, on="Id", how="left")
    test_df = df[df["is_train"] != True].copy()
    train_df = df[df["is_train"] == True].copy()
    train_df = train_df.drop(['is_train'], axis=1)
    target_cols = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']
    test_df = test_df.drop(target_cols + ['is_train'], axis=1)
    features = loading_features + fnc_features 
    #-----------------Normalizing------------------------
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train_df[features] = scaler.fit_transform(train_df[features],train_df[target_cols])
    test_df[features] = scaler.transform(test_df[features])
    #----------------------------------------------------
    # Giving less importance to FNC features since they are easier to overfit due to high dimensionality.
    train_df[fnc_features] = train_df[fnc_features].mul(1/600)
    test_df[fnc_features]  = test_df[fnc_features].mul(1/600) 
    #imputing missing values in targets
    from sklearn.impute import KNNImputer
    imputer = KNNImputer(n_neighbors = 5, weights="distance")
    train_df = cudf.from_pandas(pd.DataFrame(imputer.fit_transform(train_df), columns = list(train_df.columns)))
    test_df = cudf.from_pandas(test_df)#necessary for casting to gpu matrix
    del df
    gc.collect()
    return train_df,test_df,features,target_cols
