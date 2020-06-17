import os
import pandas as pd
import cudf
def get_train_test(fnc_file,loadings_file,lablels_file):
    '''
    function to get training and test data sets
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
    df = df[df["is_train"] == True].copy()
    # Giving less importance to FNC features since they are easier to overfit due to high dimensionality.
    df[fnc_features] *= 1/600
    test_df[fnc_features] *= 1/600
    #imputing missing values in targets
    from sklearn.impute import KNNImputer
    imputer = KNNImputer(n_neighbors = 5, weights="distance")
    df = cudf.DataFrame(pd.DataFrame(imputer.fit_transform(df), columns = list(df.columns)))
    test_df = cudf.DataFrame(test_df)#necessary for casting to gpu matrix
    return df,test_df,fnc_features, loading_features
