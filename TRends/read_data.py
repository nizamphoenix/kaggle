import os
import pandas as pd
import cudf#rapidsai--to run on GPU
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
    return df,test_df,fnc_features, loading_features
