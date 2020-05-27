import numpy as np
import pandas as pd
fnc_df = pd.read_csv("../input/trends-assessment-prediction/fnc.csv")
loading_df = pd.read_csv("../input/trends-assessment-prediction/loading.csv")

fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])
df = fnc_df.merge(loading_df, on="Id")


target_df = pd.read_csv("../input/trends-assessment-prediction/train_scores.csv")
target_df["is_train"] = 1

df = df.merge(target_df, on="Id", how="left")

test_df = df[df["is_train"] != 1].copy()
df = df[df["is_train"] == 1].copy()

print("training shape={0} | testing shape={1}".format(df.shape, test_df.shape))
