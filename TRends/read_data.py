import cudf#rapid ai
def get_data():
  '''
  returns train dataframe,test dataframe,test dataframe from site2(note:not all are mentioned)
  '''
  fnc_df = cudf.read_csv("../input/trends-assessment-prediction/fnc.csv")
  loading_df = cudf.read_csv("../input/trends-assessment-prediction/loading.csv")
  fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])
  df = fnc_df.merge(loading_df, on="Id")
  target_df = cudf.read_csv("../input/trends-assessment-prediction/train_scores.csv")
  target_df["is_train"] = 1
  df = df.merge(target_df, on="Id", how="left")
  test_df = df[df["is_train"] != 1].copy()
  df = df[df["is_train"] == 1].copy()
  print("training shape={0} | testing shape={1}".format(df.shape, test_df.shape))
  #510 Ids of site2 are known, there are more 510 is not all. training data does not have site2 Ids
  site2_df = cudf.read_csv("../input/trends-assessment-prediction/reveal_ID_site2.csv")
  testdf_site2 = test_df[test_df['Id'].isin(list(site2_df['Id']))]
  return df,test_df,testdf_site2
