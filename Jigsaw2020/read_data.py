import pandas as pd

def read_data():
    df_train = pd.read_csv("/kaggle/input/jigsaw-public-dataset-/train.csv", usecols=["comment_text", "toxic"])
    df_train = df_train.sample(frac=1).reset_index(drop=True)#shuffling
    df_valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
    test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
    sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')

    import gc
    del train1, train2
    gc.collect(); gc.collect();
    print(df_train.shape, df_valid.shape)
    gc.collect(); gc.collect(); gc.collect();
    return df_train,df_valid,test
