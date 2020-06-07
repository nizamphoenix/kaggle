import pandas as pd

def read_data():
    df_train = pd.read_csv("/kaggle/input/jigsaw-public-dataset-/train.csv", usecols=["comment_text", "toxic"])
    df_train = df_train.sample(frac=1).reset_index(drop=True)#shuffling
    df_valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
    test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
    sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
    return df_train,df_valid,test
