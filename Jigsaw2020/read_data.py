import pandas as pd

def read_data():
    train1 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv", usecols=["comment_text", "toxic"])
    train2 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv", usecols=["comment_text", "toxic"])
    train2.toxic = train2.toxic.round().astype(int)

    df_valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
    test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
    sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')

    df_train = pd.concat([
        train1[['comment_text', 'toxic']],
        train2[['comment_text', 'toxic']].query('toxic==1'),
        train2[['comment_text', 'toxic']].query('toxic==0').sample(n=99937, random_state=0),])
    df_train = df_train.sample(frac=1).reset_index(drop=True)#shuffling
    import gc
    del train1, train2
    gc.collect(); gc.collect();
    print(df_train.shape, df_valid.shape)
    gc.collect(); gc.collect(); gc.collect();
    return df_train,df_valid,test
