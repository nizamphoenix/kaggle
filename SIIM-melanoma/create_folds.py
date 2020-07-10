import os
import pandas as pd
from sklearn import model_selection

if __name__=="__main__":
    input_path = ""
    df=pd.read_csv(os.path.join(input_path,"train.csv"))
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.target.values
    kf = model_selection.StratifiedKFold(n_splits=5)#try 10 since positive classes are less
    for fold,(train_idx,valid_idx) in enumerate(kf.split(X=df,y=y)):
        df.loc[:,'kfold'] = fold
    df.to_csv(os.path.join(input_path,"train_folds.csv"),index=False)
