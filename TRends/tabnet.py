!pip install fastai2>/dev/null
!pip install fast_tabnet>/dev/null
from fastai2.basics import *
from fastai2.tabular.all import *
from fast_tabnet.core import *


# Loading 

from sklearn.model_selection import KFold, train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
import gc


fnc_df = pd.read_csv("../input/trends-assessment-prediction/fnc.csv")
loading_df = pd.read_csv("../input/trends-assessment-prediction/loading.csv")
labels_df = pd.read_csv("../input/trends-assessment-prediction/train_scores.csv")

fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])
df = fnc_df.merge(loading_df, on="Id")
labels_df["is_train"] = True

df = df.merge(labels_df, on="Id", how="left")

target_cols = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']

#imputing missing values in targets
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors = 5, weights="distance")
df[target_cols] = pd.DataFrame(imputer.fit_transform(df[target_cols]), columns = target_cols)

test_df = df[df["is_train"] != True].copy()
train_df = df[df["is_train"] == True].copy()

#y_train_df = train_df[target_cols]
train_df = train_df.drop(['is_train'], axis=1)
#train_df = train_df.drop(target_cols + ['is_train'], axis=1)
test_df = test_df.drop(target_cols+['is_train'], axis=1)

#------feature transformations------
train_df[features]=train_df[features].pow(2)
train_df[fnc_features]=train_df[fnc_features].mul(1/100)
train_df[fnc_features]=train_df[fnc_features].pow(2)

test_df[features]=test_df[features].pow(2)
test_df[fnc_features]=test_df[fnc_features].mul(1/100)
test_df[fnc_features]=test_df[fnc_features].pow(2)
#-------------------------------------
targets=['age','domain1_var1','domain1_var2', 'domain2_var1','domain2_var2']
features=list(set(train_df.columns)-set(targets)-set(['Id']))
#-------Normalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_df[features] = scaler.fit_transform(train_df[features],train_df[targets])
test_df[features] = scaler.transform(test_df[features])
#------------------------------
print(train_df.shape,test_df.shape)
print("Train and test dataframes contain Id columns,too!!")




#using fastai's tabnet with GPU

targets=['age','domain1_var1','domain1_var2', 'domain2_var1','domain2_var2']
features=list(set(train_df.columns)-set(targets))
import torch
device=torch.device('cuda')

%%time 

path = '/kaggle/working/'

to = TabularPandas(
    df=train_df,
    procs=[FillMissing, Normalize],
    cat_names=None,
    cont_names=features,
    y_names=targets,
    y_block=TransformBlock(),
    splits=RandomSplitter(valid_pct=0.2, seed=0)(range_of(train_df)),
    do_setup=True,
    device=device,
    inplace=False,
    reduce_memory=True,
)
dls = to.dataloaders(bs=512, path=path)
dls.show_batch()

to_tst = to.new(test_df)
to_tst.process()
to_tst.all_cols.head()
emb_szs = get_emb_sz(to)
print(emb_szs)

def trends_scorer_multitask_scoring_gpu(y_true,y_preds):
    '''
    custom scoring function used for evaluation in this competition
    '''
    import numpy as np
    y_true = y_true.cpu().detach().numpy()
    y_preds= y_preds.cpu().detach().numpy()
    w = np.array([.3, .175, .175, .175, .175])
    op = np.mean(np.matmul(np.abs(y_true-y_preds),w/np.mean(y_true,axis=0)),axis=0)
    return op


from fastai2.layers import L1LossFlat,MSELossFlat
from fastai2.metrics import mae
model=TabNetModel(
    emb_szs,
    n_cont=len(features),
    out_sz=5,#Number of outputs we expect from our network - in this case 2.
    embed_p=0.0,
    y_range=None,
    n_d=8,
    n_a=8,
    n_steps=3,
    gamma=1.5,
    n_independent=2,
    n_shared=2,
    epsilon=1e-15,
    virtual_batch_size=128,
    momentum=0.02,
)
opt_func = partial(Adam, wd=0.01, eps=1e-5)
learn = Learner(dls, model, loss_func=MSELossFlat(), opt_func=opt_func, lr=3e-2, metrics=[trends_scorer_multitask_scoring_gpu])


#Training
learn.lr_find()
cb = SaveModelCallback()
learn.fit_one_cycle(n_epoch=100, cbs=cb)
learn.load('model')
tst_dl = dls.valid.new(to_tst)
tst_dl.show_batch()


#Predictions
learn.metrics = []
tst_preds,_ = learn.get_preds(dl=tst_dl)
res = pd.DataFrame(np.array(tst_preds),columns=[targets])
ids=pd.DataFrame(test_df.Id.values,columns=['Id'])
a=pd.concat([ids,res],axis=1)
b=a.iloc[:,0:6]
b.columns=['Id','age','domain1_var1','domain1_var2', 'domain2_var1','domain2_var2']
b.head()

#Writing to submission.csv file
output = b #
sub_df = pd.melt(output[["Id", "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]], id_vars=["Id"], value_name="Predicted")
sub_df["Id"] = sub_df["Id"].astype("str") + "_" +  sub_df["variable"].astype("str")
sub_df = sub_df.drop("variable", axis=1).sort_values("Id")
assert sub_df.shape[0] == test_df.shape[0]*5
sub_df.head(10)
