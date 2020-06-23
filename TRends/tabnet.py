!pip install fastai2>/dev/null
!pip install fast_tabnet>/dev/null
from fastai2.basics import *
from fastai2.tabular.all import *
from fast_tabnet.core import *

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
