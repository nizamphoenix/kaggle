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
