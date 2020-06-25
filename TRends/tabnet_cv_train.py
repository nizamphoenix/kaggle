from sklearn.model_selection import KFold
from import get_model
from import get_tabnet_data
from losses import SmoothMaeLoss,MSEMaeLoss

NUM_FOLDS = 7
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)
all_preds = []
for i,(train_index, val_index) in enumerate(kf.split(train_df,train_df)):
    print('fold-',i+1)
    to,n_features=get_tabnet_data(train_df,(list(train_index), list(val_index)))
    dls = to.dataloaders(bs=512, path='/kaggle/working/')
    emb_szs = get_emb_sz(to)
    learn=get_model(emb_szs,dls,n_features,loss=SmoothMaeLoss(l1=0.8))
    learn.fit_one_cycle(n_epoch=100,lr_max=0.33,cbs=None)
    #predicting
    to_tst = to.new(test_df)
    to_tst.process()
    tst_dl = dls.valid.new(to_tst)
    tst_preds,_ = learn.get_preds(dl=tst_dl)
    all_preds.append(tst_preds)
    
    
