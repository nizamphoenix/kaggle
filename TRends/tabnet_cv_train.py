from sklearn.model_selection import KFold
from tabnet_model import get_model
from tabnet_data import get_tabnet_data
from losses import SmoothMaeLoss,MSEMaeLoss


NUM_FOLDS = 5
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)
all_preds = []
for i,(train_index, val_index) in enumerate(kf.split(train_df,train_df)):
    print('fold-',i+1)
    #get data
    to,n_features=get_tabnet_data(train_df,(list(train_index), list(val_index)))
    dls = to.dataloaders(bs=512, path='/kaggle/working/')
    emb_szs = get_emb_sz(to)
    #get model
    model = get_model(emb_szs,dls,n_features)
    opt_func = partial(Adam,lr=0.01,mom=0.9,sqr_mom=0.99,wd=0.01,eps=1e-5,decouple_wd=True)
    learn = Learner(dls, model, loss_func=SmoothMSELoss(l1=0.0), opt_func=opt_func, metrics=trends_scorer_multitask_scoring_gpu,callback_fns=[SaveModelCallback(monitor=[trends_scorer_multitask_scoring_gpu],min_delta=0.01,fname='model',every_epoch=False,with_opt=True),EarlyStoppingCallback(monitor='valid_loss',mode='min', min_delta=0, patience=1)])
    learn.fit_one_cycle(n_epoch=70,lr_max=0.33,cbs=None)
    #learn.fine_tune(30,base_lr=0.00001,cbs=cb)
    #predicting
    to_tst = to.new(test_df)
    to_tst.process()
    tst_dl = dls.valid.new(to_tst)
    tst_preds,_ = learn.get_preds(dl=tst_dl)
    all_preds.append(tst_preds)
    
