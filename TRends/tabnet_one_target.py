overal_score = 0.0
!rm -rf /kaggle/working/models
from sklearn.model_selection import KFold
from torch.nn import SmoothL1Loss
NUM_FOLDS = 7
def my_metric(y_true, y_pred):
    return np.mean(np.sum(np.abs(y_true - y_pred), axis=0)/np.sum(y_true, axis=0))
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)
for target,w in tqdm([('domain2_var1',0.175),('domain2_var2',0.175),('age',0.3),('domain1_var1',0.175),('domain1_var2',0.175)]):
    y_oof = np.zeros(train_df.shape[0])
    y_test = np.zeros((test_df.shape[0], NUM_FOLDS))
    print('*'*20,target,'*'*20)
    for i,(train_index, valid_index) in enumerate(kf.split(train_df, train_df)):
        print('>'*20,'Fold-',i+1)
        _, val_df = train_df.iloc[train_index], train_df.iloc[valid_index]
        #get data
        target_name=[]
        target_name.append(target)
        to,features,n_labels = get_tabnet_data(train_df,(list(train_index), list(valid_index)),target_name)
        dls = to.dataloaders(bs=512, path='/kaggle/working/')
        emb_szs = get_emb_sz(to)
        #get model
        model = get_model(emb_szs,dls,len(features),n_labels)
        opt_func = partial(Adam,lr=0.01,mom=0.9,sqr_mom=0.99,wd=0.01,eps=1e-5,decouple_wd=True)
        
        learn = Learner(dls, model, loss_func=SmoothL1Loss(), opt_func=opt_func, metrics=my_metric_gpu)
        learn.fit_one_cycle(
        30,
        lr_max=0.2,
        div=25.0,
        div_final=1000000,
        pct_start=0.3,
        cbs=[EarlyStoppingCallback(min_delta=0.01,patience=20),
                             SaveModelCallback(fname="model_{}".format(i+1),min_delta=0.01)]
    )
        learn.load("model_{}".format(i+1))
        learn.fit_one_cycle(
        20,
        lr_max=1e-2, wd=0.3,
        div=10.0,
        div_final=1000000,
        pct_start=0.4,
        cbs=[EarlyStoppingCallback(min_delta=0.001,patience=20),
                             SaveModelCallback(fname="model_{}".format(i+1),min_delta=0.001)]
    )   
        learn.load("model_{}".format(i+1))
        learn.fit_one_cycle(
        20,
        lr_max=1e-5, wd=0.1,
        div=10.0,
        div_final=1000000,
        pct_start=0.6,
        cbs=[EarlyStoppingCallback(min_delta=0.003,patience=20),
                             SaveModelCallback(fname="model_{}".format(i+1),min_delta=0.003)]
    )   
        

        learn.load("model_{}".format(i+1))
        print("Best model:",learn.loss)
        print("len(features):",len(features))
        #validation
        to_val = to.new(val_df[features])
        to_val.process()
        val_dl = dls.valid.new(to_val)
        val_pred,_ = learn.get_preds(dl=val_dl)
        
        #prediction
        to_tst = to.new(test_df[features])
        to_tst.process()
        tst_dl = dls.valid.new(to_tst)
        test_pred,_ = learn.get_preds(dl=tst_dl)
    
        val_pred=val_pred.reshape(-1,)
        y_oof[valid_index] = val_pred
        
        test_pred=test_pred.reshape(-1,)
        y_test[:, i] = test_pred

    train_df["pred_{}".format(target)] = y_oof
    test_df[target] = y_test.mean(axis=1)

    score = my_metric(train_df[train_df[target].notnull()][target].values, train_df[train_df[target].notnull()]["pred_{}".format(target)].values)
    print("="*20,target, np.round(score, 5))
    print("-"*100)
    overal_score += w*score  
        
print("Overal score:", np.round(overal_score, 5)) 
