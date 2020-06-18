import numpy as np
def trends_score(y_true,y_preds):
    y_true = np.array(y_true)
    y_preds = np.array(y_preds)
    w = np.array([.3, .175, .175, .175, .175])#weights as dictated by competition
    return np.mean(np.matmul(np.abs(y_true-y_preds),w/np.mean(y_true,axis=0)),axis=0)


def my_metric(y_true, y_pred):
    return np.mean(np.sum(np.abs(y_true - y_pred), axis=0)/np.sum(y_true, axis=0))
age=my_metric(y_train_df[3634:]['age'],m.predict(train_df[3634:]))
d1v1=my_metric(y_train_df[3634:]['domain1_var1'],model_d1.predict(train_df[3634:])[:,0])
d1v2=my_metric(y_train_df[3634:]['domain1_var2'],model_d1.predict(train_df[3634:])[:,1])
d2v1=my_metric(y_train_df[3634:]['domain2_var1'],model_d2.predict(train_df[3634:])[:,0])
d2v2=my_metric(y_train_df[3634:]['domain2_var2'],model_d2.predict(train_df[3634:])[:,1])
score = 0.3*age+0.175*(d1v1+d1v2+d2v1+d2v2)
print("Score:",score)
