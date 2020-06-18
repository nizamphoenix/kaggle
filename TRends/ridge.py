#Ridge regression individually on each of the 5 target variables
def my_score(model,X,y_true):
    import numpy as np
    y_true = np.array(y_true)
    y_pred=model.predict(X)
    return np.mean(np.sum(np.abs(y_true - y_pred), axis=0)/np.sum(y_true, axis=0))
from sklearn.linear_model import Ridge
alpha=0.17 
m=Ridge(alpha,normalize=True)
m.fit(train_df[:3634],y_train_df[:3634]['age'])
np.round(my_score(m,train_df[3634:],y_train_df[3634:]['age']),5)
#FOR AGE gives 0.1483 score
