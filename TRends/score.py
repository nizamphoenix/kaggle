import numpy as np
def trends_score(y_true,y_preds):
    y_true = np.array(y_true)
    y_preds = np.array(y_preds)
    w = np.array([.3, .175, .175, .175, .175])#weights as dictated by competition
    return np.mean(np.matmul(np.abs(y_true-y_preds),w/np.mean(y_true,axis=0)),axis=0)
