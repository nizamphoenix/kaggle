import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score
import graphviz
from sklearn import tree
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import random
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)

seed_everything(1)
def preproc(train,test):
    #Normalizing
    train_input_mean = train.signal.mean()
    train_input_sigma = train.signal.std()
    train['signal'] = (train.signal-train_input_mean)/train_input_sigma
    test['signal'] = (test.signal-train_input_mean)/train_input_sigma
    #Adding Previous signal values to make markovian
    train['prev'] = 0
    train['prev'][0+1:500000] = train['signal'][0:500000-1]
    train['prev'][500000+1:1000000] = train['signal'][500000:1000000-1]
    train['prev'][1000000+1:1500000] = train['signal'][1000000:1500000-1]
    train['prev'][1500000+1:2000000] = train['signal'][1500000:2000000-1]
    train['prev'][2000000+1:2500000] = train['signal'][2000000:2500000-1]
    train['prev'][2500000+1:3000000] = train['signal'][2500000:3000000-1]
    train['prev'][3000000+1:3500000] = train['signal'][3000000:3500000-1]
    train['prev'][3500000+1:4000000] = train['signal'][3500000:4000000-1]
    train['prev'][4000000+1:4500000] = train['signal'][4000000:4500000-1]
    train['prev'][4500000+1:5000000] = train['signal'][4500000:5000000-1]

    #Adding Previous signal values to make markovian

    test['prev'] = 0
    test['prev'][0+1:500000] = test['signal'][0:500000-1]
    test['prev'][500000+1:1000000] = test['signal'][500000:1000000-1]
    test['prev'][1000000+1:1500000] = test['signal'][1000000:1500000-1]
    test['prev'][1500000+1:2000000] = test['signal'][1500000:2000000-1]
    
    train_info_dict={}
    from collections import Counter
    for key,value in tqdm(train_dict.items()):
        train_info_dict[key] = Counter(value)
    print('len(train_dict)',len(train_dict)) 
    sns.distplot(list(train_dict.keys()))
    plt.show()
    counter = 0
    special_signals={}#key:signal value value:channels
    for key,value in tqdm(train_info_dict.items()):
        if len(value)>1:
            special_signals[key] = value
            counter+=1
            #print(key,value)
    print("No of signal values that overlap:",counter)
    channel_probs = []
    for sig,chan_dict in tqdm(special_signals.items()):
        for i,j in chan_dict.items():
            #print(sig,i,np.round(j/sum(chan_dict.values()),4))
            channel_probs.append((sig,i,np.round(j/sum(chan_dict.values()),4)))
    df_channel_probs = pd.DataFrame(channel_probs)
    df_channel_probs.columns=['signal','open_channels','prob']
    result = pd.merge(train, df_channel_probs, how='left', on=['signal','open_channels'])
    result = result.fillna(1)
    return train,test


def get_batches(df ,batch_size):
    df['group'] = df.groupby(df.index//batch_size, sort=False)['signal'].agg(['ngroup']).values
    df['group'] = df['group'].astype(np.uint16)
    return df
