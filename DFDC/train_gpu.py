import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm,trange
from sklearn.model_selection import train_test_split
import sklearn.metrics

import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import Dataset, DataLoader
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class Head(torch.nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()

        self.f = nn.Flatten()
        
        self.b1 = nn.BatchNorm1d(in_f)#in_f=2048
        self.d1 = nn.Dropout(0.40)
        
        self.l2 = nn.Linear(in_f, 512)
        self.r2  = nn.ReLU()
        self.b2 = nn.BatchNorm1d(512)
        self.d2 = nn.Dropout(0.30)

        self.l3 = nn.Linear(512, 1024)
        self.r3  = nn.ReLU()
        self.b3 = nn.BatchNorm1d(1024)
        self.d3 = nn.Dropout(0.30)
        
        self.o = nn.Linear(1024, out_f)#out_f=1
        
        
       
    def forward(self, x):
        x = self.f(x)
        
        x = self.b1(x)
        x = self.d1(x)
            
        x = self.l2(x)
        x = self.r2(x)
        x = self.b2(x)
        x = self.d2(x)
        
        x = self.l3(x)
        x = self.r3(x)
        x = self.b3(x)
        x = self.d3(x)
        
        out = self.o(x)
        return out

    
class FCN(torch.nn.Module):
    def __init__(self, base, in_f):
        super().__init__()
        self.base = base
        self.h1 = Head(in_f, 1)
  
    def forward(self, x):
        x = self.base(x)
        return self.h1(x)

model = FCN(model, 2048)

def get_path(img_name,suffix):
    path = '../input/deepfake/DeepFake'+suffix+'/DeepFake'+suffix+'/' + img_name.replace(".mp4","")+ '.jpg'
    if not os.path.exists(path):
        raise Exception
    return path

def get_all_paths(df_list,suffixes_list):
    LABELS = ['REAL','FAKE']
    paths = []
    labels = []
    for df,suffix in tqdm(zip(df_list,suffixes_list),total=len(df_list)):
        images = list(df.columns.values)
        for img_name in images:
            try:
                paths.append(get_path(img_name,suffix))
                labels.append(LABELS.index(df[img_name]['label']))
            except Exception as err:
                #print(err)
                pass
    return paths,labels



class ImageDataset(Dataset):
    def __init__(self, X, y, training=True, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        self.training = training

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img = self.X[idx]

        if self.transform is not None:
          res = self.transform(image=img)
          img = res['image']
        
        img = np.rollaxis(img, 2, 0)
        # img = np.array(img).astype(np.float32) / 255.

        labels = self.y[idx]
        labels = np.array(labels).astype(np.float32)
        return [img, labels]
val_y=[]
for df_val,num in tqdm(zip(df_vals,val_nums),total=len(df_vals)):
    images = list(df_val.columns.values)
    for x in images:
        try:
            val_paths.append(get_path(num,x))
            val_y.append(LABELS.index(df_val[x]['label']))
        except Exception as err:
            #print(err)
            pass
        
def calc_loss(pred, targets):
  return F.binary_cross_entropy(F.sigmoid(pred), targets)

def train_model(epoch, optimizer, scheduler=None, history=None):
    model.train()
    total_loss = 0
    
    t = tqdm(train_loader)
    for i, (img_batch, y_batch) in enumerate(t):
        img_batch = img_batch.cuda().float()
        y_batch = y_batch.cuda().float()

        optimizer.zero_grad()

        out = model(img_batch)
        loss = calc_loss(out, y_batch)

        total_loss += loss
        t.set_description(f'Epoch {epoch+1}/{n_epochs}, LR: %6f, Loss: %.4f'%(optimizer.state_dict()['param_groups'][0]['lr'],total_loss/(i+1)))

        if history is not None:
          history.loc[epoch + i / len(X), 'train_loss'] = loss.data.cpu().numpy()
          history.loc[epoch + i / len(X), 'lr'] = optimizer.state_dict()['param_groups'][0]['lr']

        loss.backward()
        optimizer.step()
        if scheduler is not None:
          scheduler.step()

def evaluate_model(epoch, scheduler=None, history=None):
    model.eval()
    loss = 0
    pred = []
    real = []
    with torch.no_grad():
        for img_batch, y_batch in val_loader:
            img_batch = img_batch.cuda().float()
            y_batch = y_batch.cuda().float()

            op = model(img_batch)
            temp_loss = calc_loss(op, y_batch)
            loss += temp_loss
            
            for j in op:
              pred.append(F.sigmoid(j))
            for i in y_batch:
              real.append(i.data.cpu())
    
    pred = [p.data.cpu().numpy() for p in pred]
    f_pred = pred
    pred = [np.round(p) for p in pred]
    pred = np.array(pred)
    acc = sklearn.metrics.recall_score(real, pred, average='macro')

    real = [r.item() for r in real]
    f_pred = np.array(f_pred).clip(0.1, 0.9)
    

    loss /= len(val_loader)
    
    if history is not None:
        history.loc[epoch, 'dev_loss'] = loss.cpu().numpy()
    
    if scheduler is not None:
      scheduler.step(loss)

    print(f'Development loss: %.4f, Acc: %.6f'%(loss,acc))
    
    return loss
