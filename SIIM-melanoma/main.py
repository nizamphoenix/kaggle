import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch import functional as F
import albumenations

class SEResNext50_32x4d(nn.Module):
    def __init__(self,pretrained="imagenet"):
        super().__init__()
        self.model = pretrainedmodels.__dict__["se_resnext50_32x4d"](pretrained=pretrained)
        self.out   = nn.Linear(2048,1)
        
    def forward(self,image):
        bs,_,_,_ = image.shape
        x = self.model.features(image)
        x = F.adaptive_avg_pool2d(x,1)
        x = x.reshape(bs,-1)
        out = self.out(x)
        return out

  def run(fold):
    training_data_path = ""
    df = pd.read_csv("train_folds.csv")
    device = "cuda"
    epochs=50
    train_bs = 32
    valid_bs = 16
    df_train = df[df.kfold!=fold].reset_index(drop=True)
    df_valid = df[df.kfold==fold].reset_index(drop=True)
    mean = (0.485,0.456,0.406)
    std  = (0.229,0.224,0.225)
    train_aug = albumenations.Compose(
        [
        albumenations.Normalize(mean,std,max_pixel_value=255.0,always_apply=True)
        ]
    )
    valid_aug = albumenations.Compose(
        [
        albumenations.Normalize(mean,std,max_pixel_value=255.0,always_apply=True)
        ]
    )
    
    
    
