import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch import functional as F
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
