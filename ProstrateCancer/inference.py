import cv2
from tqdm import tqdm_notebook as tqdm
import fastai
from fastai.vision import *
import os
from mish_activation import *
import warnings
warnings.filterwarnings("ignore")
import skimage.io
import numpy as np
import pandas as pd
sys.path.insert(0, '../input/semisupervised-imagenet-models/semi-supervised-ImageNet1K-models-master/')
from hubconf import *
fastai.__version__

DATA   = '../input/prostate-cancer-grade-assessment/test_images'
TEST   = '../input/prostate-cancer-grade-assessment/test.csv'
SAMPLE = '../input/prostate-cancer-grade-assessment/sample_submission.csv'
MODELS = [f'../input/panda-training/RNXT50_{i}.pth' for i in range(4)]
#MODELS = [f'../input/panda-starter-models/RNXT50_{i}.pth' for i in range(4)]#original


sz = 128
bs = 2
N = 12
nworkers = 2
import torch
import torch.nn as nn
def get_resnext(layers, pretrained, progress, **kwargs):
    from torchvision.models.resnet import ResNet, Bottleneck
    model = ResNet(Bottleneck, layers, **kwargs)
    model.load_state_dict(torch.load('../input/resnext-50-ssl/semi_supervised_resnext50_32x4-ddb3e555.pth'))
    return model


class GleasonModel(nn.Module):
    def __init__(self, n=4):
        super().__init__()
        #Set pretrained to False
        m = get_resnext([3, 4, 6, 3], pretrained=False, progress=False, groups=32,width_per_group=4)
        self.enc = nn.Sequential(*list(m.children())[:-2])       
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(AdaptiveConcatPool2d(),
                                  Flatten(),
                                  nn.Linear(2*nc,512),
                                  Mish(),
                                  nn.BatchNorm1d(512), 
                                  nn.Dropout(0.5),
                                  nn.Linear(512,256),
                                  Mish(),
                                  nn.BatchNorm1d(256), 
                                  nn.Dropout(0.3))
        self.prim =  nn.Linear(256,n)
        self.sec  =  nn.Linear(256,n)
      

        
    def forward(self, *x):
        shape = x[0].shape
        n = shape[1]# no_of_tiles
        x = x[0].view(-1,shape[2],shape[3],shape[4])
        #x: [192, 3, 128, 128]
        x = self.enc(x)
        #x: (bs*8*n) x C x 4 x 4 = 192, 2048, 4, 4
        #bs*8 because of p.view(bs,8*len(models),-1)--see below
        shape = x.shape
        #concatenate the output for tiles into a single map
        x = x.view(-1,n,shape[1],shape[2],shape[3]).permute(0,2,1,3,4).contiguous().view(-1,shape[1],shape[2]*n,shape[3])
        #x: (bs*8) x C x (n*4) x 4 = 16, 2048, 48, 4
        x = self.head(x)
        prim_gleason = self.prim(x)
        sec_gleason  = self.sec(x)
        preds = [prim_gleason,sec_gleason]
        return preds
