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

    
models = []
for path in MODELS:
    print(path)
    model = GleasonModel()
    model_dict = model.state_dict()
    pretrained_dict = torch.load(path,map_location=torch.device('cpu'))
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    model.float()
    model.eval()
    model.cuda()
    models.append(model)
    
del pretrained_dict



def tile(img):
    shape = img.shape
    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz
    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],constant_values=255)
    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    if len(img) < N:
        img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]
    img = img[idxs]
    return img

mean = torch.tensor([1.0-0.90949707, 1.0-0.8188697, 1.0-0.87795304])
std  = torch.tensor([0.36357649, 0.49984502, 0.40477625])

class PandaDataset(Dataset):
    def __init__(self, path, test):
        self.path = path
        self.names = list(pd.read_csv(test).image_id)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img = skimage.io.MultiImage(os.path.join(DATA,name+'.tiff'))[-1]
        tiles = torch.Tensor(1.0 - tile(img)/255.0)
        tiles = (tiles - mean)/std
        return tiles.permute(0,3,1,2), name
    
 def get_isup_preds(preds):
    '''
    converts gleason scores predicted by the model to isup_grades.
    returns a torch tensor of isup_preds
    '''
    lookup_map  = {(0,0):0,(1,1):1,(1,2):2,(2,1):3,(2,2):4,(3,1):4,(1,3):4,(2,3):5,(3,2):5,(3,3):5}
    lookup_map2 = {(0,1):1,(0,2):1,(0,3):2,(1,0):1,(2,0):3,(3,0):4}
    
    prim_preds = torch.stack([preds[0][0],preds[1][0],preds[2][0],preds[3][0]],dim=1)
    prim_preds = prim_preds.view(bs,8*len(models),-1).mean(dim=1).argmax(-1).cpu()#shape=2
    sec_preds  = torch.stack([preds[0][1],preds[1][1],preds[2][1],preds[3][1]],dim=1)
    sec_preds  = sec_preds.view(bs,8*len(models),-1).mean(dim=1).argmax(-1).cpu()#shape=2
    
    temp_preds = torch.cat([prim_preds.view(2,1),sec_preds.view(2,1)],dim=1)
    temp = []
    count = 0
    errors = 0
    for i in np.array(temp_preds.cpu()):
        count+=1
        try:
            temp.append(lookup_map[tuple(i)])
        except KeyError:
            errors+=1
            temp.append(lookup_map2[tuple(i)])
    print("count={0},errors={1}".format(count,errors),",correct=",count-errors)
    isup_preds = torch.tensor(temp,dtype=torch.long,device='cpu')
    return isup_preds



sub_df = pd.read_csv(SAMPLE)
lookup_map  = {(0,0):0,(1,1):1,(1,2):2,(2,1):3,(2,2):4,(3,1):4,(1,3):4,(2,3):5,(3,2):5,(3,3):5}
lookup_map2 = {(0,1):1,(0,2):1,(0,3):2,(1,0):1,(2,0):3,(3,0):4}
if os.path.exists(DATA):
    print("Predicting....")
    ds = PandaDataset(DATA,TEST)
    dl = DataLoader(ds, batch_size=bs, num_workers=nworkers, shuffle=False)
    names,preds = [],[]
    with torch.no_grad():
        for imgs,filenames in tqdm(dl):
            imgs = imgs.cuda()
            #dihedral TTA
            imgs = torch.stack([imgs,imgs.flip(-1),imgs.flip(-2),imgs.flip(-1,-2),
                         imgs.transpose(-1,-2),imgs.transpose(-1,-2).flip(-1),
                        imgs.transpose(-1,-2).flip(-2),imgs.transpose(-1,-2).flip(-1,-2)],dim=1)
            imgs = imgs.view(-1,N,3,sz,sz)
        
            all_preds = [model(imgs) for model in models]
            p=get_isup_preds(all_preds)#for 2(bs) images only
            
            names.append(filenames)
            preds.append(p)
    names = np.concatenate(names)
    preds = torch.cat(preds).numpy()
    sub_df = pd.DataFrame({'image_id': names, 'isup_grade': preds})
    sub_df.to_csv('submission.csv', index=False)
    sub_df.head()
    
sub_df.to_csv("submission.csv", index=False)
sub_df.head()
