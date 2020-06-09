import torch
import transformers
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn import model_selection
from transformers import AdamW,get_linear_schedule_with_warmup
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
from scipy import stats
import sys
import warnings
warnings.filterwarnings('ignore')


class Head(torch.nn.Module):
    '''
    replacing top layer
    '''
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
def calculate_loss(preds, targets):
    return F.binary_cross_entropy(F.sigmoid(preds), targets)

            
def train_loop_fn(data_loader,model,optimizer,device,scheduler=None,history=None):
    model.train()
    for bi,(img_batch, y_batch) in enumerate(data_loader):
        img_batch = img_batch.to(device,dtype=torch.float)#Casting to TPUs
        y_batch = y_batch.to(device,dtype=torch.float)#Casting to TPUs
        optimizer.zero_grad()
        preds_batch = model(img_batch)
        loss = calculate_loss(preds_batch, y_batch)
        loss.backward()
        xm.optimizer_step(optimizer)
        if scheduler is not None:
            scheduler.step()
        if bi%10==0:#can make change here
            xm.master_print(f"bi={bi},loss={loss}")
            
            
def eval_loop_fn(data_loader,model,device):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    for bi,(img_batch, y_batch) in enumerate(data_loader):
        img_batch = img_batch.to(device,dtype=torch.float)#Casting to TPUs
        y_batch = y_batch.to(device,dtype=torch.float)#Casting to TPUs
        optimizer.zero_grad()
        preds_batch = model(img_batch)
        fin_targets.append(y_batch.cpu().detach().numpy())
        fin_outputs.append(preds_batch.cpu().detach().numpy())
        return np.vstack(fin_outputs),np.vstack(fin_targets)
    
def run(index):
    MAX_LEN = 512
    TRAIN_BATCH_SIZE = 32
    EPOCHS = 50
    
    
    train_dataset = train_dataset
    train_sampler = torch.utils.data.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        sampler=train_sampler
    )
    
    valid_dataset = val_dataset
    valid_sampler = torch.utils.data.DistributedSampler(
        valid_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
    )
    valid_data_loader=torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=32,#can make changes here
        sampler=valid_sampler
    )
    
    device = xm.xla_device() # delegating to TPUs
    
    lr = 2e-5 * xm.xrt_world_size()#can make changes here
    num_train_steps = int(len(train_dataset) / TRAIN_BATCH_SIZE / xm.xrt_world_size() * EPOCHS)
    
    
    model = FCN(model, 2048).to(device)
    PATH = '../input/mymodels/model_niz.pth'
    model.load_state_dict(torch.load(PATH))
    
    optimizer=AdamW(model.parameters(),lr=lr,eps = 1e-8)#eps = 1e-8: to prevent any division by zero 
    
    scheduler=get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    for epoch in range(EPOCHS):
        para_loader=pl.ParallelLoader(train_data_loader,[device])
        train_loop_fn(para_loader.per_device_loader(device),model,optimizer,device,scheduler)
        
        para_loader=pl.ParallelLoader(valid_data_loader,[device])
        o,t=eval_loop_fn(para_loader.per_device_loader(device),model,device)

        log_loss=[]
        for jj in range(t.shape[1]):
            p1=list(t[:,jj])
            p2=list(o[:,jj])
            l = np.nan_to_num(calculate_loss(p1,p2))
            log_loss.append(l)
        log_loss = np.mean(log_loss)
        xm.master_print(f"epoch={epoch},spearman={log_loss}")
        xm.save(model.state_dict(),"model3.bin")#change every time
        

if __name__=="__main__":
    xmp.spawn(run,nprocs=8)#running on a TPU v3-8, on GCP
