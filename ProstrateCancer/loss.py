import torch
import torch.nn as nn

class MultiTaskGleasonLoss(nn.Module):
    def __init__(self, task_num):
        super(MultiTaskGleasonLoss, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, preds, targets):
#         print("type(preds):",type(preds),"| len(preds):",len(preds),"| preds[0].shape:",preds[0].shape,"| preds[1].shape:",preds[1].shape)
#         print("type(targets):",type(targets),"| targets[:,0].shape:",targets[:,0].shape,"| targets[:,1].shape:",targets[:,1].shape)
        crossEntropy = nn.CrossEntropyLoss()
#         print("Before typecasting: ",preds[0].device,preds[0].dtype,targets[:,0].device,targets[:,0].dtype)
        prim_preds  = preds[0]
        prim_target = targets[:,0].long()
        loss1 = crossEntropy(prim_preds,prim_target)
#         print("After typecasting: ",prim_preds.device,prim_preds.dtype,prim_target.device,prim_target.dtype)
        precision1 = torch.exp(-self.log_vars[0])
        loss1 = precision1*loss1 + self.log_vars[0]   
        
        sec_preds  = preds[1]
        sec_target = targets[:,1].long()
        loss2 = crossEntropy(sec_preds,sec_target)
        precision2 = torch.exp(-self.log_vars[1])
        loss2 = precision2*loss2 + self.log_vars[1]   
        
        return loss1+loss2

class BiTaskGleasonLoss(nn.Module):
    def __init__(self, task_num):
        super(BiTaskGleasonLoss, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, preds, targets):
        temp=[]
        primloss = nn.CrossEntropyLoss(weight=prim_weights)
        prim_preds  = preds[0]
        prim_target = targets[:,0].long()
        loss0 = primloss(prim_preds,prim_target)
        temp.append(loss0)
        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0*loss0 + self.log_vars[0]   
        
        secloss  = nn.CrossEntropyLoss(weight=sec_weights)
        sec_preds  = preds[1]
        sec_target = targets[:,1].long()
        loss1 = secloss(sec_preds,sec_target)
        temp.append(loss1)
        precision1 = torch.exp(-self.log_vars[1])
        loss1 = precision1*loss1 + self.log_vars[1]   
        
        
        print("precisions:", precision0,precision1)
        print(self.log_vars[0],self.log_vars[1].dtype)
        print("simple loss:",sum(temp))
        print("complicated loss:","loss0+loss1:",loss0+loss1,"loss0+loss1:",loss0+loss1)
        return loss1+loss0
    
class TriTaskGleasonLoss(nn.Module):
    def __init__(self, task_num):
        super(TriTaskGleasonLoss, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, preds, targets):
#         print("type(preds):",type(preds),"| len(preds):",len(preds),"| preds[0].shape:",preds[0].shape,"| preds[1].shape:",preds[1].shape,"| preds[2].shape:",preds[2].shape)
#         print("type(targets):",type(targets),"| targets[:,0].shape:",targets[:,0].shape,"| targets[:,1].shape:",targets[:,1].shape,"| targets[:,2].shape:",targets[:,2].shape)
        crossEntropy = nn.CrossEntropyLoss()
        prim_preds  = preds[0]
        prim_target = targets[:,0].long()
        loss0 = crossEntropy(prim_preds,prim_target)
        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0*loss0 + self.log_vars[0]   
        
        sec_preds  = preds[1]
        sec_target = targets[:,1].long()
        loss1 = crossEntropy(sec_preds,sec_target)
        precision1 = torch.exp(-self.log_vars[1])
        loss1 = precision1*loss1 + self.log_vars[1]   
        
        isup_preds  = preds[2]
        isup_target = targets[:,2].long()
        loss2 = crossEntropy(isup_preds,isup_target)
        precision2 = torch.exp(-self.log_vars[2])
        loss2 = precision2*loss2 + self.log_vars[2]   
        
        return loss0+loss1+loss2
