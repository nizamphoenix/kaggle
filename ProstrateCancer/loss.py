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
