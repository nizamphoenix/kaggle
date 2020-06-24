import torch.nn
import numpy as np


from torch.nn import SmoothL1Loss
class MSEMAELOSS(torch.nn.Module):
    from fastai2.layers import L1LossFlat,MSELossFlat
    '''
    For use with GPU only
    MSELOSS*(1-x) + MAELOSS*x
    '''
    def __init__(self,l1):
        super().__init__()
        self.l1=l1
        
    def forward(self,y, y_hat):
        loss = (1-self.l1)*MSELossFlat()(y, y_hat) + self.l1*L1LossFlat()(y, y_hat)
        return loss
    
class Regress_Loss(torch.nn.Module):
    
    def __init__(self):
        super(Regress_Loss,self).__init__()
        
    def forward(self,x,y):
        y_shape = y.size()[1]
        x_added_dim = x.unsqueeze(1)
        x_stacked_along_dimension1 = x_added_dim.repeat(1,NUM_WORDS,1)
        diff = torch.sum((y - x_stacked_along_dimension1)**2,2)
        totloss = torch.sum(torch.sum(torch.sum(diff)))
        return totloss
    
class GammaLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,y, y_hat):
        p=2
        loss = - y * torch.pow(y_hat, 1 - p) / (1 - p) + torch.pow(y_hat, 2 - p) / (2 - p)
        return torch.mean(loss)

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
#         y_t=y_t.cpu().detach().numpy()
#         y_prime_t=y_prime_t.cpu().detach().numpy()
        ey_t = torch.abs(y_t - y_prime_t)
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-16)))
