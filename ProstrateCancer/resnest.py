class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, preds, gleason):
        print(type(preds),len(preds),preds[0].shape,preds[1].shape,type(gleason),gleason.shape)

        crossEntropy = CrossEntropyFlat()
        
        loss1 = crossEntropy(preds[0],prim_gleason)
        loss2 = crossEntropy(preds[1],sec_gleason)

    
        precision1 = torch.exp(-self.log_vars[0])
        loss1 = precision1*loss1 + self.log_vars[0]

        precision2 = torch.exp(-self.log_vars[1])
        loss2 = precision2*loss2 + self.log_vars[1]
        
        return loss1+loss2
    a
class GleasonModel(nn.Module):
    def __init__(self, arch='resnext50_32x4d_ssl', n=4, pre=True):
        super().__init__()
        m = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
        #m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)
        self.enc = nn.Sequential(*list(m.children())[:-2])       
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(AdaptiveConcatPool2d(),
                                  Flatten(),
                                  nn.Linear(2*nc,128),
                                  Mish(),
                                  nn.BatchNorm1d(128), 
                                  nn.Dropout(0.5))
        self.prim =  nn.Linear(128,n)
        self.sec  =  nn.Linear(128,n)
      
        
        
    def forward(self, *x):
        shape = x[0].shape
        n = len(x)
        x = torch.stack(x,1).view(-1,shape[1],shape[2],shape[3])
        #x: bs*N x 3 x 128 x 128
        x = self.enc(x)
        #x: bs*N x C x 4 x 4
        shape = x.shape
        #concatenate the output for tiles into a single map
        x = x.view(-1,n,shape[1],shape[2],shape[3]).permute(0,2,1,3,4).contiguous().view(-1,shape[1],shape[2]*n,shape[3])
        #x: bs x C x N*4 x 4
        x = self.head(x)
        #x: bs x n
        prim_gleason = self.prim(x)
        sec_gleason  = self.sec(x)
        return prim_gleason,sec_gleason
