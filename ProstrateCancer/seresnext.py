class Gleason(nn.Module):
    def __init__(self):
        super().__init__()
        m = se_resnext50_32x4d(4,loss='softmax', pretrained=True)
        self.enc = nn.Sequential(*list(m.children())[:-2])       
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(AdaptiveConcatPool2d(),
                                  Flatten(),
                                  nn.Linear(2*nc,512),
                                  Mish(),
                                  nn.BatchNorm1d(512), 
                                  nn.Dropout(0.3))
        self.prim =  nn.Sequential(nn.Linear(512,128),nn.ReLU(),nn.BatchNorm1d(128),nn.Dropout(0.3),nn.Linear(128,4))
        self.sec  =  nn.Sequential(nn.Linear(512,128),nn.ReLU(),nn.BatchNorm1d(128),nn.Dropout(0.3),nn.Linear(128,4))
      
      
        
        
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
        preds = [prim_gleason,sec_gleason]
        return preds
