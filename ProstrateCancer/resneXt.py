class Model(nn.Module):
    def __init__(self, arch='resnext50_32x4d_ssl', n=4, pre=True):
        super().__init__()
        
        m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)
        #m = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)

        self.enc = nn.Sequential(*list(m.children())[:-2])       
        nc = list(m.children())[-1].in_features
        self.primary_head = nn.Sequential(AdaptiveConcatPool2d(),
                                  Flatten(),
                                  nn.Linear(in_features=2*nc,out_features=512),
                                  Mish(),
                                  nn.BatchNorm1d(512), 
                                  nn.Dropout(0.5),
                                  nn.Linear(in_features=512,out_features=n))
        self.secondary_head = nn.Sequential(AdaptiveConcatPool2d(),
                                  Flatten(),
                                  nn.Linear(in_features=2*nc,out_features=512),
                                  Mish(),
                                  nn.BatchNorm1d(512), 
                                  nn.Dropout(0.5),
                                  nn.Linear(in_features=512,out_features=n))
        
    def forward(self, *x):
        shape = x[0].shape
        n = len(x)
        x = torch.stack(x,dim=1).view(-1,shape[1],shape[2],shape[3])
        #x: bs*N x 3 x 128 x 128
        x = self.enc(x)
        #x: bs*N x C x 4 x 4
        shape = x.shape
        #concatenate the output for tiles into a single map
        x = x.view(-1,n,shape[1],shape[2],shape[3]).permute(0,2,1,3,4).contiguous().view(-1,shape[1],shape[2]*n,shape[3])
        #x: bs x C x N*4 x 4
        primary_gleason_logit = self.primary_head(x)
        #x: bs x n
        secondary_gleason_logit = self.secondary_head(x)
        return torch.cat([primary_gleason_logit, secondary_gleason_logit], axis=1)
