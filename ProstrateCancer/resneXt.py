def get_resnext(layers, pretrained, progress, **kwargs):
    from torchvision.models.resnet import ResNet, Bottleneck
    model = ResNet(Bottleneck, layers, **kwargs)
    model.load_state_dict(torch.load('../input/resnext-50-ssl/semi_supervised_resnext50_32x4-ddb3e555.pth'))
    return model


class GleasonModel(nn.Module):
    def __init__(self, n=4):
        super().__init__()
        #m = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
        #m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models',model='resnext50_32x4d_ssl')
        m = get_resnext([3, 4, 6, 3], pretrained=True, progress=False, groups=32,width_per_group=4)
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
