import fastai
from fastai.vision import *
from fastai.callbacks import SaveModelCallback

def get_isup_preds_targs(preds,target):
    lookup_map = {(0,0):0,(1,1):1,(1,2):2,(2,1):3,(2,2):4,(3,1):4,(1,3):4,(2,3):5,(3,2):5,(3,3):5}
    prim_preds = preds[0].argmax(-1).view(-1,1)
    sec_preds  = preds[1].argmax(-1).view(-1,1)
    temp_preds = torch.cat([prim_preds,sec_preds],dim=1)
    temp = []
    for i in np.array(temp_preds.cpu()):
        try:
            temp.append(lookup_map[tuple(i)])
        except KeyError:
            print(tuple(i)," missins")
            temp.append(2)
    isup_preds = torch.tensor(temp,dtype=torch.long,device='cpu')
    temp = []
    for i in np.array(target.cpu()):
        temp.append(lookup_map[tuple(i)])
    isup_targs = torch.tensor(temp,dtype=torch.long,device='cpu')    
    return isup_preds,isup_targsa

class ConfusionMatrix(Callback):
    "Computes the confusion matrix."

    def on_train_begin(self, **kwargs):
        self.n_classes = 0

    def on_epoch_begin(self, **kwargs):
        self.cm = None

    def on_batch_end(self, last_output:Tensor, last_target:Tensor, **kwargs):
        preds = last_output.argmax(-1).view(-1).cpu()
        targs = last_target.cpu()
        if self.n_classes == 0:
            self.n_classes = last_output.shape[-1]
        if self.cm is None: self.cm = torch.zeros((self.n_classes, self.n_classes), device=torch.device('cpu'))
        cm_temp_numpy = self.cm.numpy()
        np.add.at(cm_temp_numpy, (targs ,preds), 1)
        self.cm = torch.from_numpy(cm_temp_numpy)

    def on_epoch_end(self, **kwargs):
        self.metric = self.cm
        
@dataclass
class KappaScore(ConfusionMatrix):
    "Computes the rate of agreement (Cohens Kappa)."
    weights:Optional[str]=None      # None, `linear`, or `quadratic`
        
    def on_batch_end(self, last_output:Tensor, last_target:Tensor, **kwargs):
#         print("Customised scoring function....")
#         print(last_output[0].shape,last_output[1].shape,last_target.shape)
#         print(last_output)
#         print(last_target)
        preds,targs = get_isup_preds_targs(last_output,last_target)#convert gleasons-->isup for evaluatio

        if self.n_classes == 0:
            self.n_classes = 6 #n_classes in isup_grade
        if self.cm is None: 
            #This executes only once
            self.cm = torch.zeros((self.n_classes, self.n_classes), device=torch.device('cpu'))
        cm_temp_numpy = self.cm.numpy()
        np.add.at(cm_temp_numpy, (targs ,preds), 1)
        self.cm = torch.from_numpy(cm_temp_numpy)
        
        
    def on_epoch_end(self, last_metrics, **kwargs):
        sum0 = self.cm.sum(dim=0)
        sum1 = self.cm.sum(dim=1)
        expected = torch.einsum('i,j->ij', (sum0, sum1)) / sum0.sum()
        if self.weights is None:
            w = torch.ones((self.n_classes, self.n_classes))
            w[self.x, self.x] = 0
        elif self.weights == "linear" or self.weights == "quadratic":
            w = torch.zeros((self.n_classes, self.n_classes))
            w += torch.arange(self.n_classes, dtype=torch.float)
            w = torch.abs(w - torch.t(w)) if self.weights == "linear" else (w - torch.t(w)) ** 2
        else: raise ValueError('Unknown weights. Expected None, "linear", or "quadratic".')
        k = torch.sum(w * self.cm) / torch.sum(w * expected)
        return add_metrics(last_metrics, 1-k)