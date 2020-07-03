def train():
  fname = 'RNXT50'
  pred,target = [],[]
  for fold in range(1):#nfolds
      data = get_data(fold)
      model = Model()
      learn = Learner(data, model, loss_func=nn.CrossEntropyLoss(), opt_func=Over9000, metrics=[KappaScore(weights='quadratic')]).to_fp16()
      logger = CSVLogger(learn, f'log_{fname}_{fold}')
      learn.clip_grad = 1.0
      learn.split([model.head])
      learn.unfreeze()
      learn.fit_one_cycle(1, max_lr=1, div_factor=1, pct_start=0.0,callbacks = [SaveModelCallback(learn,name=f'model',monitor='kappa_score')])
      torch.save(learn.model.state_dict(), f'{fname}_{fold}.pth')

      learn.model.eval()
      with torch.no_grad():
          for step, (x, y) in progress_bar(enumerate(data.dl(DatasetType.Valid)),total=len(data.dl(DatasetType.Valid))):
              p = learn.model(*x)
              pred.append(p.float().cpu())
              target.append(y.cpu())
