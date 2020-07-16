def get_data()
    import pandas as pd
    susp = pd.read_csv('../input/suspicious/PANDA_Suspicious_Slides.csv')
    # ['marks', 'No Mask', 'Background only', 'No cancerous tissue but ISUP Grade > 0', 'tiss', 'blank']
    to_drop = susp.query("reason in ['marks','Background only','tiss','blank']")['image_id']
    df = pd.read_csv(LABELS).set_index('image_id')
    good_index = list(set(df.index)-set(to_drop))
    df = df.loc[good_index]
    df = df.reset_index()
    splits = StratifiedKFold(n_splits=nfolds, random_state=SEED, shuffle=True)
    splits = list(splits.split(df,df.isup_grade))
    folds_splits = np.zeros(len(df)).astype(np.int)
    for i in range(nfolds): 
        folds_splits[splits[i][1]] = i
    df['split'] = folds_splits
    df['gleason_score']=df['gleason_score'].replace('negative','0+0')
    df[['prim_gleason','secon_gleason']] = df.gleason_score.str.split("+",expand=True)
    df[['prim_gleason','secon_gleason']] = df[['prim_gleason','secon_gleason']].astype(np.int64)
    df['prim_gleason']=df['prim_gleason'].replace(3,1)
    df['prim_gleason']=df['prim_gleason'].replace(4,2)
    df['prim_gleason']=df['prim_gleason'].replace(5,3)
    df['secon_gleason']=df['secon_gleason'].replace(3,1)
    df['secon_gleason']=df['secon_gleason'].replace(4,2)
    df['secon_gleason']=df['secon_gleason'].replace(5,3)
    print("****************df shape:",df.shape,"***********************")
    print(">>>>>>>>>Before sampling<<<<<<<<<<<<<")
    for isup in [0,1,2,3,4,5]:
        print("isup grade:",isup,"| n_instances:",df.query('isup_grade=={0}'.format(isup)).shape[0],"| corresponding gleason score:",df[['isup_grade','gleason_score']].query('isup_grade=={0}'.format(isup))['gleason_score'].unique())
        print("----"*20)
    #df.drop([df[df['image_id']=="b0a92a74cb53899311acc30b7405e101"].index[0]],inplace=True)
    #b0a92a74cb53899311acc30b7405e101 is the only image id with gleason 4+3 mapping to isup=2
    df = pd.concat([df.query('isup_grade==0').iloc[:1200],df.query('isup_grade==1').iloc[:1200],df.query('isup_grade==2 or isup_grade==3 or isup_grade==4 or isup_grade==5')],axis=0)
    df = df.sample(n=2000,random_state=SEED).reset_index(drop=True)#shuffling
    print(">>>>>>>>>After sampling<<<<<<<<<<")
    for isup in [0,1,2,3,4,5]:
        print("isup grade:",isup,"| n_instances:",df.query('isup_grade=={0}'.format(isup)).shape[0],"| corresponding gleason score:",df[['isup_grade','gleason_score']].query('isup_grade=={0}'.format(isup))['gleason_score'].unique())
        print("----"*20)
    return df
df = get_data()
df[['isup_grade','split']].hist(bins=50)
df.head()
