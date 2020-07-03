def get_data(fold=0):
    return (MImageItemList.from_df(df, path=TRAIN+"/train", cols='image_id')
      .split_by_idx(df.index[df.split == fold].tolist())
      .label_from_df(cols=['prim_gleason','secon_gleason'])
      .transform(get_transforms(flip_vert=True,max_rotate=15),size=sz,padding_mode='zeros')
      .databunch(bs=bs,num_workers=4,device='cuda'))
