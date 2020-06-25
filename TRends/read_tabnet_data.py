def get_tabnet_data(df,train_val_idx):
    targets=['age','domain1_var1','domain1_var2', 'domain2_var1','domain2_var2']
    features=list(set(df.columns)-set(targets))
    to = TabularPandas(
        df=df,
        procs=[Normalize],
        cat_names=None,
        cont_names=features,
        y_names=targets,
        y_block=TransformBlock(),
        splits=train_val_idx,
        do_setup=True,
        device=device,
        inplace=False,
        reduce_memory=True,
    )
    return to,len(features)
