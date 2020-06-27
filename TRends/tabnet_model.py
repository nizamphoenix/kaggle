def get_model(emb_szs,dls,n_features):
    model=TabNetModel(
        emb_szs,
        n_cont=n_features,
        out_sz=5,
        embed_p=0.0,
        y_range=None,
        n_d=32,
        n_a=32,
        n_steps=2,
        gamma=1.194,
        n_independent=0,
        n_shared=4,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.8,
    )
    return model
