def get_model(emb_szs,dls,n_features,loss):
    model=TabNetModel(
        emb_szs,
        n_cont=n_features,
        out_sz=5,
        embed_p=0.0,
        y_range=None,
        n_d=10,
        n_a=18,
        n_steps=2,
        gamma=1.194,
        n_independent=0,
        n_shared=4,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.1623,
    )
    opt_func = partial(Adam, wd=0.01, eps=1e-5)
    learn = Learner(dls, model, loss_func=loss, opt_func=opt_func, metrics=[trends_scorer_multitask_scoring_gpu])
    return learn
