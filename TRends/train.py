def cv_train_predict(df,test_df,features):
    NUM_FOLDS = 7
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)
    overal_score = 0
    for target, c, w in [("age", 100, 0.3), ("domain1_var1", 10, 0.175), ("domain1_var2", 10, 0.175), ("domain2_var1", 10, 0.175), ("domain2_var2", 10, 0.175)]:    
        y_oof = np.zeros(df.shape[0])
        y_test = np.zeros((test_df.shape[0], NUM_FOLDS))

        for f, (train_ind, val_ind) in enumerate(kf.split(df, df)):
            train_df, val_df = df.iloc[train_ind], df.iloc[val_ind]
            train_df = train_df[train_df[target].notnull()]
            model = SVR(C=c, epsilon=0.5, cache_size=3000.0)#cache_size of the kernel(in MB).C=inverse prop to l2regularisation.
            model.fit(train_df[features], train_df[target])
            y_oof[val_ind] = model.predict(val_df[features])
            y_test[:, f] = model.predict(test_df[features])
        df["pred_{}".format(target)] = y_oof
        test_df[target] = y_test.mean(axis=1)
        score = metric(df[df[target].notnull()][target].values, df[df[target].notnull()]["pred_{}".format(target)].values)
        overal_score += w*score
        print(target, np.round(score, 4))
        print()

    print("Overal score:", np.round(overal_score, 4))
