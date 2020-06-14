from sklearn.linear_model import MultiTaskElasticNet,MultiTaskElasticNetCV

#cross-validating to find best hyperparams
cv_model = MultiTaskElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],verbose=1)
cv_model.fit(X_train,y_train)

#fitting model with hyperparameters from above
model = MultiTaskElasticNet(alpha=cv_model.alpha_, l1_ratio=cv_model.l1_ratio_,random_state=0)
model.fit(X_train,y_train)

#predicting
preds = model.predict(X_test)
test_df[['age','domain1_var1','domain1_var2', 'domain2_var1','domain2_var2']] = preds
test_df.drop(columns=["is_train"],inplace=True)
test_df.head()

#predictions housekeeping
sub_df = cudf.melt(test_df[["Id", "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]], id_vars=["Id"], value_name="Predicted")
sub_df["Id"] = sub_df["Id"].astype("str") + "_" +  sub_df["variable"].astype("str")
sub_df = sub_df.drop("variable", axis=1).sort_values("Id")
assert sub_df.shape[0] == test_df.shape[0]*5
sub_df.head(10)
