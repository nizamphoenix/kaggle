LASSO = 1
RIDGE = 0
ELASTIC = 0.3
x <- model.matrix(age~., train)[,-c(1,2,3,4,5)]
cv <- cv.glmnet(x, train$age, alpha = LASSO)
# Display the best lambda value
cv$lambda.min
model <- glmnet(x, train$age, alpha = LASSO, lambda = cv$lambda.min)
# Display regression coefficients
coef(model)
