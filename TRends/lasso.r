
library(tidyverse)
library(caret)
library(glmnet)
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


get.data <- function(fnc.path,loading.path,target.path) {
fnc_df = read.csv(fnc.path)
loading_df = read.csv(loading.path)
labels_df = read.csv(target.path)
require(tidyr)#drop_na()
df <- merge(fnc_df,loading_df,by="Id")
labels_df$is.train  = T 
df <- merge(x=labels_df, y=df, all.x=TRUE)
df<-drop_na(df)
df<-df[,-c(1,7)]
head(df)
return(df)
}
data<-get.data("../input/trends-assessment-prediction/fnc.csv","../input/trends-assessment-prediction/loading.csv","../input/trends-assessment-prediction/train_scores.csv")
smp_size <- floor(0.75 * nrow(data))
## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(data)), size = smp_size)
train <- data[train_ind, ]
test <- data[-train_ind, ]
dim(train)
dim(test)
head(train)
library(tidyverse)
library(caret)
library(glmnet)
library(doMC)
library(tictoc)
set.seed(123)
x <- model.matrix(domain1_var1~., train)[,-c(1,2,3,4,5)]#input must be in model.matrix()
y <- train$domain1_var1
tic("cv-ing....")
#1--lassoo 0---ridge
cv <- cv.glmnet(x,y, alpha = 1,parallel = TRUE
# Performing CV to get the best lambda value
toc()
cv$lambda.min
#fitting using optimum lambda(penalty)
model <- glmnet(x, y, alpha = 1, lambda =0.1)
print(model)
#Elastic-net using caret library....CV performed to select best alpha & lambda
set.seed(123)
tic("training")
model <- train(
  domain1_var2 ~., data = train[,-c(1,2,4,5)], method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneLength = 10, parallel=TRUE
)
toc()
# Best tuning parameter
model$bestTune
#coef(model$finalModel, model$bestTune$lambda)
