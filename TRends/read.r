get.data <- function(fnc.path,loading.path,target.path) {
fnc_df = read.csv(fnc.path)
loading_df = read.csv(loading.path)
labels_df = read.csv(target.path)
library(tidyr)#drop_na()
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
