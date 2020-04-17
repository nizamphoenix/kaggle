install.packages('NHPoisson',dependencies = T)
library(NHPoisson)
tB <- BarTxTn$ano + rep(c(0:152) / 153,55)
time<-tB
feature<-BarTxTn$Tx
time.features<-cbind(time,feature)
colnames(time.features)<-c('time','obs')#DO NOT CHANGE THE COLUMNS NAME
par(mfrow=c(1,2))
op<-diplot(time.features)
ones<-rep(1,length(op$DI)-1)
diff<-op$DI[1:length(op$DI)-1] - ones * mean(op$DI[1:length(op$DI)-1])

abline(v=mean(op$thresh),col='red')
abline(v=op$thresh[match(max(op$DI[1:length(op$DI)-1]),op$DI)],col='blue')
abline(v=op$thresh[which.min(abs(diff))],col='green')

thresh1<-mean(op$thresh)#is the mean threshold
thresh2<-op$thresh[match(max(op$DI[1:length(op$DI)-1]),op$DI)]#mode threshold
thresh3<-op$thresh[which.min(abs(diff))]
