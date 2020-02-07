
require(car)
require(acepack)

Concrete <- read.csv("Concrete.csv")

set.seed(1)

sam = sample(1:1030,size=floor(.6666*1030),replace=F)

View(Concrete[sam,])

mod1 = lm(Strength~., data = Concrete[sam,])

par(mfrow=c(1,1))
plot(mod1)

summary(mod1)

VIF(mod1)

# 1b

 PredAcc = function(y,ypred){
     RMSEP = sqrt(mean((y-ypred)^2))
     MAE = mean(abs(y-ypred))
     MAPE = mean(abs(y-ypred)/y)*100
     cat("RMSEP\n")
     cat("===============\n")
     cat(RMSEP,"\n\n")
     cat("MAE\n")
     cat("===============\n")
     cat(MAE,"\n\n")
     cat("MAPE\n")
     cat("===============\n")
     cat(MAPE,"\n\n")
     return(data.frame(RMSEP=RMSEP,MAE=MAE,MAPE=MAPE))}



y = Concrete$Strength[-sam]
ypred = predict(mod1,newdata = Concrete[-sam,])
results = PredAcc(y,ypred)


#1c

ceresPlots(mod1)

pairs.plus(Concrete)

library(dplyr)

Concrete %>%
  select(-c(FineAge, CourseAgg)) -> Concrete2

View(Concrete2)


myBC()


ceresPlot(mod1)


