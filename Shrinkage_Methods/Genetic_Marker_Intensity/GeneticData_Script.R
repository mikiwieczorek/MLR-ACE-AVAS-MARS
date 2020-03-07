library(tidyr)
library(dplyr)
library(ggplot2)
library(nnet)
library(Ecfun)
library(car)
library(ISLR)
library(MASS)
library(glmnet)


#setwd("~/OneDrive - MNSCU/myGithub/Statistics/Regression_models/Multiple_Linear_Regression/MLR-ACE-AVAS-MARS/Shrinkage_Methods")
#Data prep
lu = read.csv("Lu2004.csv")
X = scale(model.matrix(Age~., data = lu)[,-1])
#Xs = scale(X)
y = lu$Age

#Explore the response variable

hist(lu$Age)
Statplot(y)
BCtran(lu$Age)
y.trans = bcPower(lu$Age, 0.5)
Statplot(y.trans)

#PART A
grid = 10^seq(10,-2,length = 200)

#ridge
ridge.mod = glmnet(X,y,alpha=0,lambda=grid)
plot(ridge.mod)
plot(ridge.mod,xvar="lambda")

#lasso
lasso.mod = glmnet(X,y,alpha=1,lambda=grid)
plot(lasso.mod)
plot(lasso.mod, xvar = "lambda")

#elastic
en.mod = glmnet(X,y,alpha=0.5, lambda = grid)
plot(en.mod)
plot(en.mod,xvar="lambda")



#Ridge regression
cv.ridge= cv.glmnet(X, y, alpha = 0)
plot(cv.ridge)
bestlam.ridge = cv.ridge$lambda.min
log(bestlam.ridge)
#ridge.results = glmnet.ssmc(X,y, p=.75, M=1000,alpha=0,lambda=bestlam.ridge)


#Transformed resposne ridge regression
y.back.trans = invBoxCox(y.trans, 0.5)
cv.ridge.trans= cv.glmnet(X, y.trans, alpha = 0)
plot(cv.ridge.trans)
bestlam.ridge.trans = cv.ridge.trans$lambda.min
log(bestlam.ridge.trans)
#ridge.trans.results = glmnet.ssmc(X,y.back.trans,p=.75,M=1000,alpha=0,lambda=bestlam.ridge.log)


#Lasso regression
cv.lasso = cv.glmnet(X, y, alpha = 1)
plot(cv.lasso)
bestlam.lasso = cv.lasso$lambda.min
log(bestlam.lasso)
#lasso.results = glmnet.ssmc(X,y,p=.75, M=1000,alpha=1,lambda=bestlam.lasso)

#Transformed response lasso 
cv.lasso.trans = cv.glmnet(X, y.trans, alpha = 1)
plot(cv.lasso.trans)
bestlam.lasso.trans = cv.lasso.trans$lambda.min
log(bestlam.lasso.trans)
#lasso.trans.results = glmnet.ssmc(X,y.back.trans,p=.75,M=1000,alpha=1,lambda=bestlam.lasso.trans)


#Elastic net
cv.en = cv.glmnet(X,y,alpha=0.1)
plot(cv.en)
bestlam.en = cv.en$lambda.min
log(bestlam.en)
#en.results = glmnet.ssmc(X,y,p=.75, M=1000,alpha=0.1,lambda=bestlam.en)

#Transformed response en 
cv.en.trans = cv.glmnet(X,y.trans,alpha=0.1)
plot(cv.en.trans)
bestlam.en.trans = cv.en.trans$lambda.min
log(bestlam.en.trans)
#en.trans.results = glmnet.ssmc(X,y.back.trans, M=1000,alpha=0.1,lambda=bestlam.en.trans)


# Fit the optimal ridge and Lasso regression models and construct plots of the predicted ages (y ̂) vs. actual age (y).
ridge.mod = glmnet(X,y,alpha=0,lambda=bestlam.ridge)
plot(y,predict(ridge.mod,newx=X),xlab="Actual Age (y-values)",ylab="Predicted Age (yhat-values)")

ridge.cor = cor(y, predict(ridge.mod, newx = X))
ridge.rsqaured = cor(y, predict(ridge.mod, newx = X))^2

lasso.mod = glmnet(X,y,alpha=1,lambda=bestlam.lasso)
plot(y,predict(lasso.mod,newx=X),xlab="Actual Age (y-values)",ylab="Predicted Age (yhat-values)")

en.mod = glmnet(X,y,alpha=.001, lambda =bestlam.en.trans)
plot(y,predict(en.mod,newx=X),xlab="Actual Age (y-values)",ylab="Predicted Age (yhat-values)")

final = glmnet.ssmc(X,y,p=.75, M=1000,alpha=1,lambda=bestlam.lasso)

ridge.results = glmnet.ssmc(X,y, p=.75, M=1000,alpha=0,lambda=bestlam.ridge)
ridge.trans.results = glmnet.ssmc(X,y.back.trans,p=.75,M=1000,alpha=0,lambda=bestlam.ridge.log)

lasso.results = glmnet.ssmc(X,y,p=.75, M=1000,alpha=1,lambda=bestlam.lasso)
lasso.trans.results = glmnet.ssmc(X,y.back.trans,p=.75,M=1000,alpha=1,lambda=bestlam.lasso.trans)

en.results = glmnet.ssmc(X,y,p=.75, M=1000,alpha=0.1,lambda=bestlam.en)
en.trans.results = glmnet.ssmc(X,y.back.trans,p=.75, M=1000,alpha=0.1,lambda=bestlam.en.trans)

#Compare the three methods
names = c("Ridge", "Ridge Transformed", "Lasso", "Lasso Transformed", "Elastic Net", "Elastic Net Transormed")
metrics =cbind(((names)), rbind(
  do.call(cbind, lapply(ridge.results,mean)), 
  do.call(cbind, lapply(ridge.trans.results, mean)),
  do.call(cbind, lapply(lasso.results, mean)),
  do.call(cbind, lapply(lasso.trans.results, mean)),
  do.call(cbind, lapply(en.results, mean)),
  do.call(cbind, lapply(en.trans.results, mean))))

