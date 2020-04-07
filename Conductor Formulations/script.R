#Libraries
{
library(tidyr)
library(dplyr)
library(ggplot2)
library(nnet)
#library(Ecfun)
library(car)
library(ISLR)
#library(MASS)
library(glmnet)
library(pls)
library(corrplot)
require(stringr)
library(gridExtra)
}
#Functions
{
PredAcc = function(y, ypred){
  RMSEP = sqrt(mean((y-ypred)^2))
  MAE = mean(abs(y-ypred))
  MAPE = mean(abs(y-ypred)/y)*100
  cat("RMSEP\n")
  cat("================\n")
  cat(RMSEP, "\n\n")
  cat("MAE\n")
  cat("================\n")
  cat(MAE, "\n\n")
  cat("MAPE\n")
  cat("================\n")
  cat(MAPE, "\n\n")
  return(data.frame(RMSEP = RMSEP, MAE = MAE, MAPE = MAPE))
  
}
myBC = function(y) {
  require(car)
  BCtran(y)
  results = powerTransform(y)
  summary(results)
}
kfold.MLR.log = function(fit,k=10) {
  sum.sqerr = rep(0,k)
  sum.abserr = rep(0,k)
  sum.pererr = rep(0,k)
  y = fit$model[,1]
  y = exp(y)
  x = fit$model[,-1]
  data = fit$model
  n = nrow(data)
  folds = sample(1:k,nrow(data),replace=T)
  for (i in 1:k) {
    fit2 <- lm(formula(fit),data=data[folds!=i,])
    ypred = predict(fit2,newdata=data[folds==i,])
    sum.sqerr[i] = sum((y[folds==i]-ypred)^2)
    sum.abserr[i] = sum(abs(y[folds==i]-ypred))
    sum.pererr[i] = sum(abs(y[folds==i]-ypred)/y[folds==i])
  }
  cv = return(data.frame(RMSEP=sqrt(sum(sum.sqerr)/n),
                         MAE=sum(sum.abserr)/n,
                         MAPE=sum(sum.pererr)/n))
}
bootlog.cv = function(fit,B=100,data=fit$model) {
  yt=fit$fitted.values+fit$residuals
  yact = exp(yt)
  yhat = exp(fit$fitted.values)
  resids = yact - yhat
  ASR=mean(resids^2)
  AAR=mean(abs(resids))
  APE=mean(abs(resids)/yact)
  boot.sqerr=rep(0,B)
  boot.abserr=rep(0,B)
  boot.perr=rep(0,B)
  y = fit$model[,1]
  x = fit$model[,-1]
  n = nrow(data)
  for (i in 1:B) {
    sam=sample(1:n,n,replace=T)
    samind=sort(unique(sam))
    temp=lm(formula(fit),data=data[sam,])
    ytp=predict(temp,newdata=data[-samind,])
    ypred = exp(ytp)
    boot.sqerr[i]=mean((exp(y[-samind])-ypred)^2)
    boot.abserr[i]=mean(abs(exp(y[-samind])-ypred))
    boot.perr[i]=mean(abs(exp(y[-samind])-ypred)/exp(y[-samind]))
  }
  ASRo=mean(boot.sqerr)
  AARo=mean(boot.abserr)
  APEo=mean(boot.perr)
  OPsq=.632*(ASRo-ASR)
  OPab=.632*(AARo-AAR)
  OPpe=.632*(APEo-APE)
  RMSEP=sqrt(ASR+OPsq)
  MAEP=AAR+OPab
  MAPEP=(APE+OPpe)*100
  cat("RMSEP\n")
  cat("===============\n")
  cat(RMSEP,"\n\n")
  cat("MAE\n")
  cat("===============\n")
  cat(MAEP,"\n\n")
  cat("MAPE\n")
  cat("===============\n")
  cat(MAPEP,"\n\n")
  return(data.frame(RMSEP=RMSEP,MAE=MAEP,MAPE=MAPEP))  
}
nnet.sscv = function(x,y,fit,data,p=.667,B=10,size=5,decay=.001,skip=F,linout=T,maxit=25000) {
  require(nnet)
  n = length(y)
  MSEP = rep(0,B)
  MAEP = rep(0,B)
  MAPEP = rep(0,B)
  ss = floor(n*p)
  for (i in 1:B) {
    sam = sample(1:n,ss,replace=F)
    fit2 = nnet(formula(fit),size=size,linout=linout,skip=skip,decay=decay,maxit=maxit,
                trace=F,data=data[sam,])
    yhat = predict(fit2,newdata=x[-sam,])
    ypred = exp(yhat)
    yact = exp(y[-sam])
    MSEP[i] = mean((ypred-yact)^2)
    MAEP[i] = mean(abs(ypred-yact))
    MAPEP[i] = mean(abs(ypred-yact)/yact)
  }
  RMSEP = sqrt(mean(MSEP))
  MAE = mean(MAEP)
  MAPE = mean(MAPEP)
  cat("RMSEP\n")
  cat("=============================\n")
  cat(RMSEP,"\n\n")
  cat("MAE\n")
  cat("=============================\n")
  cat(MAE,"\n\n")
  cat("MAPE\n")
  cat("=============================\n")
  cat(MAPE*100,"\n\n")
  temp = data.frame(RMSEP=sqrt(MSEP),MAEP=MAEP,MAPEP=MAPEP*100)
  return(temp)
}
#Monte Carlo Cross-Validation of Ridge and Lasso Regression
glmnet.ssmc = function(X,y,p=.667,M=100,alpha=1,lambda=1) {
  RMSEP = rep(0,M)
  MAEP = rep(0,M)
  MAPEP = rep(0,M)
  n = nrow(X)
  for (i in 1:M) {
    ss = floor(n*p)
    sam = sample(1:n,ss,replace=F)
    fit = glmnet(X[sam,],y[sam],lambda=lambda,alpha=alpha)
    ypred = predict(fit,newx=X[-sam,])
    RMSEP[i] = sqrt(mean((y[-sam]-ypred)^2))
    MAEP[i] = mean(abs(y[-sam]-ypred))
    yp = ypred[y[-sam]!=0]
    ya = y[-sam][y[-sam]!=0]
    MAPEP[i]=mean(abs(yp-ya)/ya)
  }
  cat("RMSEP =",mean(RMSEP),"  MAEP=",mean(MAEP),"  MAPEP=",mean(MAPEP))
  cv = return(data.frame(RMSEP=RMSEP,MAEP=MAEP,MAPEP=MAPEP)) 
}
#when response is logged
glmnet.sslog = function(X,y,p=.667,M=100,alpha=1,lambda=1) {
  RMSEP = rep(0,M)
  MAEP = rep(0,M)
  MAPEP = rep(0,M)
  n = nrow(X)
  for (i in 1:M) {
    ss = floor(n*p)
    sam = sample(1:n,ss,replace=F)
    fit = glmnet(X[sam,],y[sam],lambda=lambda,alpha=alpha)
    ypred = predict(fit,newx=X[-sam,])
    ya = exp(y[-sam])
    ypred = exp(ypred)
    RMSEP[i] = sqrt(mean((ya-ypred)^2))
    MAEP[i] = mean(abs(ya-ypred))
    MAPEP[i]=mean(abs(ypred-ya)/ya)
  }
  cat("RMSEP =",mean(RMSEP),"  MAEP=",mean(MAEP),"  MAPEP=",mean(MAPEP))
  cv = return(data.frame(RMSEP=RMSEP,MAEP=MAEP,MAPEP=MAPEP))
  
}
#Monte Carlo Cross-Validation of OLS Regression Models
MLR.ssmc = function(fit,p=.667,M=100) {
  RMSEP = rep(0,M)
  MAEP = rep(0,M)
  MAPEP = rep(0,M)
  y = fit$model[,1]
  x = fit$model[,-1]
  data = fit$model
  n = nrow(data)
  for (i in 1:M) {
    ss = floor(n*p)
    sam = sample(1:n,ss,replace=F)
    fit2 = lm(formula(fit),data=data[sam,])
    ypred = predict(fit2,newdata=x[-sam,])
    RMSEP[i] = sqrt(mean((y[-sam]-ypred)^2))
    MAEP[i] = mean(abs(y[-sam]-ypred))
    yp = ypred[y[-sam]!=0]
    ya = y[-sam][y[-sam]!=0]
    MAPEP[i]=mean(abs(yp-ya)/ya)
  }
  cat("RMSEP =",mean(RMSEP),"  MAEP=",mean(MAEP),"  MAPEP=",mean(MAPEP))
  cv = return(data.frame(RMSEP=RMSEP,MAEP=MAEP,MAPEP=MAPEP))
}
}
#Load data:
{setwd("~/OneDrive - MNSCU/myGithub/Supervised_Learning/Multiple_Linear_Regression/MachineLearning-SupervisedLearning/Conductor Formulations")
#Train set
cond_train = read.csv("Conductors (train).csv")
#Test set
cond_test = read.csv("Conductors (test).csv")
}
#Preparing data: transform response
{
cond_train %>%
mutate(formation_energy_ev_natom_log_y_plus_1 = log(formation_energy_ev_natom + 1),
       bandgap_energy_ev_log_y_plus_1 = log(bandgap_energy_ev + 1)) -> cond_train_trans
drop.cols <- c("formation_energy_ev_natom", "bandgap_energy_ev")

cond_train_trans = cond_train_trans[,-c(1,13,14)]

y_train_form_eng_trans = cond_train_trans$formation_energy_ev_natom_log_y_plus_1
y_train_bandgap_eng_trans = cond_train_trans$bandgap_energy_ev_log_y_plus_1
x_train = cond_train_trans[,-c(12,13)]
}
#Rearrange the data for two models
{#detach("package:Ecfun", unload = TRUE)
  cond_train_trans %>%
  select(12,1:11) -> cond_train_trans_model1
  
  cond_train_trans%>%
    select(13, 1:11) -> cond_train_trans_model2
} 

#Model 1: resposne formation of energy
#resposne" y_train_form_eng_trans
#x's: x_train: 11 variables

#Explore pairs.plus
pairs.plus(x_train)
x_train.mat = as.matrix(x_train)
corr.x_train = cor(x_train.mat)
corrplot(corr.x_train, order = 'hclust')
summary(corr.x_train)
?corrplot


####PCA
x_train.scale = scale(x_train.mat)
cond.pca = princomp(x_train.scale)
summary(cond.pca)
par(mfrow=c(1,1))
biplot(cond.pca, cex=0.2)

{
  PC1 = cond.pca$x[,1]
  PC2 = cond.pca$x[,2]
  PC3 = cond.pca$x[,3]
  par(mfrow=c(1,1))
  plot(PC1, PC2, type = "n")
  text(PC1, PC2, as.character(y_train_form_eng_trans), cex=0.5, col = "black")
  #text(PC1, PC2, paste(as.numeric(lu$district_number),":"),adj=c(2.9,.5), cex = .5, col = "blue")
  abline(h=0, v=0, lty=2,col="red",lwd=1)
}
require(factoextra)
require(FactoMineR)
cond.PCA = PCA(x_train.scale)
summary(cond.PCA)
#Selecting only 8 features whose quality of representaiton was the highest:
fviz_pca_var(cond.pca, repel = T, select.var = list(cos2 = 5), col.var = "cos2")
#Selecting only 8 features whose contribution was the highest:
fviz_pca_var(cond.pca, repel = T, select.var = list(contrib = 5), col.var = "contrib")
#Biplot with highest quality of representation cos2=8 variables
#fviz_pca_biplot(cond.pca, repel = F, select.var = list(cos2=8), col.var = "cos2")

#fviz_pca_biplot(lu.pca, repel = T, select.var = list(contrib=6),col.var = "contrib", cex=.4)

?write.csv


#LOADINGSPar can be used to set or query graphical parameters
{par(mfrow=c(2,1)) #will have 3 graphs horizontally
  plot(lu.pca$rotation[,1], type = "h", col="blue", xlab = "Variable", ylab = "PC1 Loadings")
  plot(lu.pca$rotation[,2], type = "h", col="blue", xlab = "Variable", ylab = "PC2 Loadings")
}
#Scree plot
{par(mfrow=c(1,1))
  plot(cond.pca, type = "line", main = "Scree Plot", xlim = c(1,10), ylim = c(0,10))
  abline(h=1, lty=2,col="red",lwd=2)
}
{fviz_screeplot(cond.pca, addlabels = TRUE, ylim = c(0, 45))
}

{
  Investigate(lu.PCA, file = "PCA.Rmd", document = "html_document")
} 
  
  ###### Filter on the top contributors for variables cos2> or contrb >
  
write.csv(cond_train_trans, col.names = TRUE, row.names = FALSE, file = "cond_train_trans.csv")


#Model 2: resposne bandgap energy
#resposne: y_train_bandgap_eng_trans
#x's: x_train
  
  
  