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

