require(nnet)
require(Ecfun)
require(car)
#require(ca)




load("~/OneDrive - MNSCU/myGithub/Statistics/Regression_models/Multiple_Linear_Regression/MLR-ACE-AVAS-MARS/Data/mult.Rdata")
load("~/OneDrive - MNSCU/myGithub/Statistics/Regression_models/Multiple_Linear_Regression/MLR-ACE-AVAS-MARS/Data/Regression.Rdata")
setwd("~/OneDrive - MNSCU/myGithub/Statistics/Regression_models/Multiple_Linear_Regression/MLR-ACE-AVAS-MARS/Data")
Concrete = read.csv("Concrete.csv")
setwd("~/OneDrive - MNSCU/myGithub/Statistics/Regression_models/Multiple_Linear_Regression/MLR-ACE-AVAS-MARS/NeuralNets")


#redo transformations from assignment 2

{
  #Create lm objects and update the model
  Concrete.trans = Concrete
  set.seed(1)
  sam = sample(1:1030, size = floor(.6666*1030), replace = F)
  lm1 = lm(Strength~., data = Concrete.trans[sam,])
  lm2 = update(lm1, Strength~. - CourseAgg - FineAge, data = Concrete.trans[sam,])
  lm2.step = step(lm2)
  #Apply transformations
  #pairs.plus(Concrete.trans)
  Concrete.trans$Age = log(Concrete.trans$Age)
  Concrete.trans$Superplast = yjPower(Concrete.trans$Superplast, 0.3)
  Concrete.trans$BlastFurn = yjPower(Concrete.trans$FlyAsh, -0.1)
  Concrete.trans$BlastFurn = log(Concrete.trans$BlastFurn+1)
  Concrete.trans$Cement = bcPower(Concrete.trans$Cement, 0.2)
  Concrete.trans$Water = bcPower(Concrete.trans$Water, 0.8)
  Concrete.trans$Strength = bcPower(Concrete.trans$Strength, 0.6)
  lm.trans = update(lm2.step, Strength~. , data = Concrete.trans[sam,])
  #Adding polynomials to lm.trans
  lm.poly = update(lm.trans, Strength~. + poly(Superplast, 2) + poly(Age,3) + poly(Water, 2))
  data.frame(R.sq = c(summary(lm.poly)$r.squared*100), Adj.R.sq=c(summary(lm.poly)$adj.r.squared)*100)
  summary(lm.poly)
  write.csv(Concrete.trans, file = "Concrete.trans.csv")
}




#nnet

nnet.sscv =  function(x,y,fit,p=.667,B=100,size=3,decay=fit$decay,skip=T,
                    linout=T,maxit=10000){
  n = length(y)
  MSEP = rep(0,B)
  MAEP = rep(0,B)
  MAPEP = rep(0,B)
  ss = floor(n*p)
  for (i in 1:B){
    sam = sample(1:n,ss,replace=F)
    fit2 = nnet(x[sam,],y[sam],size=size,linout=linout,skip=skip,decay=decay,
                maxit=maxit,trace=F)
    ynew = predict(fit2,newdata=x[-sam,])
    MSEP[i]=mean((y[-sam]-ynew)^2)
    MAEP[i]=mean(abs(y[-sam]-ynew))
    MAPEP[i]=mean(abs(y[-sam]-ynew)/y[-sam])
  }
  RMSEP = sqrt(mean(MSEP))
  MAE = mean(MAEP)
  MAPE = mean(MAPEP)
  cat("RMSEP\n")
  cat("===============\n")
  cat(RMSEP,"\n\n")
  cat("MAE\n")
  cat("===============\n")
  cat(MAE,"\n\n")
  cat("MAPE\n")
  cat("===============\n")
  cat(MAPE,"\n\n")
  temp = data.frame(MSEP=MSEP,MAEP=MAEP,MAPEP=MAPEP)
  return(temp)
}



#nnet log

nnet.sscv.log =  function(x,y,fit,p=.667,B=100,size=3,decay=fit$decay,skip=T,
                      linout=T,maxit=10000){
  n = length(y)
  MSEP = rep(0,B)
  MAEP = rep(0,B)
  MAPEP = rep(0,B)
  ss = floor(n*p)
  for (i in 1:B){
    sam = sample(1:n,ss,replace=F)
    fit2 = nnet(x[sam,],y[sam],size=size,linout=linout,skip=skip,decay=decay,
                maxit=maxit,trace=F)
    ynew = predict(fit2,newdata=x[-sam,])
    ystar = exp(y)
    ynew = exp(ynew)
    MSEP[i]=mean((ystar[-sam]-ynew)^2)
    MAEP[i]=mean(abs(ystar[-sam]-ynew))
    MAPEP[i]=mean(abs(ystar[-sam]-ynew)/ystar[-sam])
    
  }
  RMSEP = sqrt(mean(MSEP))
  MAE = mean(MAEP)
  MAPE = mean(MAPEP)
  cat("RMSEP\n")
  cat("===============\n")
  cat(RMSEP,"\n\n")
  cat("MAE\n")
  cat("===============\n")
  cat(MAE,"\n\n")
  cat("MAPE\n")
  cat("===============\n")
  cat(MAPE,"\n\n")
  temp = data.frame(MSEP=MSEP,MAEP=MAEP,MAPEP=MAPEP)
  return(temp)
}


Concrete.trans$Strength = bcPower(Concrete.trans$Strength, 0.6)


nnet.sscv.undobc =  function(x,y,fit,p=.667,B=100,size=3,decay=fit$decay,skip=T,
                          linout=T,maxit=10000){
  n = length(y)
  MSEP = rep(0,B)
  MAEP = rep(0,B)
  MAPEP = rep(0,B)
  ss = floor(n*p)
  for (i in 1:B){
    sam = sample(1:n,ss,replace=F)
    fit2 = nnet(x[sam,],y[sam],size=size,linout=linout,skip=skip,decay=decay,
                maxit=maxit,trace=F)
    ynew = predict(fit2,newdata=x[-sam,])
    ystar = invBoxCox(y, 0.6)
    ynew = invBoxCox(ynew, 0.6)
    MSEP[i]=mean((ystar[-sam]-ynew)^2)
    MAEP[i]=mean(abs(ystar[-sam]-ynew))
    MAPEP[i]=mean(abs(ystar[-sam]-ynew)/ystar[-sam])
    
  }
  RMSEP = sqrt(mean(MSEP))
  MAE = mean(MAEP)
  MAPE = mean(MAPEP)
  cat("RMSEP\n")
  cat("===============\n")
  cat(RMSEP,"\n\n")
  cat("MAE\n")
  cat("===============\n")
  cat(MAE,"\n\n")
  cat("MAPE\n")
  cat("===============\n")
  cat(MAPE,"\n\n")
  temp = data.frame(MSEP=MSEP,MAEP=MAEP,MAPEP=MAPEP)
  return(temp)
}







#start with just box-cox transformations, fit just a basic model

concrete.nn = nnet(Strength~., data = Concrete.trans, size = 10, linout = T, skip = T, maxit = 10000, decay = 0.001   )

summary(concrete.nn)


trendscat(Concrete.trans$Strength, fitted(concrete.nn), xlab = "Strength", ylab = "Fitted Values (Strength)")


cor(Concrete.trans$Strength, fitted(concrete.nn))^2

#this is  a skip layer, only 10 layers, 10000 iterations with standard decay. We got 0.9232631

X = model.matrix(Strength~., data = Concrete.trans)[,-1]

Y = Concrete.trans$Strength

#Crossvalidation to choose a model

nnet.sscv.undobc(X,Y,concrete.nn, B = 20, size = 10, maxit = 10000, decay = 0.001, linout = T)



nnet.sscv.undobc(X,Y,concrete.nn, B = 20, size = 8, maxit = 10000, decay = 0.001, linout = T)



nnet.sscv.undobc(X,Y,concrete.nn, B = 20, size = 5, maxit = 10000, decay = 0.001, linout = T)

#increase nodes



concrete.nn2 = nnet(Strength~., data = Concrete.trans, size = 20, linout = T, skip = T, maxit = 10000, decay = 0.001   )

nnet.sscv.undobc(X,Y,concrete.nn2, B = 20, size = 5, maxit = 10000, decay = 0.001, linout = T)


#remove skip layer

concrete.nn3 = nnet(Strength~., data = Concrete.trans, size = 10, linout = T, skip = F, maxit = 10000, decay = 0.001   )

nnet.sscv.undobc(X,Y,concrete.nn3, B = 20, size = 5, maxit = 10000, decay = 0.001, linout = T)



#jack up maxit

concrete.nn4.1 = nnet(Strength~., data = Concrete.trans, size = 10, linout = T, skip = F, maxit = 50000, decay = 0.001   )

nnet.sscv.undobc(X,Y,concrete.nn4.1, B = 20, size = 5, maxit = 10000, decay = 0.001, linout = T)


#more decay

concrete.nn4 = nnet(Strength~., data = Concrete.trans, size = 10, linout = T, skip = T, maxit = 10000, decay = 0.01   )

nnet.sscv.undobc(X,Y,concrete.nn4, B = 20, size = 10, maxit = 10000, decay = 0.005, linout = T)



#a balanced approach

concrete.nn5 = nnet(Strength~., data = Concrete.trans, size = 10, linout = T, skip = T, maxit = 100000, decay = 0.01   )

nnet.sscv.undobc(X,Y,concrete.nn5, B = 30, size = 9, maxit = 100000, decay = 0.01, linout = T)


nnet.sscv.undobc(X,Y,concrete.nn5, B = 30, size = 6, maxit = 100000, decay = 0.01, linout = T)


nnet.sscv.undobc(X,Y,concrete.nn5, B = 30, size = 7, maxit = 100000, decay = 0.02, linout = T)


nnet.sscv.undobc(X,Y,concrete.nn5, B = 30, size = 8, maxit = 100000, decay = 3, linout = T)
