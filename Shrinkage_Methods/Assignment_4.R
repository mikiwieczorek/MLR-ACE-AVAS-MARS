
library(tidyr)
library(dplyr)
library(ggplot2)
require(nnet)
require(Ecfun)
require(car)
require(ISLR)
require(MASS)
require(glmnet)


#install.packages("tidyr")
#install.packages("dplyr")
#install.packages("ggplot2")
#install.packages("nnet")
#install.packages("Ecfun")
#install.packages("car")
#install.packages("ISLR")
#install.packages("MASS")



College2 = data.frame(PctAccept=100*(College$Accept/College$Apps),College[,-c(2:3)])


attach(College)

College4 = data.frame(logApps=log(Apps),Private,logAcc=log(Accept),logEnr=log(Enroll),Top10perc,
                      Top25perc,logFull=log(F.Undergrad),logPart=log(P.Undergrad),Outstate,Room.Board,Books,Personal,PhD,Terminal,S.F.Ratio,perc.alumni,logExp=log(Expend),Grad.Rate)

detach(College)

#1


#a

#we use this vector as a means to seperate our training and test set

set.seed(1)
sam = sample(1:length(y), size = floor(.6666*1030), replace = F)


#b


X = model.matrix(PctAccept~.,data=College2)[,-1]
y = College2$PctAccept
Xs = scale(X)
College2.temp = data.frame(y,Xs)
PA.ols = lm(y~Xs,data=College2.temp,subset=sam)
ypred = predict(PA.ols,newdata=College2.temp[-sam,])
RMSEP.ols = sqrt(mean((y[-sam]-ypred)^2))
RMSEP.ols


#> RMSEP.ols
#[1] 16.82557



#c

#training model versions



grid = 10^seq(10,-2,length=200)
ridge.mod = glmnet(Xs[sam,],y[sam],alpha=0,lambda=grid)

lasso.mod = glmnet(Xs[sam,],y[sam],alpha=1,lambda=grid)


plot(ridge.mod) 

plot(ridge.mod,xvar="lambda") 




#Both using lambda and the L1 norm tell the same story in different scales. Overall, they look fairly stadnard and we can definetly
#see that certain varaibles are more important that others by a noatable margain. From the plot with x = lambda we also can note that
#likley would never want a lambda value greater than 5, as subesquent increases won't have any notable impact. It will be interesting 
#to see how this stacks up against the LASSO versions


plot(lasso.mod) 

plot(lasso.mod,xvar="lambda") 


#As expected, Lasso is much more contrasting and rigid with its regularization, capping out with a lambda closer to 4 and showing
#certain varaibles to be leeps and bounds ahead of others. Furthermore, for both ridge and LASSO, while the overall lambda values shown
#may be much smaller then many of the examples we looked at, given that the college dataset only has 18 to start with, narrowing it down
#to even sub 5 is an admirable result. However, these plots alone are not a preffered way to choose lambda, as cross-validation 
#tends to be much more prescise in that regard.


#d

cv.out_ridge = cv.glmnet(Xs[sam,],y[sam],alpha=0)
plot(cv.out_ridge)
bestlam_ridge = cv.out_ridge$lambda.min
bestlam_ridge

# 0.7006997


cv.out_lasso = cv.glmnet(Xs[sam,],y[sam],alpha=1)
plot(cv.out_lasso)
bestlam_lasso = cv.out_lasso$lambda.min
bestlam_lasso

# 0.1065001


#e


lasso.mod = glmnet(Xs[sam,], y[sam], alpha = 1, lambda = bestlam_lasso)
lasso.pred = predict(lasso.mod, newx = Xs[-sam,])

coef(lasso.mod)

# 17 x 1 sparse Matrix of class "dgCMatrix"
# s0
# (Intercept) 74.78642725
# PrivateYes   2.97828704
# Enroll       1.88664016
# Top10perc   -5.12584834
# Top25perc    .         
# F.Undergrad  .         
# P.Undergrad -1.46944963
# Outstate     1.52456756
# Room.Board  -2.84813667
# Books       -1.09240424
# Personal     0.08458621
# PhD          .         
# Terminal     0.18881525
# S.F.Ratio   -1.50029175
# perc.alumni  0.39489197
# Expend      -3.48052571
# Grad.Rate   -1.89225948


ridge.mod = glmnet(Xs[sam,], y[sam], alpha = 0, lambda = bestlam_lasso)
ridge.pred = predict(ridge.mod, newx = Xs[-sam,])

coef(ridge.mod)

# 17 x 1 sparse Matrix of class "dgCMatrix"
# s0
# (Intercept) 74.80034676
# PrivateYes   3.05989001
# Enroll       3.84015012
# Top10perc   -5.23619146
# Top25perc   -0.09616916
# F.Undergrad -1.79839190
# P.Undergrad -1.55654831
# Outstate     1.88266710
# Room.Board  -3.01241402
# Books       -1.17509769
# Personal     0.27983585
# PhD          0.13618524
# Terminal     0.36713206
# S.F.Ratio   -1.67566162
# perc.alumni  0.52068348
# Expend      -3.84823795
# Grad.Rate   -2.19810475


#compare to OLS


coef(PA.ols)

# (Intercept)  XsPrivateYes      XsEnroll   XsTop10perc   XsTop25perc XsF.Undergrad XsP.Undergrad    XsOutstate  XsRoom.Board 
# 74.8034528     3.0824234     4.5978281    -5.4983918     0.1320567    -2.5290500    -1.5431593     1.9605473    -3.0547464 
# XsBooks    XsPersonal         XsPhD    XsTerminal   XsS.F.Ratio Xsperc.alumni      XsExpend   XsGrad.Rate 
# -1.1763839     0.2956915     0.1537753     0.3688423    -1.7119667     0.5165208    -3.8963058    -2.2630644 






