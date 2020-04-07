######### Libraries ######
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
  library(ipred)
  library(randomForest)
}
###### Monte Carlo Cross-Validation of Elastic Net, Ridge and Lasso Regression ####
{glmnet.ssmc = function(X,y,p=.667,M=100,alpha=1,lambda=1) {
  RMSEP = rep(0,M)
  RMSLEP = rep(0,M)
  MAEP = rep(0,M)
  MAPEP = rep(0,M)
  n = nrow(X)
  for (i in 1:M) {
    ss = floor(n*p)
    sam = sample(1:n,ss,replace=F)
    fit = glmnet(X[sam,],y[sam],lambda=lambda,alpha=alpha)
    ypred = predict(fit,newx=X[-sam,])
    RMSEP[i] = sqrt(mean((y[-sam]-ypred)^2))
    RMSLEP[i] = sqrt(mean((log(ypred +1) - log(y[-sam] +1))^2)) 
    MAEP[i] = mean(abs(y[-sam]-ypred))
    yp = ypred[y[-sam]!=0]
    ya = y[-sam][y[-sam]!=0]
    MAPEP[i]=mean(abs(yp-ya)/ya)
  }
  cat("RMSEP=",mean(RMSEP),
      "RMSLEP=",mean(RMSLEP),
      "MAEP=",mean(MAEP),
      "MAPEP=",mean(MAPEP))
  cv = return(data.frame(RMSEP=RMSEP, RMSLEP = RMSLEP, MAEP=MAEP,MAPEP=MAPEP)) 
}
}
###### Load and prepare data ######
#Load data:
{#setwd("~/OneDrive - MNSCU/myGithub/Supervised_Learning/Multiple_Linear_Regression/MachineLearning-SupervisedLearning/Conductor Formulations")
  #Train set
  cond_train = read.csv("Conductors (train).csv")
  #Test set
  cond_test = read.csv("Conductors (test).csv")
  
  ### New training sets
  cond_train_new_vars = read.csv("cond_train_new_variables.csv")
  #No_box_cox
  #angles measures are still in degrees; no transformaitons on predictors
  cond_train_new_vars_no_box_cox = read.csv("cond_train_new_variables_no_box_cox.csv")
  
  ####### New Variables #######
  
  #Resposne vectors 
  y_train_form_eng_trans = cond_train_new_vars$formation_energy_ev_natom_log_y_plus_1
  y_train_bandgap_eng_trans = cond_train_new_vars$bandgap_energy_ev_log_y_plus_1
  
  #Resposne vectors for no box_cox
  #y_train_form_eng_trans = cond_train_new_vars_no_box_cox$formation_energy_ev_natom_log_y_plus_1
  #y_train_bandgap_eng_trans = cond_train_new_vars_no_box_cox$bandgap_energy_ev_log_y_plus_1
  
  #Df of all x's
  x_train = cond_train_new_vars[,-c(1,25,26)]
  #x_train =cond_train_new_vars_no_box_cox[,-c(1,2,26,27)]
}
#Rearrange the data for two models: Model 1: formation energy as response; Model 2: bandgap energy as response
{#detach("package:Ecfun", unload = TRUE)
  cond_train_new_vars %>%
    select(25,2:24,27:29) -> train_model1
  
  cond_train_new_vars%>%
    select(26, 2:24,27:29) -> train_model2
} 
#Rearrange the data for two models for no box_cox
{
  #detach("package:Ecfun", unload = TRUE)
  cond_train_new_vars_no_box_cox %>%
    select(26,3:25,28:30) -> train_model1_no_box_cox
  
  cond_train_new_vars_no_box_cox%>%
    select(27, 3:25,28:30) -> train_model2_no_box_cox
} 


######## Model 1:response is formation energy ########
####### Set up #########
{
  X = scale(model.matrix(formation_energy_ev_natom_log_y_plus_1~., data = train_model1)[,-1])
  y = train_model1$formation_energy_ev_natom_log_y_plus_1
}
###### For Elastic Net: finding lambda and fitting the model ########
#Find best lambda
{ 
  par(mfrow=c(4,4))
  #Change alpha
  alpha = 0.8
  cv.en = cv.glmnet(X,y,alpha=alpha)
  bestlam.en = cv.en$lambda.min
  #plot(cv.en)
  #title(main = paste("Best log(lambda) for Elastic Net", (round(log(bestlam.en),digits = 2))), sub = paste("Best lambda:", round(bestlam.en,5)))
  #en.results = glmnet.ssmc(X,y,p=.75, M=1000,alpha=0.1,lambda=bestlam.en)
}
#Fit with optimal lambda
{
  en.mod = glmnet(X,y,alpha=alpha, lambda =bestlam.en)
  #y and yhat correlation
  en.cor = cor(y, predict(en.mod, newx = X))
  #y and yhat correlation^2 = R^2
  en.rsqaured = cor(y, predict(en.mod, newx = X))^2
  plot(y,predict(en.mod,newx=X),xlab="Actual Age (y-values)",ylab="Predicted Age (yhat-values)", main = paste("Elastic Net Model:", "   ", "Correlation:", round(en.cor,4), "   ", "R^2:", round(en.rsqaured,4)),
       sub = paste("alpha:", " ", alpha))
}

###### Elastic net #########
{alpha1 = 0.95
cv.en = cv.glmnet(X,y,alpha=alpha1)
bestlam.en = cv.en$lambda.min

alpha2 = 0.5
cv.en = cv.glmnet(X,y,alpha=alpha2)
bestlam.en = cv.en$lambda.min

alpha3 = 0.05
cv.en = cv.glmnet(X,y,alpha=alpha3)
bestlam.en = cv.en$lambda.min
#plot(cv.en)
#title(main = paste("Best log(lambda) for Elastic Net", (round(log(bestlam.en),digits = 2))), sub = paste("Best lambda:", round(bestlam.en,5)))
en.results = glmnet.ssmc(X,y,p=.75, M=1000,alpha=alpha1,lambda=bestlam.en)
en.results2 = glmnet.ssmc(X,y,p=.75, M=1000,alpha=alpha2,lambda=bestlam.en)
en.results3 = glmnet.ssmc(X,y,p=.75, M=1000,alpha=alpha3,lambda=bestlam.en)
}
###### Ridge regression##########
{cv.ridge= cv.glmnet(X, y, alpha = 0)
bestlam.ridge = cv.ridge$lambda.min
#plot(cv.ridge)
#title(main = paste("Best log(lambda) for Ridge", (round(log(bestlam.ridge),digits = 2))), sub = paste("Best lambda:", round(bestlam.ridge)))
ridge.results = glmnet.ssmc(X,y, p=.75, M=1000,alpha=0,lambda=bestlam.ridge)
}
###### Lasso regression##############
{cv.lasso = cv.glmnet(X, y, alpha = 1)
bestlam.lasso = cv.lasso$lambda.min
#plot(cv.lasso)
#title(main = paste("Best log(lambda) for Lasso", (round(log(bestlam.lasso),digits = 4))), sub = paste("Best lambda:", round(bestlam.lasso, 4)))
lasso.results = glmnet.ssmc(X,y,p=.75, M=1000,alpha=1,lambda=bestlam.lasso)
}
###### Compare the three methods ###########
{names = c("Ridge", "Lasso", "Elastic Net, alpha = 0.95", "Elastic Net, alpha = 0.5", "Elastic Net, alpha = 0.0.005")
metrics =cbind(((names)), rbind(
  do.call(cbind, lapply(ridge.results,mean)), 
  do.call(cbind, lapply(lasso.results, mean)),
  do.call(cbind, lapply(en.results, mean)),
  do.call(cbind, lapply(en.results2, mean)),
  do.call(cbind, lapply(en.results3, mean))))

df.metrics = as.data.frame(metrics)
write.csv(df.metrics, file = "Model_1_shrinkage_regression_results.csv", row.names = FALSE)
}




###### Model 2:response is bandgap energy ########
###### Set up #########
{
X = scale(model.matrix(bandgap_energy_ev_log_y_plus_1~., data = train_model2)[,-1])
y = train_model2$bandgap_energy_ev_log_y_plus_1
}
###### Elastic net #########
{alpha1 = 0.8
cv.en = cv.glmnet(X,y,alpha=alpha1)
bestlam.en = cv.en$lambda.min

alpha2 = 0.5
cv.en = cv.glmnet(X,y,alpha=alpha2)
bestlam.en = cv.en$lambda.min

alpha3 = 0.4
cv.en = cv.glmnet(X,y,alpha=alpha3)
bestlam.en = cv.en$lambda.min
#plot(cv.en)
#title(main = paste("Best log(lambda) for Elastic Net", (round(log(bestlam.en),digits = 2))), sub = paste("Best lambda:", round(bestlam.en,5)))
en.results = glmnet.ssmc(X,y,p=.75, M=1000,alpha=alpha1,lambda=bestlam.en)
en.results2 = glmnet.ssmc(X,y,p=.75, M=1000,alpha=alpha2,lambda=bestlam.en)
en.results3 = glmnet.ssmc(X,y,p=.75, M=1000,alpha=alpha3,lambda=bestlam.en)
}
###### Ridge regression##########
{cv.ridge= cv.glmnet(X, y, alpha = 0)
bestlam.ridge = cv.ridge$lambda.min
#plot(cv.ridge)
#title(main = paste("Best log(lambda) for Ridge", (round(log(bestlam.ridge),digits = 2))), sub = paste("Best lambda:", round(bestlam.ridge)))
ridge.results = glmnet.ssmc(X,y, p=.75, M=1000,alpha=0,lambda=bestlam.ridge)
}
###### Lasso regression##############
{cv.lasso = cv.glmnet(X, y, alpha = 1)
bestlam.lasso = cv.lasso$lambda.min
#plot(cv.lasso)
#title(main = paste("Best log(lambda) for Lasso", (round(log(bestlam.lasso),digits = 4))), sub = paste("Best lambda:", round(bestlam.lasso, 4)))
lasso.results = glmnet.ssmc(X,y,p=.75, M=1000,alpha=1,lambda=bestlam.lasso)
}
###### Compare the three methods ###########
{names = c("Ridge", "Lasso", "Elastic Net, alpha = 0.8", "Elastic Net, alpha = 0.5", "Elastic Net, alpha = 0.4")
metrics =cbind(((names)), rbind(
  do.call(cbind, lapply(ridge.results,mean)), 
  do.call(cbind, lapply(lasso.results, mean)),
  do.call(cbind, lapply(en.results, mean)),
  do.call(cbind, lapply(en.results2, mean)),
  do.call(cbind, lapply(en.results3, mean))))

df.metrics = as.data.frame(metrics)
write.csv(df.metrics, file = "Model_2_shrinkage_regression_results.csv", row.names = FALSE)
}

###### For Elastic Net: finding lambda and fitting the model ########
#Find best lambda
{ 
  par(mfrow=c(4,4))
  #Change alpha
  alpha = 0.8
  cv.en = cv.glmnet(X,y,alpha=alpha)
  bestlam.en = cv.en$lambda.min
  #plot(cv.en)
  #title(main = paste("Best log(lambda) for Elastic Net", (round(log(bestlam.en),digits = 2))), sub = paste("Best lambda:", round(bestlam.en,5)))
  #en.results = glmnet.ssmc(X,y,p=.75, M=1000,alpha=0.1,lambda=bestlam.en)
}
#Fit with optimal lambda
{
  en.mod = glmnet(X,y,alpha=alpha, lambda =bestlam.en)
  #y and yhat correlation
  en.cor = cor(y, predict(en.mod, newx = X))
  #y and yhat correlation^2 = R^2
  en.rsqaured = cor(y, predict(en.mod, newx = X))^2
  plot(y,predict(en.mod,newx=X),xlab="Actual Age (y-values)",ylab="Predicted Age (yhat-values)", main = paste("Elastic Net Model:", "   ", "Correlation:", round(en.cor,4), "   ", "R^2:", round(en.rsqaured,4)),
       sub = paste("alpha:", " ", alpha))
}

