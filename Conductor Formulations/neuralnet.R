###### Libraries #####
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
  library(nnet)
  library(GGally)
}

###### Functions #####
#detach("package:Ecfun", unload = TRUE)
#rf.sscv Monte-Carlo CV function
{rf.sscv = function(fit,data,p=.667,B=100,mtry=fit$mtry,ntree=fit$ntree) {
  RMSEP = rep(0,B)
  RMSLEP = rep(0,B)
  MAEP = rep(0,B)
  MAPEP = rep(0,B)
  y = fit$y
  n = nrow(data)
  ss <- floor(n*p)
  for (i in 1:B) {
    sam = sample(1:n,ss,replace=F)
    fit2 = randomForest(formula(fit),data=data[sam,],mtry=mtry,ntree=ntree)
    ynew = predict(fit2,newdata=data[-sam,])
    RMSEP[i] = sqrt(mean((y[-sam]-ynew)^2))
    RMSLEP[i] = sqrt(mean((log(ynew +1) - log(y[-sam] +1))^2))
    MAEP[i] = mean(abs(y[-sam]-ynew))
    MAPEP[i] = mean((abs(y[-sam]-ynew)/y[-sam]))
    
  }
  RMSEP = mean(RMSEP)
  RMSLEP = mean(RMSLEP)
  MAEP = mean(MAEP)
  MAPEP = mean(MAPEP)
  cat("RMSEP\n")
  cat("===============\n")
  cat(RMSEP,"\n\n")
  cat("RMSLEP\n")
  cat("===============\n")
  cat(RMSLEP,"\n\n")
  cat("MAEP\n")
  cat("===============\n")
  cat(MAEP,"\n\n")
  cat("MAPEP\n")
  cat("===============\n")
  cat(MAPEP,"\n\n")
  temp = data.frame(RMSEP=RMSEP, RMSLEP=RMSLEP, MAEP=MAEP,MAPEP=MAPEP)
  return(temp)
}

}
#Predict Accuracy function
{PredAcc = function(y, ypred){
  RMSEP = sqrt(mean((y-ypred)^2))
  RMSLEP = sqrt(mean((log(ypred +1) - log(y+1))^2))
  MAE = mean(abs(y-ypred))
  MAPE = mean(abs(y-ypred)/y)*100
  cat("RMSEP\n")
  cat("================\n")
  cat(RMSEP, "\n\n")
  cat("RMSLEP\n")
  cat("================\n")
  cat(RMSLEP, "\n\n")
  cat("MAE\n")
  cat("================\n")
  cat(MAE, "\n\n")
  cat("MAPE\n")
  cat("================\n")
  cat(MAPE, "\n\n")
  return(data.frame(RMSEP = RMSEP, RMSLEP = RMSLEP, MAE = MAE, MAPE = MAPE))
}
}
#Plotting variable importance
{rfimp = function(rffit, horiz=T) {barplot(sort(rffit$importance[,1]),horiz=horiz,
                                           xlab="Mean Decreasing in Accuracy",main="Variable Importance")
}
}

###### Load and prepare data ######
#Load data:
{setwd("~/OneDrive - MNSCU/myGithub/Supervised_Learning/Multiple_Linear_Regression/MachineLearning-SupervisedLearning/Conductor Formulations")
  #Train set
  cond_train = read.csv("Conductors (train).csv")
  #Test set
  cond_test = read.csv("Conductors (test).csv")
  
  ### New training sets
  cond_train_new_vars = read.csv("cond_train_new_variables.csv")
  #No_box_cox
  #angles measures are still in degrees; no transformaitons on predictors
  cond_train_new_vars_no_box_cox = read.csv("cond_train_new_variables_no_box_cox.csv")
  
  ### New Variables
  
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

#Rearrange the data for two models
#Model 1: formation energy as response
#Model 2: bandgap energy as response
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


###### Model 1 #####
{y = train_model1$formation_energy_ev_natom_log_y_plus_1
X = train_model1
}

#tcX = train_model1_no_box_cox[,c(1,2,25,26,27)]
#ggpairs(tcX)
#require(caret)
set.seed(1111)
mod1 = nnet(formation_energy_ev_natom_log_y_plus_1~., data = X, size=6, skip=T, linout=T, decay=0.0001, maxit=25000)
summary(mod1)
plot(X$formation_energy_ev_natom_log_y_plus_1, fitted(mod1))
abline(0,1, lwd=2, col="blue")

###### Create train, dev, and test sets #######

{n = nrow(X)
set.seed(1111)
sam = sample(1:n,size=floor(n*.6667),replace=F)
X.train = X[sam,] #form training dataset
X.valid = X[-sam,] #form validation dataset
y.train = X.train$formation_energy_ev_natom_log_y_plus_1
y.valid = X.valid$formation_energy_ev_natom_log_y_plus_1

#cond.rf = randomForest(y.train~.,data=X.train)
#plot(cond.rf)
#abline(h=0.000128,col="red",lwd=2)
} 

nn1 = nnet(formation_energy_ev_natom_log_y_plus_1~., data = X.train, size=6, skip=F, linout=T, decay=0.01, maxit=25000)
nn2 = nnet(formation_energy_ev_natom_log_y_plus_1~., data = X.train, size=8, skip=F, linout=T, decay=0.005, maxit=25000)
nn3 = nnet(formation_energy_ev_natom_log_y_plus_1~., data = X.train, size=8, skip=T, linout=T, decay=0.05, maxit=25000)

nn = nn2
#best nn2 so far
{ypred = predict(nn, newdata = X.valid)
PredAcc(y.valid, ypred)

par(mfrow=c(1,2))
plot(y.train,predict(nn1),xlab="Actual Formation Energy",ylab="Fitted Formation Energy",
     main= paste("Training Data", " R^2: ", 
                 round((cor(y.train, 
                            predict(nn1))^2),4)))
abline(0,1,col="red",lwd=2)

plot(y.valid, ypred, xlab="Actual Formation Energy",ylab="Predicted Formation Energy",
     main=paste("Test Data", "R^2: ",
                round((cor(y.valid, 
                           ypred)^2),4)))
abline(0,1,col="red",lwd=2)
par(mfrow=c(1,1))
}

###### Model 2 #####
{y = train_model2$bandgap_energy_ev_log_y_plus_1
X = train_model2
}

#tcX = train_model1_no_box_cox[,c(1,2,25,26,27)]
#ggpairs(tcX)
#require(caret)
set.seed(1111)
mod1 = nnet(bandgap_energy_ev_log_y_plus_1~., data = X, size=6, skip=T, linout=T, decay=0.0001, maxit=25000)
summary(mod1)
plot(X$bandgap_energy_ev_log_y_plus_1, fitted(mod1))
abline(0,1, lwd=2, col="blue")

###### Create train, dev, and test sets #######

{n = nrow(X)
set.seed(1111)
sam = sample(1:n,size=floor(n*.6667),replace=F)
X.train = X[sam,] #form training dataset
X.valid = X[-sam,] #form validation dataset
y.train = X.train$bandgap_energy_ev_log_y_plus_1
y.valid = X.valid$bandgap_energy_ev_log_y_plus_1

#cond.rf = randomForest(y.train~.,data=X.train)
#plot(cond.rf)
#abline(h=0.000128,col="red",lwd=2)
} 

nn1 = nnet(bandgap_energy_ev_log_y_plus_1~., data = X.train, size=6, skip=F, linout=T, decay=0.01, maxit=25000)
nn2 = nnet(bandgap_energy_ev_log_y_plus_1~., data = X.train, size=8, skip=F, linout=T, decay=0.005, maxit=25000)
nn3 = nnet(bandgap_energy_ev_log_y_plus_1~., data = X.train, size=10, skip=T, linout=T, decay=0.05, maxit=25000)

nn = nn3
#best nn3 so far
{ypred = predict(nn, newdata = X.valid)
  PredAcc(y.valid, ypred)
  
  par(mfrow=c(1,2))
  plot(y.train,predict(nn1),xlab="Actual Formation Energy",ylab="Fitted Formation Energy",
       main= paste("Training Data", " R^2: ", 
                   round((cor(y.train, 
                              predict(nn1))^2),4)))
  abline(0,1,col="red",lwd=2)
  
  plot(y.valid, ypred, xlab="Actual Formation Energy",ylab="Predicted Formation Energy",
       main=paste("Test Data", "R^2: ",
                  round((cor(y.valid, 
                             ypred)^2),4)))
  abline(0,1,col="red",lwd=2)
  par(mfrow=c(1,1))
}


####### KERAS ########
require(keras)
{y = train_model1$formation_energy_ev_natom_log_y_plus_1
train_data = as.matrix(scale(train_model1))
}

metric_RMSLEP <- function(y, ) {
  metric_mean_squared_logarithmic_error(y,y_pred)
}



build_model <- function() {
  
  model <- keras_model_sequential() %>%
    layer_dense(units = 64, activation = "relu",
                input_shape = dim(train_data)[2]) %>%
    layer_dropout(0.6) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dropout(0.6) %>%
    layer_dense(units = 1, activation = "linear")
  
  model %>% compile(
    loss = "mse",
    optimizer = optimizer_rmsprop(),
    metrics = list("mean_squared_logarithmic_error")
  )
  
 return(model)
}

model <- build_model()
model %>% summary()


# Display training progress by printing a single dot for each completed epoch.
print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs) {
    if (epoch %% 80 == 0) cat("\n")
    cat(".")
  }
)    

epochs <- 50

# Fit the model and store training stats
history <- model %>% fit(
  train_data,
  y,
  epochs = epochs,
  validation_split = 0.2,
  verbose = 0,
  callbacks = list(print_dot_callback)
)

#library(ggplot2)

plot(history, metrics = "mean_squared_logarithmic_error", smooth = FALSE) +
  coord_cartesian(ylim = c(0, 5))

history$metrics
# The patience parameter is the amount of epochs to check for improvement.
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)

model <- build_model()
history <- model %>% fit(
  train_data,
  y,
  epochs = epochs,
  validation_split = 0.2,
  verbose = 0,
  callbacks = list(early_stop, print_dot_callback)
)

plot(history, metrics = "mean_absolute_error", smooth = FALSE) +
  coord_cartesian(xlim = c(0, 50), ylim = c(0, 5))

