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
cond.rf = randomForest(y~.,data=X,importance=F)
}
#1st: choose optimal mtry's

# results = rf.sscv(cond.rf,X,mtry=3)
# results = rf.sscv(cond.rf,X,mtry=4)
# results = rf.sscv(cond.rf,X,mtry=5)
# results = rf.sscv(cond.rf,X,mtry=6)

#Compare mtry's with errorest function
{#m=4
# m=4
# myforest = function(formula,data){randomForest(formula,data,mtry=m)}
# error.RF = numeric(10)
# for (i in 1:10) error.RF[i] = errorest(y~.,
#                                        data=X,model=myforest)$error
# mean(error.RF)
# summary(error.RF)
# #m=5
# m=5
# myforest = function(formula,data){randomForest(formula,data,mtry=m)}
# error.RF = numeric(10)
# for (i in 1:10) error.RF[i] = errorest(y~.,
#                                        data=X,model=myforest)$error
# mean(error.RF)
# summary(error.RF)
# #m=6
# m=6
# myforest = function(formula,data){randomForest(formula,data,mtry=m)}
# error.RF = numeric(10)
# for (i in 1:10) error.RF[i] = errorest(y~.,
#                                        data=X,model=myforest)$error
# mean(error.RF)
# summary(error.RF)
# 
# #m=7
# m=7
# myforest = function(formula,data){randomForest(formula,data,mtry=m)}
# error.RF = numeric(10)
# for (i in 1:10) error.RF[i] = errorest(y~.,
#                                        data=X,model=myforest)$error
# mean(error.RF)
# summary(error.RF)
}
#2nd: Increasing the number of trees beyonf the default ntree=500
#BEST = based on best mtry from above
#BEST = 5

# results = rf.sscv(cond.rf,X,mtry=BEST,ntree=500,B=250)
# results = rf.sscv(cond.rf,X,mtry=BEST,ntree=750,B=250)
# results = rf.sscv(cond.rf,X,mtry=BEST,ntree=1000,B=250)


####### Create validation set #######
#trainmodel1 is X
#response is y
{n = nrow(X)
sam = sample(1:n,size=floor(n*.6667),replace=F)
X.train = X[sam,] #form training dataset
X.valid = X[-sam,] #form validation dataset
y.train = X.train$formation_energy_ev_natom_log_y_plus_1
y.valid = X.valid$formation_energy_ev_natom_log_y_plus_1

#cond.rf = randomForest(y.train~.,data=X.train)
#plot(cond.rf)
#abline(h=0.000128,col="red",lwd=2)
} 
###### Compare best mtry and ntree combos #######                        
#ntree of 50, 80, 100, 350

# cond.rf = randomForest(formation_energy_ev_natom_log_y_plus_1~.,data=X.train,ntree=50, mtry=9)
# results = rf.sscv(cond.rf,X.train)
# cond.rf = randomForest(formation_energy_ev_natom_log_y_plus_1~.,data=X.train,ntree=80)
# results = rf.sscv(cond.rf,X.train)
# cond.rf = randomForest(formation_energy_ev_natom_log_y_plus_1~.,data=X.train,ntree=100)
# results = rf.sscv(cond.rf,X.train)
# cond.rf = randomForest(formation_energy_ev_natom_log_y_plus_1~.,data=X.train,ntree=350)
# results = rf.sscv(cond.rf,X.train)
# cond.rf = randomForest(formation_energy_ev_natom_log_y_plus_1~.,data=X.train,ntree=150, mtry =12)
# results = rf.sscv(cond.rf,X.train)

#From best to least: ntree = 350, then 80, then 50, then 100

#mtry

# cond.rf = randomForest(formation_energy_ev_natom_log_y_plus_1~.,data=X.train,mtry = 6,ntree=1000)
# results = rf.sscv(cond.rf,X.train)
# cond.final = randomForest(formation_energy_ev_natom_log_y_plus_1~.,data=X.train,mtry=5,ntree=750)
# results = rf.sscv(cond.rf,X.train)

#mtry=5, ntree=350
#mtry=5, ntree=750
#mtry=6, ntree=1000
#mtry=6, ntree=350

#Best combo is: mtry=5 and ntree=350

####### Final randomForest model #######
{cond.final = randomForest(formation_energy_ev_natom_log_y_plus_1~.,data=X.train,mtry=5,ntree=350, importance = T)
ypred = predict(cond.final,newdata=X.valid)
PredAcc(y.valid,ypred)

par(mfrow=c(1,2))
plot(y.train,predict(cond.final),xlab="Actual Formation Energy",ylab="Fitted Formation Energy",
     main= paste("Training Data", " R^2: ", 
                 round((cor(y.train, 
                            predict(cond.final))^2),4)))
abline(0,1,col="red",lwd=2)

plot(y.valid, ypred, xlab="Actual Formation Energy",ylab="Predicted Formation Energy",
     main=paste("Test Data", "R^2: ",
                round((cor(y.valid, 
                           ypred)^2),4)))
abline(0,1,col="red",lwd=2)
par(mfrow=c(1,1))
}
##### Final RMSLEP metric ####
{metrics = PredAcc(y.valid, ypred)
metrics}
########## Variables' importance ############
{par(mfrow=c(3,3))
partialPlot(cond.final,X.train,number_of_total_atoms)
partialPlot(cond.final,X.train,atoms_10)
partialPlot(cond.final,X.train,atoms_20)
partialPlot(cond.final,X.train,atoms_30)
partialPlot(cond.final,X.train,atoms_40)
partialPlot(cond.final,X.train,atoms_60)
partialPlot(cond.final,X.train,atoms_80)
partialPlot(cond.final,X.train,spacegroup)
partialPlot(cond.final,X.train,Spacegroup_12)
}
{
par(mfrow=c(3,3))
partialPlot(cond.final,X.train,Spacegroup_33)
partialPlot(cond.final,X.train,Spacegroup_167)
partialPlot(cond.final,X.train,Spacegroup_194)
partialPlot(cond.final,X.train,Spacegroup_206)
partialPlot(cond.final,X.train,Spacegroup_227)
partialPlot(cond.final,X.train,percent_atom_al)
partialPlot(cond.final,X.train,percent_atom_ga)
partialPlot(cond.final,X.train,percent_atom_in)
partialPlot(cond.final,X.train,lattice_vector_1_ang)
}
{
par(mfrow=c(3,3))
partialPlot(cond.final,X.train,lattice_vector_2_ang)
partialPlot(cond.final,X.train,lattice_vector_3_ang)
partialPlot(cond.final,X.train,lattice_angle_alpha_degree)
partialPlot(cond.final,X.train,lattice_angle_beta_degree)
partialPlot(cond.final,X.train,lattice_angle_gamma_degree)
partialPlot(cond.final,X.train,Aluminum)
partialPlot(cond.final,X.train,Gallium)
partialPlot(cond.final,X.train,Indium)
}

{par(mfrow=c(1,1))
rfimp(cond.final, horiz = F)
}



############ Model 2 #################
{y = train_model2$bandgap_energy_ev_log_y_plus_1
X = train_model2
cond.rf = randomForest(y~.,data=X,importance=T)
}
#1st: choose optimal mtry's

# results = rf.sscv(cond.rf,X,mtry=3)
# results = rf.sscv(cond.rf,X,mtry=4)
# results = rf.sscv(cond.rf,X,mtry=5)
# results = rf.sscv(cond.rf,X,mtry=6)

{#m=4
# m=4
# myforest = function(formula,data){randomForest(formula,data,mtry=m)}
# error.RF = numeric(10)
# for (i in 1:10) error.RF[i] = errorest(y~.,
#                                        data=X,model=myforest)$error
# mean(error.RF)
# summary(error.RF)
# #m=5
# m=5
# myforest = function(formula,data){randomForest(formula,data,mtry=m)}
# error.RF = numeric(10)
# for (i in 1:10) error.RF[i] = errorest(y~.,
#                                        data=X,model=myforest)$error
# mean(error.RF)
# summary(error.RF)
# #m=6
# m=6
# myforest = function(formula,data){randomForest(formula,data,mtry=m)}
# error.RF = numeric(10)
# for (i in 1:10) error.RF[i] = errorest(y~.,
#                                        data=X,model=myforest)$error
# mean(error.RF)
# summary(error.RF)
# 
# #m=7
# m=7
# myforest = function(formula,data){randomForest(formula,data,mtry=m)}
# error.RF = numeric(10)
# for (i in 1:10) error.RF[i] = errorest(y~.,
#                                        data=X,model=myforest)$error
# mean(error.RF)
# summary(error.RF)
}
#2nd: Increasing the number of trees beyonf the default ntree=500
#BEST = based on best mtry from above
#BEST = 4

# results = rf.sscv(cond.rf,X,mtry=BEST,ntree=500,B=250)
# results = rf.sscv(cond.rf,X,mtry=BEST,ntree=750,B=250)
# results = rf.sscv(cond.rf,X,mtry=BEST,ntree=1000,B=250)

#mtry=4,ntree=1000

####### Create validation set #######
#trainmodel2 is X
#response is y
{n = nrow(X)
sam = sample(1:n,size=floor(n*.6667),replace=F)
X.train = X[sam,] #form training dataset
X.valid = X[-sam,] #form validation dataset
y.train = X.train$bandgap_energy_ev_log_y_plus_1
y.valid = X.valid$bandgap_energy_ev_log_y_plus_1

# par(mfrow=c(1,1))
# cond.rf = randomForest(y.train~.,data=X.train)
# plot(cond.rf)
# abline(h=0.000128,col="red",lwd=2)
}
###### Compare best mtry and ntree combos #######                        
#ntree of 50, 80, 100, 350

# cond.rf = randomForest(bandgap_energy_ev_log_y_plus_1~.,data=X.train,ntree=40)
# results = rf.sscv(cond.rf,X.train)
# cond.rf = randomForest(bandgap_energy_ev_log_y_plus_1~.,data=X.train,ntree=80)
# results = rf.sscv(cond.rf,X.train)
# cond.rf = randomForest(bandgap_energy_ev_log_y_plus_1~.,data=X.train,ntree=750)
# results = rf.sscv(cond.rf,X.train)
# cond.rf = randomForest(bandgap_energy_ev_log_y_plus_1~.,data=X.train,ntree=350)
# results = rf.sscv(cond.rf,X.train)
# cond.rf = randomForest(bandgap_energy_ev_log_y_plus_1~.,data=X.train,ntree=150, mtry = 6)
# results = rf.sscv(cond.rf,X.train)

# #mtry

# cond.rf = randomForest(bandgap_energy_ev_log_y_plus_1~.,data=X.train,mtry = 4,ntree=1000)
# results = rf.sscv(cond.rf,X.train)
# cond.final = randomForest(bandgap_energy_ev_log_y_plus_1~.,data=X.train,mtry=5,ntree=750)
# results = rf.sscv(cond.rf,X.train)

#mtry=4, ntree=1000
#mtry=5, ntree=750

#Best combo is: mtry=4 and ntree=1000


####### Final randomForest model #######
{cond.final = randomForest(bandgap_energy_ev_log_y_plus_1~.,data=X.train,mtry=4,ntree=1000, importance = T)
ypred = predict(cond.final,newdata=X.valid)
PredAcc(y.valid, ypred)

par(mfrow=c(1,2))
plot(y.train,predict(cond.final),xlab="Actual Formation Energy",ylab="Fitted Formation Energy",
     main= paste("Training Data", " R^2: ", 
                 round((cor(y.train, 
                            predict(cond.final))^2),4)))
abline(0,1,col="red",lwd=2)

plot(y.valid, ypred, xlab="Actual Formation Energy",ylab="Predicted Formation Energy",
     main=paste("Test Data", "R^2: ",
                round((cor(y.valid, 
                           ypred)^2),4)))
abline(0,1,col="red",lwd=2)
par(mfrow=c(1,1))
}
### Final RMSLEP metric ####
{metrics = PredAcc(y.valid, ypred)
metrics
}

########## Variables' importance ############
{par(mfrow=c(3,3))
partialPlot(cond.final,X.train,number_of_total_atoms)
partialPlot(cond.final,X.train,atoms_10)
partialPlot(cond.final,X.train,atoms_20)
partialPlot(cond.final,X.train,atoms_30)
partialPlot(cond.final,X.train,atoms_40)
partialPlot(cond.final,X.train,atoms_60)
partialPlot(cond.final,X.train,atoms_80)
partialPlot(cond.final,X.train,spacegroup)
partialPlot(cond.final,X.train,Spacegroup_12)
}
{
par(mfrow=c(3,3))
partialPlot(cond.final,X.train,Spacegroup_33)
partialPlot(cond.final,X.train,Spacegroup_167)
partialPlot(cond.final,X.train,Spacegroup_194)
partialPlot(cond.final,X.train,Spacegroup_206)
partialPlot(cond.final,X.train,Spacegroup_227)
partialPlot(cond.final,X.train,percent_atom_al)
partialPlot(cond.final,X.train,percent_atom_ga)
partialPlot(cond.final,X.train,percent_atom_in)
partialPlot(cond.final,X.train,lattice_vector_1_ang)
}
{
par(mfrow=c(3,3))
partialPlot(cond.final,X.train,lattice_vector_2_ang)
partialPlot(cond.final,X.train,lattice_vector_3_ang)
partialPlot(cond.final,X.train,lattice_angle_alpha_degree)
partialPlot(cond.final,X.train,lattice_angle_beta_degree)
partialPlot(cond.final,X.train,lattice_angle_gamma_degree)
partialPlot(cond.final,X.train,Aluminum)
partialPlot(cond.final,X.train,Gallium)
partialPlot(cond.final,X.train,Indium)
}

{par(mfrow=c(1,1))
  rfimp(cond.final, horiz = F)
}




