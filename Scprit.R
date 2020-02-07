load("~/OneDrive - MNSCU/myGithub/Statistics/Regression_models/Multiple_Linear_Regression/MLR-ACE-AVAS-MARS/mult.Rdata")
load("~/OneDrive - MNSCU/myGithub/Statistics/Regression_models/Multiple_Linear_Regression/MLR-ACE-AVAS-MARS/Regression.Rdata")

library(car)
Concrete = read.csv("Concrete.csv")
set.seed(1)
sam = sample(1:1030, size = floor(.6666*1030), replace = F)
Concrete.trans = Concrete
#str(Concrete.trans)
#summary(Concrete.trans)

# BlastFurn, FlyAsh, and Superplast have zeros
lm1 = lm(Strength~., data = Concrete.trans[sam,])
par(mfrow=c(2,2))
plot(lm1)

vif(lm1)
VIF(lm1)
nrow(Concrete.trans[sam,])

#Part B
y = Concrete.trans$Strength[-sam]
ypred = predict(lm1, newdata = Concrete.trans[-sam,])
results = PredAcc(y, ypred)

##Actual vs. predicted
ehat = y-ypred
trendscat(y, ypred, xlab = "Actual Strentgh", ylab = "Predicted Strength")
abline(0,1,lwd=2,col='red')
trendscat(ypred, ehat, xlab = "Predicted Strength of Concrete", ylab = "Residuals")
abline(h=0,lwd=2,col="red")

#Stepwise: mixed model on lm1
lm1.step = step(lm1)
lm1.step$anova
#Removes CourseAgg and Fine Age
# We are deciding to remove the two predictors before conducting any transformations
# Also, the p-values  of those two predictors in the lm1 model were not significant
summary(lm1)
lm2 = update(lm1, Strength~. - CourseAgg - FineAge, data = Concrete.trans[sam,])
lm2.step = step(lm2)
#Stepwise says to keep all predictors in the model now.

#Let's plot
plot(lm2.step)

#Now, we're going to transform and add polynomial terms
#Check for skewness with pairs.plus
pairs.plus(Concrete.trans)
## Do Box-Cox transformations
myBC(Concrete.trans$Strength)
Concrete.trans$Strength = bcPower(Concrete.trans$Strength, 0.6)
Statplot(Concrete.trans$Strength)

myBC(Concrete.trans$BlastFurn+1)
#lamda = 0
Concrete.trans$BlastFurn = log(Concrete.trans$BlastFurn+1)
Statplot(Concrete.trans$BlastFurn)


myBC(Concrete.trans$FlyAsh+1)
#lamda = 0
Concrete.trans$BlastFurn = yjPower(Concrete.trans$FlyAsh, -0.1)
Statplot(Concrete.trans$FlyAsh)

#Log age
myBC(Concrete.trans$Age)
Concrete.trans$Age = log(Concrete.trans$Age)
Statplot(Concrete.trans$Age)

#lambda .2
myBC(Concrete.trans$Cement)
Concrete.trans$Cement = bcPower(Concrete.trans$Cement, 0.2)
Statplot(Concrete.trans$Cement)

#water .8
myBC(Concrete.trans$Water)
Concrete.trans$Water = bcPower(Concrete.trans$Water, 0.8)
Statplot(Concrete.trans$Water)

#labda .3
myBC(Concrete.trans$Superplast+1)
Concrete.trans$Superplast = yjPower(Concrete.trans$Superplast, 0.3)
Statplot(Concrete.trans$Superplast)


#Fit model after transformations
lm.trans = lm(Strength~. -Course , data = Concrete.trans)
lm.trans = update(lm2.step, Strength~. , data = Concrete.trans[sam,])


str(Concrete.trans)
par(mfrow=c(1,1))
plot(lm2)

y = Concrete.trans$Strength[-sam]
ypred = predict(lm.trans, newdata = Concrete.trans[-sam,])
results = PredAcc(y, ypred)

##Actual vs. predicted
ehat = y-ypred
trendscat(y, ypred, xlab = "Actual Strentgh", ylab = "Predicted Strength")
abline(0,1,lwd=2,col='red')
trendscat(ypred, ehat, xlab = "Predicted Strength of Concrete", ylab = "Residuals")
abline(h=0,lwd=2,col="red")

#Residuals vs Leverage: take out 57, 611, 225 cuz of leverage? Do it last
#Do CERES plots

#Let's add poly terms one at the time to our lm.trans model

lm.poly = update(lm.trans, Strength~. + poly(Superplast, 2))
                 
#+ poly(Age,3) + poly(Water, 2) + poly(FlyAsh, 2))

#Remove either Superplast or poly(Superplast, 2)1
lm.poly.2 = update(lm.poly, Strength~. - Superplast)

#ACE
Concrete.ace_avas = Concrete
#Create X matrix of all predictors
X = model.matrix(Strength~., data =Concrete.ace_avas)[sam,-1]
y = Concrete.ace_avas$Strength[sam]
ace = ace(X,y)
maceplot(X,y, ace)

#Water looks like a cubic 
#FlyAsh -> quadratic
#Age --> sqrt(2)
avas = avas(X,y)
maceplot(X,y,avas)

#Back to water quadratic
#FlyAsh, Cement and BlastFurn -> close to linear
#Superplast to quadratic
#Age once again a swart(2)


library(acepack)

