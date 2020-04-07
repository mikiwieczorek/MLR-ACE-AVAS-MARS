
load("~/OneDrive - MNSCU/myGithub/Statistics/Regression_models/Multiple_Linear_Regression/MLR-ACE-AVAS-MARS/Data/mult.Rdata")
load("~/OneDrive - MNSCU/myGithub/Statistics/Regression_models/Multiple_Linear_Regression/MLR-ACE-AVAS-MARS/Data/Regression.Rdata")
setwd("~/OneDrive - MNSCU/myGithub/Statistics/Regression_models/Multiple_Linear_Regression/MLR-ACE-AVAS-MARS/Data")
Concrete = read.csv("Concrete.csv")
setwd("~/OneDrive - MNSCU/myGithub/Statistics/Regression_models/Multiple_Linear_Regression/MLR-ACE-AVAS-MARS/NeuralNets")


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