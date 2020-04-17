CLASSIFYING MUSIC GENRE BASED ON AN AUDIO SAMPLE
================================================

This dataset consists of 191 continuous variables measured from an audio
sample from a piece of music. All audio samples were of the same length
timewise. Your goal is to use these data to classify the genre of the
piece of music into one of the following genres: Blues, Classical, Jazz,
Metal, Pop, and Rock. These data are contained in two files:
GenreTrain.csv (n = 10,000 samples) and GenreTest.csv (m = 2,495
samples).

Reading in the data and scaling it.

``` r
gtrain = read.csv("GenreTrain.csv")
gtest = read.csv("GenreTest.csv")

gtrain.X = scale(gtrain[,-192])

gtrain = data.frame(GENRE=gtrain$GENRE,gtrain.X)


gtest.X = scale(gtest[,-192])

gtest = data.frame(GENRE=gtest$GENRE,gtest.X)
```

Given that we have 10,000 0bservations for our training set, for model
developement purposes we should be safe to split it further into a
validation set and training set.

``` r
testcases2 = sample(1:dim(gtrain)[1],3333,replace=F)
gval = gtrain[testcases2,]
gtrain = gtrain[-testcases2,]
```

Starting out simple, we will use a procedure similar to the problem
above to build an optimal k-NN model.

Nearest Neighbor Classification
===============================

Simple kn = 1 model:

``` r
# g.sknn = sknn(GENRE~.,data=gtrain,kn=1)
# g.yhat = predict(g.sknn,newdata=gval)
# misclass(g.yhat$class,gval$GENRE)
```

Misclassification Rate = 0.0852

It’s a strong start at only 8.5% missclassification, but with
modification we should be able to bring that down. Like before, we will
start by finding the optimal number of neighbors.

``` r
# g.sknn = sknn(GENRE~.,data=gtrain,kn=2)
# g.yhat = predict(g.sknn,newdata=gval)
# 
# misclass(g.yhat$class,gval$GENRE)
```

Misclassification Rate = 0.106

``` r
# g.sknn = sknn(GENRE~.,data=gtrain,kn=3)
# g.yhat = predict(g.sknn,newdata=gval)
# 
# misclass(g.yhat$class,gval$GENRE)
```

Misclassification Rate = 0.0963

``` r
# g.sknn = sknn(GENRE~.,data=gtrain,kn=4)
# g.yhat = predict(g.sknn,newdata=gval)
# 
# misclass(g.yhat$class,gval$GENRE)
```

Misclassification Rate = 0.11

``` r
# g.sknn = sknn(GENRE~.,data=gtrain,kn=5)
# g.yhat = predict(g.sknn,newdata=gval)
# 
# misclass(g.yhat$class,gval$GENRE)
```

Misclassification Rate = 0.104

``` r
# g.sknn = sknn(GENRE~.,data=gtrain,kn=7)
# g.yhat = predict(g.sknn,newdata=gval)
# 
# misclass(g.yhat$class,gval$GENRE)
```

Misclassification Rate = 0.119

``` r
# g.sknn = sknn(GENRE~.,data=gtrain,kn=10)
# g.yhat = predict(g.sknn,newdata=gval)
# 
# misclass(g.yhat$class,gval$GENRE)
```

Misclassification Rate = 0.131

Clearly, the results are going no where with more neighbors. Its highly
likely that keeping kn at 1 is going to produce the best results. Next,
we will try to optimize weighting for our predictive model.

``` r
# g.sknn = sknn(GENRE~.,data=gtrain,kn=1, gamma = 1)
# g.yhat = predict(g.sknn,newdata=gval)
# 
# misclass(g.yhat$class,gval$GENRE)
```

Misclassification Rate = 0.0858

``` r
# g.sknn = sknn(GENRE~.,data=gtrain,kn=1, gamma = 2)
# g.yhat = predict(g.sknn,newdata=gval)
# 
# misclass(g.yhat$class,gval$GENRE)
```

Misclassification Rate = 0.0894

``` r
# g.sknn = sknn(GENRE~.,data=gtrain,kn=1, gamma = 3)
# g.yhat = predict(g.sknn,newdata=gval)
# 
# misclass(g.yhat$class,gval$GENRE)
```

Misclassification Rate = 0.1

``` r
# g.sknn = sknn(GENRE~.,data=gtrain,kn=1, gamma = 4)
# g.yhat = predict(g.sknn,newdata=gval)
# 
# misclass(g.yhat$class,gval$GENRE)
```

Misclassification Rate = 0.121

``` r
# g.sknn = sknn(GENRE~.,data=gtrain,kn=1, gamma = 5)
# g.yhat = predict(g.sknn,newdata=gval)
# 
# misclass(g.yhat$class,gval$GENRE)
```

Misclassification Rate = 0.156

``` r
# g.sknn = sknn(GENRE~.,data=gtrain,kn=1, gamma = 6)
# g.yhat = predict(g.sknn,newdata=gval)
# 
# misclass(g.yhat$class,gval$GENRE)
```

Misclassification Rate = 0.199

``` r
# g.sknn = sknn(GENRE~.,data=gtrain,kn=1, gamma = 7)
# g.yhat = predict(g.sknn,newdata=gval)
# 
# misclass(g.yhat$class,gval$GENRE)
```

Misclassification Rate = 0.25

``` r
# g.sknn = sknn(GENRE~.,data=gtrain,kn=1, gamma = 8)
# g.yhat = predict(g.sknn,newdata=gval)
# 
# misclass(g.yhat$class,gval$GENRE)
```

Misclassification Rate = 0.305

``` r
# g.sknn = sknn(GENRE~.,data=gtrain,kn=1, gamma = 9)
# g.yhat = predict(g.sknn,newdata=gval)
# 
# misclass(g.yhat$class,gval$GENRE)
```

Misclassification Rate = 0.357

``` r
# g.sknn = sknn(GENRE~.,data=gtrain,kn=1, gamma = 10)
# g.yhat = predict(g.sknn,newdata=gval)
# 
# misclass(g.yhat$class,gval$GENRE)
```

Misclassification Rate = 0.413

While a few weighting schemes did come close, most still were not quite
able to beat the more simpl model. Although gamma 1 was off by only
0.6%. Either way, it seems as though weighting will likely not be how we
fully optimize our model, but it was worth our consideration. Perhaps
utilizing some kind of distance metric in conjunction with a smoother
will prove more bountiful.

``` r
# g.knnD = train.kknn(GENRE~.,data=gtrain,kn=1, kernel=c("triangular", "rectangular", "epanechnikov", "optimal", "biweight","triweight","gaussian","cos","rank"), distance = 1)
# yhatDg = predict(g.knnD,newdata=gval)
# 
# misclass(yhatDg,gval$GENRE)
```

Misclassification Rate = 0.0789

It appears that a smallier distance paramter performs the best, getting
our missclassification rate down to about 7.89%

Looking further at that model:

``` r
# summary(g.knnD)
```

This model seems pretty solid using nominal, although we should try some
greater combinations of distances and kernels to see what kind of
improvement we might find.

``` r
# g.knnD = train.kknn(GENRE~.,data=gtrain,kn=1, kernel=c("triangular", "rectangular", "epanechnikov", "optimal", "biweight","triweight","gaussian","cos","rank"), distance = 2)
# yhatDg = predict(g.knnD,newdata=gval)
# 
# misclass(yhatDg,gval$GENRE)
```

Misclassification Rate = 0.0858

``` r
# g.knnD = train.kknn(GENRE~.,data=gtrain,kn=1, kernel=c("triangular", "rectangular", "epanechnikov", "optimal", "biweight","triweight","gaussian","cos","rank"), distance = 3)
# yhatDg = predict(g.knnD,newdata=gval)
# 
# misclass(yhatDg,gval$GENRE)
```

Misclassification Rate = 0.0969

``` r
# g.knnD = train.kknn(GENRE~.,data=gtrain,kn=1, kernel=c("triangular", "rectangular", "epanechnikov", "optimal", "biweight","triweight","gaussian","cos","rank"), distance = 4)
# yhatDg = predict(g.knnD,newdata=gval)
# 
# misclass(yhatDg,gval$GENRE)
```

Misclassification Rate = 0.11

``` r
# g.knnD = train.kknn(GENRE~.,data=gtrain,kn=1, kernel=c("triangular", "rectangular", "epanechnikov", "optimal", "biweight","triweight","gaussian","cos","rank"), distance = 5)
# yhatDg = predict(g.knnD,newdata=gval)
# 
# misclass(yhatDg,gval$GENRE)
```

Misclassification Rate = 0.123

``` r
# g.knnD = train.kknn(GENRE~.,data=gtrain,kn=1, kernel=c("triangular", "rectangular", "epanechnikov", "optimal", "biweight","triweight","gaussian","cos","rank"), distance = 6)
# yhatDg = predict(g.knnD,newdata=gval)
# 
# misclass(yhatDg,gval$GENRE)
```

Misclassification Rate = 0.131

``` r
# g.knnD = train.kknn(GENRE~.,data=gtrain,kn=1, kernel=c("triangular", "rectangular", "epanechnikov", "optimal", "biweight","triweight","gaussian","cos","rank"), distance = 7)
# yhatDg = predict(g.knnD,newdata=gval)
# 
# misclass(yhatDg,gval$GENRE)
```

Misclassification Rate = 0.143

``` r
# g.knnD = train.kknn(GENRE~.,data=gtrain,kn=1, kernel=c("triangular", "rectangular", "epanechnikov", "optimal", "biweight","triweight","gaussian","cos","rank"), distance = 8)
# yhatDg = predict(g.knnD,newdata=gval)
# 
# misclass(yhatDg,gval$GENRE)
```

Misclassification Rate = 0.147

``` r
# g.knnD = train.kknn(GENRE~.,data=gtrain,kn=1, kernel=c("triangular", "rectangular", "epanechnikov", "optimal", "biweight","triweight","gaussian","cos","rank"), distance = 9)
# yhatDg = predict(g.knnD,newdata=gval)
# 
# misclass(yhatDg,gval$GENRE)
```

Misclassification Rate = 0.152

``` r
# g.knnD = train.kknn(GENRE~.,data=gtrain,kn=1, kernel=c("triangular", "rectangular", "epanechnikov", "optimal", "biweight","triweight","gaussian","cos","rank"), distance = 10)
# yhatDg = predict(g.knnD,newdata=gval)
# 
# misclass(yhatDg,gval$GENRE)
```

Misclassification Rate = 0.155

``` r
# g.knnD = train.kknn(GENRE~.,data=gtrain,kn=1, kernel=c("triangular",  "biweight"), distance = 11)
# yhatDg = predict(g.knnD,newdata=gval)
# 
# misclass(yhatDg,gval$GENRE)
```

Misclassification Rate = 0.158

None of these models improve off of the simplest one in terms of
distance, although kn 3 was very similar, so it may be worth trying that
as well. Leaving our lowest distance model to be the best.

As one final thing we can try, we can use the tune function from e1071
to try and have it search for optimal paramters.

``` r
#g.tune = tune.knn(x = gtrain[,-1], y = gtrain[,1],k = 1:10, tunecontrol = tune.control(sampling = "boot") )

#g.tune$best.performance
```

\[1\] 0.1006017

``` r
#summary(g.tune)
```

k <int> error <dbl> dispersion <dbl> 
1 0.1006017 0.008796545  
2 0.1259869 0.005100039  
3 0.1319405 0.009213037  
4 0.1370575 0.006859476  
5 0.1389167 0.006680368  
6 0.1423835 0.009331596  
7 0.1437095 0.008892682  
8 0.1460595 0.009197514  
9 0.1468114 0.009646099  
10 0.1504351 0.008092739

Trying out other cv functions.

``` r
#g.tune2 = tune.knn(x = gtrain[,-1], y = gtrain[,1],k = 1:10, tunecontrol = tune.control(sampling = "cross") )

#g.tune2$best.performance
```

\[1\] 0.07588157

k <int> error <dbl> dispersion <dbl> 
1 0.07588157 0.009671225  
2 0.09537771 0.011274021  
3 0.09957562 0.012750349  
4 0.10706785 0.015673815  
5 0.11396666 0.018521511  
6 0.12115948 0.017660852  
7 0.12836130 0.016476175  
8 0.13136340 0.015945581  
9 0.13436191 0.014353106  
10 0.13796012 0.015357159

While this does support that the a kn of 1 is optimal, it still is not
able to produce a fundamentally different person.

Still our best model remains as a simple nominal, distance 1, kn 1
model.

``` r
#g.knnD = train.kknn(GENRE~.,data=gtrain,kn=1, kernel=c("triangular", "rectangular", "epanechnikov", "optimal", "biweight","triweight","gaussian","cos","rank"), distance = 1)
```

Monte Carlo Cross Validation for k-NN
=====================================

We will also use Monte Carlo on the full original training set.

``` r
gtrain2 = read.csv("GenreTrain.csv")
gtrain.X2 = scale(gtrain2[,-192])
gtrain2 = data.frame(GENRE=gtrain2$GENRE,gtrain.X2)

#results = kknn.sscv(gtrain2,B=100,kmax=1, distance = 1, p = 0.33, kernel = "biweight")
#summary(results)
```

With 100 passes, we seem to hover at a rate around 10%, which is a
little worse then what we saw on the validation set, but not anything
out of the realm of possibility.

Based on all of the testing done and our model building procedure, we
reccomend a fairly simple k-nn model that considers only 1 other nearest
neighbor, uses are Minkowski distance paramter of 1, and a biweight
(beta(3,3)) kernel. While a simple model may seem odd, given the diverse
nature of audio signals associated with different genres as well as the
fact that certain elements are guranteed to be present in all of them, a
model that focuses on just a few key aspects could be expected to
perform much better than one would otherwise expect.

Submitting Predictions
======================

Using the k-NN classification procedure we chose above, we are going to
use our model to predict music genre on the test data set.

``` r
# g.knnD = train.kknn(GENRE~.,data=gtrain2,kn=1, kernel="biweight", distance = 1)
# final_pred = predict(g.knnD,newdata=gtest)
#write.csv(final_pred, file = "k-NN_audio_preidctions_Andrews_Weiczorek.csv")
```

