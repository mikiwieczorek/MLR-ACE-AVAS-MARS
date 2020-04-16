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

Table of Misclassification (row = predicted, col = actual) y fit Blues
Classical Jazz Metal Pop Rock Blues 393 1 15 7 4 2 Classical 5 872 100 4
4 6 Jazz 4 30 658 3 14 18 Metal 0 0 4 229 3 9 Pop 1 0 1 3 376 6 Rock 1 1
17 7 14 521

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

Table of Misclassification (row = predicted, col = actual) y fit Blues
Classical Jazz Metal Pop Rock Blues 390 3 18 7 6 6 Classical 6 871 129 3
6 8 Jazz 6 28 619 5 12 28 Metal 0 0 2 224 3 8 Pop 2 1 3 2 371 7 Rock 0 1
24 12 17 505

Misclassification Rate = 0.106

``` r
# g.sknn = sknn(GENRE~.,data=gtrain,kn=3)
# g.yhat = predict(g.sknn,newdata=gval)
# 
# misclass(g.yhat$class,gval$GENRE)
```

Table of Misclassification (row = predicted, col = actual) y fit Blues
Classical Jazz Metal Pop Rock Blues 389 1 14 8 5 6 Classical 10 879 128
3 6 9 Jazz 5 23 633 4 13 24 Metal 0 0 1 230 4 5 Pop 0 0 3 3 369 6 Rock 0
1 16 5 18 512

Misclassification Rate = 0.0963

``` r
# g.sknn = sknn(GENRE~.,data=gtrain,kn=4)
# g.yhat = predict(g.sknn,newdata=gval)
# 
# misclass(g.yhat$class,gval$GENRE)
```

Table of Misclassification (row = predicted, col = actual) y fit Blues
Classical Jazz Metal Pop Rock Blues 386 2 13 8 6 10 Classical 13 878 149
4 5 11 Jazz 5 24 607 3 21 25 Metal 0 0 4 228 3 8 Pop 0 0 4 2 365 7 Rock
0 0 18 8 15 501

Misclassification Rate = 0.11

``` r
# g.sknn = sknn(GENRE~.,data=gtrain,kn=5)
# g.yhat = predict(g.sknn,newdata=gval)
# 
# misclass(g.yhat$class,gval$GENRE)
```

Table of Misclassification (row = predicted, col = actual) y fit Blues
Classical Jazz Metal Pop Rock Blues 385 2 10 8 5 9 Classical 12 883 151
6 6 10 Jazz 5 19 611 3 13 22 Metal 0 0 3 226 6 2 Pop 2 0 4 3 368 7 Rock
0 0 16 7 17 512

Misclassification Rate = 0.104

``` r
# g.sknn = sknn(GENRE~.,data=gtrain,kn=7)
# g.yhat = predict(g.sknn,newdata=gval)
# 
# misclass(g.yhat$class,gval$GENRE)
```

Table of Misclassification (row = predicted, col = actual) y fit Blues
Classical Jazz Metal Pop Rock Blues 380 3 14 9 6 12 Classical 16 881 165
5 8 14 Jazz 7 20 592 3 15 26 Metal 0 0 4 227 4 6 Pop 1 0 3 2 359 7 Rock
0 0 17 7 23 497

Misclassification Rate = 0.119

``` r
# g.sknn = sknn(GENRE~.,data=gtrain,kn=10)
# g.yhat = predict(g.sknn,newdata=gval)
# 
# misclass(g.yhat$class,gval$GENRE)
```

Table of Misclassification (row = predicted, col = actual) y fit Blues
Classical Jazz Metal Pop Rock Blues 377 4 15 11 4 13 Classical 18 877
190 6 7 15 Jazz 9 23 567 2 16 27 Metal 0 0 2 223 5 9 Pop 0 0 3 2 362 8
Rock 0 0 18 9 21 490

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

Table of Misclassification (row = predicted, col = actual) y fit Blues
Classical Jazz Metal Pop Rock Blues 393 1 15 7 4 2 Classical 5 872 100 4
4 6 Jazz 4 30 656 2 14 18 Metal 0 0 4 229 3 9 Pop 1 0 1 3 376 6 Rock 1 1
17 7 13 521

Misclassification Rate = 0.0858

``` r
# g.sknn = sknn(GENRE~.,data=gtrain,kn=1, gamma = 2)
# g.yhat = predict(g.sknn,newdata=gval)
# 
# misclass(g.yhat$class,gval$GENRE)
```

Table of Misclassification (row = predicted, col = actual) y fit Blues
Classical Jazz Metal Pop Rock Blues 392 1 11 7 3 2 Classical 5 870 94 4
4 6 Jazz 4 30 649 2 12 18 Metal 0 0 4 229 3 9 Pop 1 0 1 2 374 6 Rock 1 1
17 7 13 521

Misclassification Rate = 0.0894

``` r
# g.sknn = sknn(GENRE~.,data=gtrain,kn=1, gamma = 3)
# g.yhat = predict(g.sknn,newdata=gval)
# 
# misclass(g.yhat$class,gval$GENRE)
```

Table of Misclassification (row = predicted, col = actual) y fit Blues
Classical Jazz Metal Pop Rock Blues 391 1 10 6 3 2 Classical 5 864 87 3
3 6 Jazz 4 30 621 0 12 16 Metal 0 0 2 229 3 9 Pop 1 0 1 2 373 5 Rock 1 1
16 7 13 521

Misclassification Rate = 0.1

``` r
# g.sknn = sknn(GENRE~.,data=gtrain,kn=1, gamma = 4)
# g.yhat = predict(g.sknn,newdata=gval)
# 
# misclass(g.yhat$class,gval$GENRE)
```

Table of Misclassification (row = predicted, col = actual) y fit Blues
Classical Jazz Metal Pop Rock Blues 384 1 8 5 2 2 Classical 3 847 74 3 0
6 Jazz 3 23 589 0 9 14 Metal 0 0 2 227 3 8 Pop 1 0 1 2 363 3 Rock 1 1 16
7 13 520

Misclassification Rate = 0.121

``` r
# g.sknn = sknn(GENRE~.,data=gtrain,kn=1, gamma = 5)
# g.yhat = predict(g.sknn,newdata=gval)
# 
# misclass(g.yhat$class,gval$GENRE)
```

Table of Misclassification (row = predicted, col = actual) y fit Blues
Classical Jazz Metal Pop Rock Blues 379 1 6 4 2 1 Classical 2 809 47 2 0
6 Jazz 3 18 544 0 8 12 Metal 0 0 1 226 3 8 Pop 0 0 0 2 336 3 Rock 1 1 15
6 12 518

Misclassification Rate = 0.156

``` r
# g.sknn = sknn(GENRE~.,data=gtrain,kn=1, gamma = 6)
# g.yhat = predict(g.sknn,newdata=gval)
# 
# misclass(g.yhat$class,gval$GENRE)
```

Table of Misclassification (row = predicted, col = actual) y fit Blues
Classical Jazz Metal Pop Rock Blues 360 1 5 4 2 1 Classical 2 774 37 0 0
5 Jazz 3 16 499 0 5 11 Metal 0 0 0 225 3 8 Pop 0 0 0 2 303 2 Rock 1 1 14
6 8 508

Misclassification Rate = 0.199

``` r
# g.sknn = sknn(GENRE~.,data=gtrain,kn=1, gamma = 7)
# g.yhat = predict(g.sknn,newdata=gval)
# 
# misclass(g.yhat$class,gval$GENRE)
```

Table of Misclassification (row = predicted, col = actual) y fit Blues
Classical Jazz Metal Pop Rock Blues 347 0 1 3 1 1 Classical 2 727 28 0 0
4 Jazz 1 15 450 0 3 8 Metal 0 0 0 221 3 8 Pop 0 0 0 2 260 2 Rock 0 1 9 5
3 494

Misclassification Rate = 0.25

``` r
# g.sknn = sknn(GENRE~.,data=gtrain,kn=1, gamma = 8)
# g.yhat = predict(g.sknn,newdata=gval)
# 
# misclass(g.yhat$class,gval$GENRE)
```

Table of Misclassification (row = predicted, col = actual) y fit Blues
Classical Jazz Metal Pop Rock Blues 330 0 1 2 1 0 Classical 0 665 19 0 0
3 Jazz 1 12 393 0 3 7 Metal 0 0 0 218 3 8 Pop 0 0 0 2 226 0 Rock 0 0 8 5
1 485

Misclassification Rate = 0.305

``` r
# g.sknn = sknn(GENRE~.,data=gtrain,kn=1, gamma = 9)
# g.yhat = predict(g.sknn,newdata=gval)
# 
# misclass(g.yhat$class,gval$GENRE)
```

Table of Misclassification (row = predicted, col = actual) y fit Blues
Classical Jazz Metal Pop Rock Blues 308 0 0 2 1 0 Classical 0 615 11 0 0
1 Jazz 0 7 333 0 2 6 Metal 0 0 0 214 2 5 Pop 0 0 0 0 199 0 Rock 0 0 8 5
1 473

Misclassification Rate = 0.357

``` r
# g.sknn = sknn(GENRE~.,data=gtrain,kn=1, gamma = 10)
# g.yhat = predict(g.sknn,newdata=gval)
# 
# misclass(g.yhat$class,gval$GENRE)
```

Table of Misclassification (row = predicted, col = actual) y fit Blues
Classical Jazz Metal Pop Rock Blues 284 0 0 1 0 0 Classical 0 548 9 0 0
1 Jazz 0 5 283 0 2 4 Metal 0 0 0 209 0 5 Pop 0 0 0 0 170 0 Rock 0 0 7 3
1 461

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

Table of Misclassification (row = predicted, col = actual) y fit Blues
Classical Jazz Metal Pop Rock Blues 394 0 8 5 3 1 Classical 6 879 116 4
3 8 Jazz 3 25 659 0 16 15 Metal 0 0 1 232 2 4 Pop 1 0 1 2 378 6 Rock 0 0
10 10 13 528

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

Table of Misclassification (row = predicted, col = actual) y fit Blues
Classical Jazz Metal Pop Rock Blues 393 1 15 7 5 1 Classical 6 872 102 4
3 7 Jazz 4 30 657 3 14 18 Metal 0 0 3 229 3 9 Pop 0 0 1 3 375 6 Rock 1 1
17 7 15 521

Misclassification Rate = 0.0858

``` r
# g.knnD = train.kknn(GENRE~.,data=gtrain,kn=1, kernel=c("triangular", "rectangular", "epanechnikov", "optimal", "biweight","triweight","gaussian","cos","rank"), distance = 3)
# yhatDg = predict(g.knnD,newdata=gval)
# 
# misclass(yhatDg,gval$GENRE)
```

Table of Misclassification (row = predicted, col = actual) y fit Blues
Classical Jazz Metal Pop Rock Blues 384 5 19 8 4 2 Classical 8 869 117 4
5 9 Jazz 7 29 634 4 11 20 Metal 1 0 2 225 3 9 Pop 3 0 3 1 380 4 Rock 1 1
20 11 12 518

Misclassification Rate = 0.0969

``` r
# g.knnD = train.kknn(GENRE~.,data=gtrain,kn=1, kernel=c("triangular", "rectangular", "epanechnikov", "optimal", "biweight","triweight","gaussian","cos","rank"), distance = 4)
# yhatDg = predict(g.knnD,newdata=gval)
# 
# misclass(yhatDg,gval$GENRE)
```

Table of Misclassification (row = predicted, col = actual) y fit Blues
Classical Jazz Metal Pop Rock Blues 383 8 19 10 4 5 Classical 10 862 132
3 7 11 Jazz 8 33 618 5 13 24 Metal 1 0 2 222 3 10 Pop 1 1 4 2 376 6 Rock
1 0 20 11 12 506

Misclassification Rate = 0.11

``` r
# g.knnD = train.kknn(GENRE~.,data=gtrain,kn=1, kernel=c("triangular", "rectangular", "epanechnikov", "optimal", "biweight","triweight","gaussian","cos","rank"), distance = 5)
# yhatDg = predict(g.knnD,newdata=gval)
# 
# misclass(yhatDg,gval$GENRE)
```

Table of Misclassification (row = predicted, col = actual) y fit Blues
Classical Jazz Metal Pop Rock Blues 382 11 25 10 3 4 Classical 10 852
135 5 7 11 Jazz 9 39 602 4 17 30 Metal 1 0 2 217 3 12 Pop 1 2 8 2 372 8
Rock 1 0 23 15 13 497

Misclassification Rate = 0.123

``` r
# g.knnD = train.kknn(GENRE~.,data=gtrain,kn=1, kernel=c("triangular", "rectangular", "epanechnikov", "optimal", "biweight","triweight","gaussian","cos","rank"), distance = 6)
# yhatDg = predict(g.knnD,newdata=gval)
# 
# misclass(yhatDg,gval$GENRE)
```

Table of Misclassification (row = predicted, col = actual) y fit Blues
Classical Jazz Metal Pop Rock Blues 380 12 24 12 6 6 Classical 11 852
148 5 8 12 Jazz 10 38 591 4 16 33 Metal 1 0 2 217 3 13 Pop 1 2 9 2 367
10 Rock 1 0 21 13 15 488

Misclassification Rate = 0.131

``` r
# g.knnD = train.kknn(GENRE~.,data=gtrain,kn=1, kernel=c("triangular", "rectangular", "epanechnikov", "optimal", "biweight","triweight","gaussian","cos","rank"), distance = 7)
# yhatDg = predict(g.knnD,newdata=gval)
# 
# misclass(yhatDg,gval$GENRE)
```

Table of Misclassification (row = predicted, col = actual) y fit Blues
Classical Jazz Metal Pop Rock Blues 376 11 29 11 6 6 Classical 15 848
159 4 10 14 Jazz 11 40 579 5 16 37 Metal 0 0 0 214 3 16 Pop 1 4 6 4 361
9 Rock 1 1 22 15 19 480

Misclassification Rate = 0.143

``` r
# g.knnD = train.kknn(GENRE~.,data=gtrain,kn=1, kernel=c("triangular", "rectangular", "epanechnikov", "optimal", "biweight","triweight","gaussian","cos","rank"), distance = 8)
# yhatDg = predict(g.knnD,newdata=gval)
# 
# misclass(yhatDg,gval$GENRE)
```

Table of Misclassification (row = predicted, col = actual) y fit Blues
Classical Jazz Metal Pop Rock Blues cv 13 27 13 7 6 Classical 14 843 161
5 10 13 Jazz 11 42 574 4 12 35 Metal 1 0 0 208 3 17 Pop 3 5 11 4 365 11
Rock 1 1 22 19 18 480

Misclassification Rate = 0.147

``` r
# g.knnD = train.kknn(GENRE~.,data=gtrain,kn=1, kernel=c("triangular", "rectangular", "epanechnikov", "optimal", "biweight","triweight","gaussian","cos","rank"), distance = 9)
# yhatDg = predict(g.knnD,newdata=gval)
# 
# misclass(yhatDg,gval$GENRE)
```

Table of Misclassification (row = predicted, col = actual) y fit Blues
Classical Jazz Metal Pop Rock Blues 369 14 27 13 9 6 Classical 17 839
161 5 10 13 Jazz 12 45 572 4 13 35 Metal 1 0 0 206 3 16 Pop 3 5 12 5 362
12 Rock 2 1 23 20 18 480

Misclassification Rate = 0.152

``` r
# g.knnD = train.kknn(GENRE~.,data=gtrain,kn=1, kernel=c("triangular", "rectangular", "epanechnikov", "optimal", "biweight","triweight","gaussian","cos","rank"), distance = 10)
# yhatDg = predict(g.knnD,newdata=gval)
# 
# misclass(yhatDg,gval$GENRE)
```

Table of Misclassification (row = predicted, col = actual) y fit Blues
Classical Jazz Metal Pop Rock Blues 367 14 30 13 10 7 Classical 16 838
163 5 10 14 Jazz 14 46 568 4 14 36 Metal 1 0 0 207 3 17 Pop 3 5 12 5 360
12 Rock 3 1 22 19 18 476

Misclassification Rate = 0.155

``` r
# g.knnD = train.kknn(GENRE~.,data=gtrain,kn=1, kernel=c("triangular",  "biweight"), distance = 11)
# yhatDg = predict(g.knnD,newdata=gval)
# 
# misclass(yhatDg,gval$GENRE)
```

Table of Misclassification (row = predicted, col = actual) y fit Blues
Classical Jazz Metal Pop Rock Blues 365 13 32 14 11 7 Classical 16 839
165 5 10 14 Jazz 16 45 563 4 14 39 Metal 1 0 1 207 3 16 Pop 3 5 12 4 359
13 Rock 3 2 22 19 18 473

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

k <int> error <dbl> dispersion <dbl> 1 0.1006017 0.008796545  
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

k <int> error <dbl> dispersion <dbl> 1 0.07588157 0.009671225  
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

