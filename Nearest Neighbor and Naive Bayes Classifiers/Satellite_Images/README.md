SATELLITE IMAGE DATA
====================

Data Description
================

The goal here is to predict the type of ground cover from a satellite
image broken up into pixels. Description from UCI Machine Learning
database:

The database consists of the multi-spectral values of pixels in 3x3
neighborhoods in a satellite image, and the classification associated
with the central pixel in each neighborhood. The aim is to predict this
classification, given the multi-spectral values. In the sample database,
the class of a pixel is coded as a number.

The Landsat satellite data is one of the many sources of information
available for a scene. The interpretation of a scene by integrating
spatial data of diverse types and resolutions including multispectral
and radar data, maps indicating topography, land use etc. is expected to
assume significant importance with the onset of an era characterized by
integrative approaches to remote sensing (for example, NASA’s Earth
Observing System commencing this decade). Existing statistical methods
are ill-equipped for handling such diverse data types. Note that this is
not true for Landsat MSS data considered in isolation (as in this sample
database). This data satisfies the important requirements of being
numerical and at a single resolution, and standard maximum-likelihood
classification performs very well. Consequently, for this data, it
should be interesting to compare the performance of other methods
against the statistical approach.

One frame of Landsat MSS imagery consists of four digital images of the
same scene in different spectral bands. Two of these are in the visible
region (corresponding approximately to green and red regions of the
visible spectrum) and two are in the (near) infra-red. Each pixel is a
8-bit binary word, with 0 corresponding to black and 255 to white. The
spatial resolution of a pixel is about 80m x 80m. Each image contains
2340 x 3380 such pixels.

The database is a (tiny) sub-area of a scene, consisting of 82 x 100
pixels. Each line of data corresponds to a 3x3 square neighborhood of
pixels completely contained within the 82x100 sub-area. Each line
contains the pixel values in the four spectral bands (converted to
ASCII) of each of the 9 pixels in the 3x3 neighborhood and a number
indicating the classification label of the central pixel. The number is
a code for the following classes:

Number Class 1 red soil 2 cotton crop 3 grey soil 4 damp grey soil 5
soil with vegetation stubble 6 mixture class (all types present)  
7 very damp grey soil

Note: There are no examples with class 6 in this dataset.

The data is given in random order and certain lines of data have been
removed so you cannot reconstruct the original image from this dataset.

In each line of data the four spectral values for the top-left pixel are
given first followed by the four spectral values for the top-middle
pixel and then those for the top-right pixel, and so on with the pixels
read out in sequence left-to-right and top-to-bottom. Thus, the four
spectral values for the central pixel are given by attributes 17,18,19
and 20.

``` r
setwd(getwd())
SATimage = read.csv("SATimage.csv")
SATimage = data.frame(class=as.factor(SATimage$class),SATimage[,1:36])
```

This command makes sure that the response is interpreted as a factor
(categorical) rather than as a number. Use the SATimage as the data
frame throughout.

Create a test and training set using the code below:

``` r
set.seed(888)
testcases = sample(1:dim(SATimage)[1],1000,replace=F)
SATtest = SATimage[testcases,]
SATtrain = SATimage[-testcases,]
```

K-NN and Naïve Bayes Classification Models
==========================================

We will be comparing k-NN classification and Naïve Bayes classification
for predicting the test cases in SATtest

Mounting Libraries

``` r
require(kknn)
require(class)
require(klaR)
```

Nearest Neighbor classification missclassification function:

``` r
misclass = function(fit,y) {
temp <- table(fit,y)
cat("Table of Misclassification\n")
cat("(row = predicted, col = actual)\n")
print(temp)
cat("\n\n")
numcor <- sum(diag(temp))
numinc <- length(y) - numcor
mcr <- numinc/length(y)
cat(paste("Misclassification Rate = ",format(mcr,digits=3)))
cat("\n")
}
```

Scaling was considered, but since the variables are all in the same
scale, it was deemed unescescary.

Nearest Neighbor Classification:
================================

Starting with just a simple kn = 1 model

``` r
sat.sknn = sknn(class~.,data=SATtrain,kn=1)
yhat = predict(sat.sknn,newdata=SATtest)

misclass(yhat$class,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 245   0   1   1   3   0
    ##   2   0 103   0   3   2   0
    ##   3   4   0 199  11   0   7
    ##   4   0   0  15  62   0  14
    ##   5   3   1   1   0  88   4
    ##   7   0   1   6  15   9 202
    ## 
    ## 
    ## Misclassification Rate =  0.101

Even before putting work, we are only above wrong about 10% of the time,
which for minimal effort is quite impressive. However, we can still
attempt to improve this by modifying additional features.

First, we will attempt to optimize our number of neighbors.

``` r
sat.sknn = sknn(class~.,data=SATtrain,kn=2)
yhat = predict(sat.sknn,newdata=SATtest)

misclass(yhat$class,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 242   0   1   0   3   0
    ##   2   0 102   0   1   2   2
    ##   3   6   0 195   9   0   6
    ##   4   0   0  18  68   2  11
    ##   5   4   2   1   1  85   5
    ##   7   0   1   7  13  10 203
    ## 
    ## 
    ## Misclassification Rate =  0.105

``` r
sat.sknn = sknn(class~.,data=SATtrain,kn=3)
yhat = predict(sat.sknn,newdata=SATtest)

misclass(yhat$class,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 246   0   1   0   3   0
    ##   2   0 102   0   1   2   1
    ##   3   5   0 201  11   0   4
    ##   4   0   1  13  65   1  10
    ##   5   1   1   0   0  88   5
    ##   7   0   1   7  15   8 207
    ## 
    ## 
    ## Misclassification Rate =  0.091

``` r
sat.sknn = sknn(class~.,data=SATtrain,kn=4)
yhat = predict(sat.sknn,newdata=SATtest)

misclass(yhat$class,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 246   0   1   1   2   0
    ##   2   0 104   0   0   3   1
    ##   3   5   0 199  13   1   3
    ##   4   0   0  16  61   1  18
    ##   5   1   0   0   1  87   5
    ##   7   0   1   6  16   8 200
    ## 
    ## 
    ## Misclassification Rate =  0.103

``` r
sat.sknn = sknn(class~.,data=SATtrain,kn=5)
yhat = predict(sat.sknn,newdata=SATtest)

misclass(yhat$class,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 247   0   1   1   2   0
    ##   2   0 104   0   0   2   1
    ##   3   4   0 204  16   0   5
    ##   4   0   0  14  58   0  15
    ##   5   1   0   0   1  89   5
    ##   7   0   1   3  16   9 201
    ## 
    ## 
    ## Misclassification Rate =  0.097

``` r
sat.sknn = sknn(class~.,data=SATtrain,kn=6)
yhat = predict(sat.sknn,newdata=SATtest)

misclass(yhat$class,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 247   0   1   0   5   0
    ##   2   0 104   0   1   3   1
    ##   3   4   0 203  17   0   5
    ##   4   0   0  13  57   0  16
    ##   5   1   0   0   1  84   6
    ##   7   0   1   5  16  10 199
    ## 
    ## 
    ## Misclassification Rate =  0.106

``` r
sat.sknn = sknn(class~.,data=SATtrain,kn=7)
yhat = predict(sat.sknn,newdata=SATtest)

misclass(yhat$class,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 247   0   1   0   3   0
    ##   2   0 104   0   0   2   1
    ##   3   4   0 206  18   0   4
    ##   4   0   0  12  56   0  14
    ##   5   1   0   0   1  86   4
    ##   7   0   1   3  17  11 204
    ## 
    ## 
    ## Misclassification Rate =  0.097

``` r
sat.sknn = sknn(class~.,data=SATtrain,kn=8)
yhat = predict(sat.sknn,newdata=SATtest)

misclass(yhat$class,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 248   0   1   0   6   0
    ##   2   0 103   0   0   2   0
    ##   3   3   0 205  18   1   6
    ##   4   0   0  11  56   1  15
    ##   5   1   0   0   1  80   5
    ##   7   0   2   5  17  12 201
    ## 
    ## 
    ## Misclassification Rate =  0.107

``` r
sat.sknn = sknn(class~.,data=SATtrain,kn=9)
yhat = predict(sat.sknn,newdata=SATtest)

misclass(yhat$class,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 247   0   2   1   7   0
    ##   2   0 102   0   0   1   1
    ##   3   4   0 205  16   1   6
    ##   4   0   1   9  57   0  16
    ##   5   1   1   0   1  80   4
    ##   7   0   1   6  17  13 200
    ## 
    ## 
    ## Misclassification Rate =  0.109

``` r
sat.sknn = sknn(class~.,data=SATtrain,kn=10)
yhat = predict(sat.sknn,newdata=SATtest)

misclass(yhat$class,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 247   0   2   1   8   0
    ##   2   0 101   0   0   2   1
    ##   3   4   0 205  17   1   5
    ##   4   0   1   8  57   0  17
    ##   5   1   2   0   0  77   3
    ##   7   0   1   7  17  14 201
    ## 
    ## 
    ## Misclassification Rate =  0.112

Incresing the number of nearest neighbors beyond this doesn’t make a lot
of sense given the pattern were seeing. While the imporvment is around
1%, kn = 3 performs the best.

``` r
sat.sknn3 = sknn(class~.,data=SATtrain,kn=3)
yhat3 = predict(sat.sknn,newdata=SATtest)
```

Next, weighting will tried in a manner similar to more traditional
nearest neighbor regression.

``` r
sat.sknn3w = sknn(class~.,data=SATtrain,kn=3, gamma = 1)
yhat3w = predict(sat.sknn3w,newdata=SATtest)

misclass(yhat3w$class,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 212   0   1   0   0   0
    ##   2   0  49   0   1   0   0
    ##   3   0   0 182  10   0   3
    ##   4   0   0  15  55   0  11
    ##   5   2   0   0   0  53   2
    ##   7   0   0   4  11   2 183
    ## 
    ## 
    ## Misclassification Rate =  0.266

Immediate results aren’t boading well, but perhaps this changes with
weight increases.

``` r
sat.sknn3w = sknn(class~.,data=SATtrain,kn=3, gamma = 2)
yhat3w = predict(sat.sknn3w,newdata=SATtest)

misclass(yhat3w$class,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1  81   0   0   0   0   0
    ##   2   0  18   0   0   0   0
    ##   3   0   0  95   4   0   0
    ##   4   0   0   9  34   0   1
    ##   5   0   0   0   0  17   0
    ##   7   0   0   0   2   0 116
    ## 
    ## 
    ## Misclassification Rate =  0.639

``` r
sat.sknn3w = sknn(class~.,data=SATtrain,kn=3, gamma = 3)
yhat3w = predict(sat.sknn3w,newdata=SATtest)

misclass(yhat3w$class,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit  1  2  3  4  5  7
    ##   1 10  0  0  0  0  0
    ##   2  0  4  0  0  0  0
    ##   3  0  0 21  1  0  0
    ##   4  0  0  3  5  0  0
    ##   5  0  0  0  0  0  0
    ##   7  0  0  0  0  0 53
    ## 
    ## 
    ## Misclassification Rate =  0.907

``` r
sat.sknn3w = sknn(class~.,data=SATtrain,kn=3, gamma = 4)
yhat3w = predict(sat.sknn3w,newdata=SATtest)

misclass(yhat3w$class,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit  1  2  3  4  5  7
    ##   1  1  0  0  0  0  0
    ##   2  0  1  0  0  0  0
    ##   3  0  0  1  0  0  0
    ##   4  0  0  0  0  0  0
    ##   5  0  0  0  0  0  0
    ##   7  0  0  0  0  0 13
    ## 
    ## 
    ## Misclassification Rate =  0.984

``` r
sat.sknn3w = sknn(class~.,data=SATtrain,kn=3, gamma = 4)
yhat3w = predict(sat.sknn3w,newdata=SATtest)

misclass(yhat3w$class,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit  1  2  3  4  5  7
    ##   1  1  0  0  0  0  0
    ##   2  0  1  0  0  0  0
    ##   3  0  0  1  0  0  0
    ##   4  0  0  0  0  0  0
    ##   5  0  0  0  0  0  0
    ##   7  0  0  0  0  0 13
    ## 
    ## 
    ## Misclassification Rate =  0.984

``` r
sat.sknn3w = sknn(class~.,data=SATtrain,kn=3, gamma = 5)
yhat3w = predict(sat.sknn3w,newdata=SATtest)

misclass(yhat3w$class,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit 1 2 3 4 5 7
    ##   1 0 0 0 0 0 0
    ##   2 0 0 0 0 0 0
    ##   3 0 0 0 0 0 0
    ##   4 0 0 0 0 0 0
    ##   5 0 0 0 0 0 0
    ##   7 0 0 0 0 0 1
    ## 
    ## 
    ## Misclassification Rate =  0.999

Weighting clearly isn’t the answer, as with rbf up to 5 we almost are up
to predicintingg wrong everytime, with any weight at all doubling our
missclassification rate. Perhaps by using the manhattan or taxi cab
metric we can see some actual improvement.

``` r
sat.knnD = train.kknn(class~.,data=SATtrain,kn=3, kernel=c("triangular", "rectangular", "epanechnikov", "optimal"), distance = 1)  
yhatD = predict(sat.knnD,newdata=SATtest)

misclass(yhatD,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 247   0   1   0   3   0
    ##   2   0 103   0   1   2   0
    ##   3   3   0 208  13   0   7
    ##   4   0   0   9  61   0  12
    ##   5   2   1   0   0  87   4
    ##   7   0   1   4  17  10 204
    ## 
    ## 
    ## Misclassification Rate =  0.09

``` r
sat.knnD2 = train.kknn(class~.,data=SATtrain,kn=3, kernel=c("triangular", "rectangular", "epanechnikov", "optimal"), distance = 2)  
yhatD2 = predict(sat.knnD2,newdata=SATtest)

misclass(yhatD2,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 247   0   1   0   3   0
    ##   2   0 103   0   1   2   1
    ##   3   4   0 205  12   0   6
    ##   4   0   0  11  64   0  14
    ##   5   1   1   0   0  86   5
    ##   7   0   1   5  15  11 201
    ## 
    ## 
    ## Misclassification Rate =  0.094

``` r
sat.knnD2 = train.kknn(class~.,data=SATtrain,kn=3, kernel=c("triangular", "rectangular", "epanechnikov", "optimal"), distance = 3)  
yhatD2 = predict(sat.knnD2,newdata=SATtest)

misclass(yhatD2,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 247   0   1   0   4   0
    ##   2   0 103   0   1   2   1
    ##   3   4   0 206  11   0   6
    ##   4   0   0  11  63   0  17
    ##   5   1   1   0   0  86   5
    ##   7   0   1   4  17  10 198
    ## 
    ## 
    ## Misclassification Rate =  0.097

Putting more emphasis on Mikowski distance isn’t helping, perhaps
modifying the kernel will make a difference.

``` r
sat.knnD2 = train.kknn(class~.,data=SATtrain,kn=1, kernel=c("triangular", "epanechnikov", "optimal"), distance = 3)  
yhatD2 = predict(sat.knnD2,newdata=SATtest)

misclass(yhatD2,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 247   0   1   0   4   0
    ##   2   0 103   0   1   2   1
    ##   3   4   0 206  11   0   6
    ##   4   0   0  11  63   0  17
    ##   5   1   1   0   0  86   5
    ##   7   0   1   4  17  10 198
    ## 
    ## 
    ## Misclassification Rate =  0.097

removing rectangular did not help, even being the simplest.

``` r
sat.knnD2 = train.kknn(class~.,data=SATtrain,kn=3, kernel=c("triangular", "rectangular", "epanechnikov", "optimal", "biweight"), distance = 1)  
yhatD2 = predict(sat.knnD2,newdata=SATtest)

misclass(yhatD2,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 247   0   1   0   3   0
    ##   2   0 103   0   1   2   0
    ##   3   3   0 204  14   0   8
    ##   4   0   0  12  63   0  10
    ##   5   2   1   0   0  89   3
    ##   7   0   1   5  14   8 206
    ## 
    ## 
    ## Misclassification Rate =  0.088

Adding biweight as an additional helped, but what if triweight pushes it
further?

``` r
sat.knnD2 = train.kknn(class~.,data=SATtrain,kn=3, kernel=c("triangular", "rectangular", "epanechnikov", "optimal", "biweight", "triweight"), distance = 1)  
yhatD2 = predict(sat.knnD2,newdata=SATtest)

misclass(yhatD2,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 247   0   1   0   3   0
    ##   2   0 103   0   1   2   0
    ##   3   3   0 204  14   0   8
    ##   4   0   0  12  63   0  10
    ##   5   2   1   0   0  89   3
    ##   7   0   1   5  14   8 206
    ## 
    ## 
    ## Misclassification Rate =  0.088

It does the same. What if we do just triweight?

``` r
sat.knnD2 = train.kknn(class~.,data=SATtrain,kn=3, kernel=c("triangular", "rectangular", "epanechnikov", "optimal", "triweight"), distance = 1)  
yhatD2 = predict(sat.knnD2,newdata=SATtest)

misclass(yhatD2,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 248   0   1   0   3   0
    ##   2   0 103   0   1   2   0
    ##   3   2   0 206  12   0   8
    ##   4   0   0  11  65   0  10
    ##   5   2   1   0   0  89   3
    ##   7   0   1   4  14   8 206
    ## 
    ## 
    ## Misclassification Rate =  0.083

Having both was definetly holding us back.

``` r
sat.knnD2 = train.kknn(class~.,data=SATtrain,kn=3, kernel=c("triangular", "rectangular", "epanechnikov", "optimal", "triweight", "gaussian"), distance = 1)  
yhatD2 = predict(sat.knnD2,newdata=SATtest)

misclass(yhatD2,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 248   0   1   0   3   0
    ##   2   0 103   0   1   2   0
    ##   3   2   0 206  12   0   8
    ##   4   0   0  11  65   0  10
    ##   5   2   1   0   0  89   3
    ##   7   0   1   4  14   8 206
    ## 
    ## 
    ## Misclassification Rate =  0.083

Guassian as an addition did not help.

``` r
sat.knnD2 = train.kknn(class~.,data=SATtrain,kn=3, kernel=c("triangular", "rectangular", "epanechnikov", "optimal", "triweight", "cos"), distance = 1)  
yhatD2 = predict(sat.knnD2,newdata=SATtest)

misclass(yhatD2,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 248   0   1   0   3   0
    ##   2   0 103   0   1   2   0
    ##   3   2   0 206  12   0   8
    ##   4   0   0  11  65   0  10
    ##   5   2   1   0   0  89   3
    ##   7   0   1   4  14   8 206
    ## 
    ## 
    ## Misclassification Rate =  0.083

A similar sotry with a cos addition. Perhaps we need to make the model
simpler.

``` r
sat.knnD2 = train.kknn(class~.,data=SATtrain,kn=3, kernel=c( "rectangular", "epanechnikov", "optimal", "triweight"), distance = 1)  
yhatD2 = predict(sat.knnD2,newdata=SATtest)

misclass(yhatD2,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 248   0   1   0   3   0
    ##   2   0 103   0   1   2   0
    ##   3   2   0 206  12   0   8
    ##   4   0   0  11  65   0  10
    ##   5   2   1   0   0  89   3
    ##   7   0   1   4  14   8 206
    ## 
    ## 
    ## Misclassification Rate =  0.083

This leaves our best model as a triweghit kn = 3 model with Minkowski
distance of 1

``` r
sat.knnD2 = train.kknn(class~.,data=SATtrain,kn=3, kernel=c("triweight"), distance = 1)  
yhatD2 = predict(sat.knnD2,newdata=SATtest)

misclass(yhatD2,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 248   0   1   0   3   0
    ##   2   0 103   0   1   2   0
    ##   3   2   0 206  12   0   8
    ##   4   0   0  11  65   0  10
    ##   5   2   1   0   0  89   3
    ##   7   0   1   4  14   8 206
    ## 
    ## 
    ## Misclassification Rate =  0.083

Overall, from the simplest model, we improved classification performance
by around 2%.

Naïve Bayes:
============

Packages

``` r
require(e1071)
require(s20x)
```

Given the high number of variables we are working with, checking all of
them for normality is not practical. It will be far more efficent to
simply run the analysis assuming both potential conditions and see which
performs better.

Starting simple with klar’s implementation.

``` r
sat.NB = NaiveBayes(class~.,data=SATtrain)

summary(sat.NB)
```

    ##           Length Class      Mode     
    ## apriori    6     table      numeric  
    ## tables    36     -none-     list     
    ## levels     6     -none-     character
    ## call       3     -none-     call     
    ## x         36     data.frame list     
    ## usekernel  1     -none-     logical  
    ## varnames  36     -none-     character

``` r
yprednb = predict(sat.NB,newdata=SATtest)

misclass(yprednb$class,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 195   5   3   1  11   0
    ##   2   0  90   0   0   1   0
    ##   3   9   0 196  12   0   2
    ##   4   0   0  21  62   1  46
    ##   5  48  10   0   0  78   8
    ##   7   0   0   2  17  11 171
    ## 
    ## 
    ## Misclassification Rate =  0.208

Given that assuming a conditional normal distribution for each predictor
lead us to a result with more twice the missclassification rate of even
the simplest nearest neighbor model, we likely are oging to want to
consider more advnaced models using smoothers.

``` r
sat.NB = NaiveBayes(class~.,data=SATtrain, usekernel = T)
yprednb = predict(sat.NB,newdata=SATtest)

misclass(yprednb$class,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 205   3   1   1  10   0
    ##   2   0  97   0   0   2   0
    ##   3  11   0 198  13   0   2
    ##   4   0   1  21  63   0  47
    ##   5  36   4   2   0  80   4
    ##   7   0   0   0  15  10 174
    ## 
    ## 
    ## Misclassification Rate =  0.183

An improvement for sure, but still far behind a result we would consider
good. Perhaps if we force the kernel to be more smooth we will get a
result more desireable.

``` r
sat.NB = NaiveBayes(class~.,data=SATtrain, usekernel = T, bw = T)
yprednb = predict(sat.NB,newdata=SATtest)

misclass(yprednb$class,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 209   3   1   1  10   0
    ##   2   0  97   0   0   3   0
    ##   3  10   0 199  13   0   1
    ##   4   0   1  21  59   0  48
    ##   5  33   4   1   4  80   5
    ##   7   0   0   0  15   9 173
    ## 
    ## 
    ## Misclassification Rate =  0.183

No difference is made apparent. Perhaps a different library, such as
e1071, has a slight edge over klar for this problem.

``` r
sat.NB = naiveBayes(class~.,data=SATtrain)
yprednb = predict(sat.NB,newdata=SATtest)

misclass(yprednb,SATtest$class)
```

    ## Table of Misclassification
    ## (row = predicted, col = actual)
    ##    y
    ## fit   1   2   3   4   5   7
    ##   1 195   5   3   1  11   0
    ##   2   0  90   0   0   1   0
    ##   3   9   0 196  12   0   2
    ##   4   0   0  21  62   1  46
    ##   5  48  10   0   0  78   8
    ##   7   0   0   2  17  11 171
    ## 
    ## 
    ## Misclassification Rate =  0.208

It seems that it made no difference. Our kernel smoothed Klar model
comes out on top.

Final Models:

Nearest-Neighbor:

``` r
sat.knnD2 = train.kknn(class~.,data=SATtrain,kn=3, kernel=c("triweight"), distance = 1)
```

Missclassification Rate = 8.3%

Naive-Bayes:

``` r
sat.NB = NaiveBayes(class~.,data=SATtrain, usekernel = T, bw = T)
```

Missclassification Rate = 18.3%

Monte Carlo Cross-Validation
============================

Using Split-Sample Monte Carlo cross-validation to compare k-NN and
Naïve Bayes to compare these methods of classification.

First, we will look at our k-NN model.

Monte Carlo Function for k-NN:
------------------------------

``` r
kknn.sscv = function(train,y=train[,1],B=25,p=.333,kmax=3,kernel="optimal",distance=2) {
  y = as.factor(y)
  data = data.frame(y=y,train[,-1])
  n = length(y)
  cv <- rep(0,B)
  leaveout = floor(n*p)
  for (i in 1:B) {
    sam <- sample(1:n,leaveout,replace=F)
    fit <- train.kknn(y~.,data=data[-sam,],kmax=kmax,kernel=kernel,distance=distance)
    ypred = predict(fit,newdata=data[sam,])
    tab <- table(y[sam],ypred)
    mc <- leaveout - sum(diag(tab))
    cv[i] <- mc/leaveout
  }
  cv
}
```

Result:

``` r
#results = kknn.sscv(SATimage,B=100,kmax=3,kernel="triweight", distance = 1, p = 0.33)

#summary(results)
```

Min. 1st Qu. Median Mean 3rd Qu. Max. 0.08808 0.09756 0.10196 0.10159
0.10586 0.11856

Over 100 iterations, it seems that our observed metric of 8.3%
missclassification was on the low end for a model of that type. Still,
averaging about 11% is not horrible, but the result was not what one
might have fully hoped for.

Monte Carlo Function for Naive Bayes
------------------------------------

``` r
nB.cv = function(X,y,B=25,p=.333,laplace=0) {
y = as.factor(y)
data = data.frame(y,X)
n = length(y)
cv <- rep(0,B)
leaveout = floor(n*p)
for (i in 1:B) {
        sam <- sample(1:n,leaveout,replace=F)
        temp <- data[-sam,]
        fit <- naiveBayes(y~.,data=temp,laplace=laplace)
        pred = predict(fit,newdata=X[sam,])
        tab <- table(y[sam],pred)
        mc <- leaveout - sum(diag(tab))
        cv[i] <- mc/leaveout
        }
  cv
}
```

Result:

``` r
#results2 = nB.cv(SATimage[,-1],SATimage[,1],B=100)

#summary(results2)
```

Min. 1st Qu. Median Mean 3rd Qu. Max. 0.1809 0.1963 0.2033 0.2031 0.2087
0.2256

Naive-Bayes peformed similarly, being slightly worse through 100 passes
in a Monte Carlo function. However, it still holds true that our k-NN
model is roughly twice as effective than Naive-Bayes using
missclassification as our chosen metric, with the average pass for Bayes
garnering wrong answers anout 20% of the time compared to 10% from k-NN.

