
Concrete <- read.csv("Concrete.csv")

set.seed(1)

sam = sample(1:1030,size=floor(.6666*1030),replace=F)
