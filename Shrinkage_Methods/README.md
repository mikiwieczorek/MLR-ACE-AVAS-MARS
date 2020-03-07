Using the data frame Lu2004.
=============================

The response Age is the first column of the data frame, the remaining 403 columns are genetic marker intensity measurements for 403 different genes.

Using ridge and Lasso regression to develop optimal models for predicting Age of the subject. There are only n = 30 subjects in the full data set. This is an example of a wide data problem because n &lt;&lt; p (because 30 &lt;&lt; 403). Based on the research paper by Lu et al. (2004) that was published based on their analysis of these data.




College data frame in the ISLR library.
=======================================

U.S. News and World Report’s College Data

<b>Description</b> 
<br>Statistics for a large number of US Colleges from the
1995 issue of US News and World Report.</br>
<br><b>Usage</b> </br>
College
<br><b>Format</b></br>
A data frame with 777 observations on the following 18
variables.

Private - A factor with levels No and Yes indicating private or public
university

Apps - Number of applications received Accept - Number of applications
accepted

Enroll - Number of new students enrolled Top10perc - Pct. new students
from top 10% of H.S. class Top25perc - Pct. new students from top 25% of
H.S. class F.Undergrad - Number of fulltime undergraduates P.Undergrad -
Number of parttime undergraduates Outstate - Out-of-state tuition
Room.Board - Room and board costs Books - Estimated book costs Personal
- Estimated personal spending PhD - Pct. of faculty with Ph.D.’s
Terminal - Pct. of faculty with terminal degree S.F.Ratio -
Student/faculty ratio perc.alumni - Pct. alumni who donate Expend -
Instructional expenditure per student Grad.Rate - Graduation rate

<b>Source</b> This dataset was taken from the StatLib library which is
maintained at Carnegie Mellon University. The dataset was used in the
ASA Statistical Graphics Section’s 1995 Data Analysis Exposition.

<b>References</b> Games, G., Witten, D., Hastie, T., and Tibshirani, R.
(2013) An Introduction to Statistical Learning with applications in R,
www.StatLearning.com, Springer-Verlag, New York
