Annotated R code - Practical Machine Learning Course Project
============
Load libraries and data.


```r
library( caret );
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library( doMC );
```

```
## Loading required package: foreach
## Loading required package: iterators
## Loading required package: parallel
```

```r
registerDoMC( cores = 2 );
projectTrain <- read.csv( '~/Downloads/pml-training.csv' );
```

Split the data into training and validation sets.


```r
inTrain <- createDataPartition( projectTrain$classe, p=0.7, list=FALSE);
train <- projectTrain[ inTrain,];
validate <- projectTrain[ -inTrain,];
```

Visually inspecting data shows it is time series for 6 different user_names with the dependent variable 'classe' occurring in groups if the data is sorted by time.  So I tried a simple model with those variables.


```r
simpleModel <- train( classe ~ user_name + num_window, data=train, method='rf', trControl=trainControl( method='oob' ), tuneGrid = expand.grid( mtry = 6 ));
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

Inspect model results in training and validation data.


```r
simpleModel;
```

```
## Random Forest 
## 
## 13737 samples
##   159 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Out of Bag Resampling 
## 
## Summary of sample sizes:  
## 
## Resampling results
## 
##   Accuracy  Kappa
##   1         1    
## 
## Tuning parameter 'mtry' was held constant at a value of 6
## 
```

```r
confusionMatrix( predict( simpleModel, validate ), validate$classe );
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    1    0    0    0
##          B    0 1138    0    0    0
##          C    0    0 1026    0    0
##          D    0    0    0  964    1
##          E    0    0    0    0 1081
## 
## Overall Statistics
##                                     
##                Accuracy : 1         
##                  95% CI : (0.999, 1)
##     No Information Rate : 0.284     
##     P-Value [Acc > NIR] : <2e-16    
##                                     
##                   Kappa : 1         
##  Mcnemar's Test P-Value : NA        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.999    1.000    1.000    0.999
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          0.999    1.000    1.000    0.999    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.174    0.164    0.184
## Detection Prevalence    0.285    0.193    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```
The model is a perfect fit in the training data and 99.9% accurate in out-of-sample validation.  So I created the answers for course project based on test set.

```r
test <- read.csv( '~/downloads/pml-testing.csv' );
answers <- predict( simpleModel, test );
```
To make a more real world example I removed columns with #N/A or '' entries and then removed the 'user_name' and all time related columns.  This seems more useful in the real world where exercise feedback will need to respond to situations where the exercise style in the previous period is not known.


```r
train <- train [, colSums( is.na( train ) ) == 0];
train <- train [, colSums(  train == '' ) == 0];
train <- train [, 8:60];
```
This left 52 predictors available which was computationally heavy.  So I tried to identify the most interesting predictors.


```r
ctrl <- rfeControl( functions=rfFuncs, method='oob', index=TRUE );
rfProfile <- rfe( train[,1:52], train[,53], sizes=12 , rfeControl=ctrl );
rfProfile;
```

```
## 
## Recursive feature selection
## 
## Outer resampling method: Out of Bag Resampling 
## 
## Resampling performance over subset size:
## 
##  Variables Accuracy Kappa AccuracySD KappaSD Selected
##         12        1     1         NA      NA        *
##         52        1     1         NA      NA         
## 
## The top 5 variables (out of 12):
##    roll_belt, yaw_belt, magnet_dumbbell_z, pitch_belt, magnet_dumbbell_y
```

```r
predictors( rfProfile );
```

```
##  [1] "roll_belt"         "yaw_belt"          "magnet_dumbbell_z"
##  [4] "pitch_belt"        "magnet_dumbbell_y" "pitch_forearm"    
##  [7] "roll_forearm"      "accel_dumbbell_y"  "roll_dumbbell"    
## [10] "magnet_dumbbell_x" "magnet_forearm_z"  "roll_arm"
```
This listed the 12 best predictors.  I tried the identified predictors in a new model.  And viewed its results in training data.


```r
model <- train( classe ~ roll_belt + yaw_belt + magnet_dumbbell_z + magnet_dumbbell_y + pitch_belt + pitch_forearm + accel_dumbbell_y + roll_forearm + magnet_forearm_z + roll_dumbbell + accel_dumbbell_z + roll_arm  , data=train, method='rf',trControl=trainControl( method='oob'), tuneGrid=expand.grid( mtry=3));
model;
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Out of Bag Resampling 
## 
## Summary of sample sizes:  
## 
## Resampling results
## 
##   Accuracy  Kappa
##   1         1    
## 
## Tuning parameter 'mtry' was held constant at a value of 3
## 
```
I iterated the steps above a few times trying smaller and larges 'sizes' in `rfe()` to get the best fit in the training data.  

The ultimate result was 12 variables that got an out-of-sample accuracy on the validation set of 99%.


```r
confusionMatrix( predict( model, validate ), validate$classe );
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1669    9    0    0    0
##          B    4 1122    5    0    6
##          C    0    7 1017    7    2
##          D    0    1    4  955    4
##          E    1    0    0    2 1070
## 
## Overall Statistics
##                                         
##                Accuracy : 0.991         
##                  95% CI : (0.988, 0.993)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.989         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.997    0.985    0.991    0.991    0.989
## Specificity             0.998    0.997    0.997    0.998    0.999
## Pos Pred Value          0.995    0.987    0.985    0.991    0.997
## Neg Pred Value          0.999    0.996    0.998    0.998    0.998
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.191    0.173    0.162    0.182
## Detection Prevalence    0.285    0.193    0.176    0.164    0.182
## Balanced Accuracy       0.997    0.991    0.994    0.994    0.994
```
One choice made during this analysis was to focus on 'random forest' models.  They ran faster than than other models and achieved superior fit.  Other models were tried in the `rfe()` step and subsequent model fit.

I also used 'oob' ( out of bag ) as the error measure in random forest models.  It gave a slightly better fit than other methods like 'cv' and 'LOOCV'.  It also ran faster which was most noticeable when working with all 52 variables.
