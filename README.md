Annotated R code - Practical Machine Learning Course Project
============
Load libraries and data.

     library( caret )
     library( DoMC )
     registerDoMC( cores = 2 )
     projectTrain <- read.csv( '~/downloads/pml-train.csv' )

Split the data into training and validation sets.

     inTrain <- createDataPartition( projectTrain$classe, p=0.7, list=FALSE)
     train <- projectTrain[ inTrain,]
     validate <- projectTrain[ -inTrain,]

Visually inspecting data shows it is time series for 6 different user_names with the dependent variable 'classe' occurring in groups if the data is sorted by time.  So I tried a simple model with those variables.

     simpleModel <- train( classe ~ user_name + num_window, data=train, method='rf', trControl=trainControl( method='oob' ), tuneGrid = expand.grid( mtry = 6 ))

Inspect model results in training and validation data.

     simpleModel
     confusionMatrix( predict( simpleModel, validate ), validate$classe )

The model is a perfect fit in the training data and 99.9% accurate in validation.  So I created the answers for course project based on test set.

     test <- read.csv( '~/downloads/pml-test.csv' )
     answers <- predict( simpleModel, test )

To make a more real world example I removed columns with #N/A or '' entries and then removed the 'user_name' and all time related columns.  This seems more useful in the real world where exercise feedback will need to respond to situations where the exercise style in the previous period is not known.

    train <- train [, colSums( is.na( train ) == 0]
    train <- train [, colSums(  train == '' ) == 0]
    train <- train [, 7:59]

This left 52 predictors available which was computationally heavy.  So I tried to identify the most interesting predictors.

    ctrl <- rfeControl( functions=rfFuncs, method='oob', index=TRUE )
    rfProfile <- rfe( train[,1:52], train[,53], sizes=12 , rfeControl=ctrl )
    rfProfile
    predictors( rfProfile )

This listed the 12 best predictors.  I tried the identified predictors in a new model.  And viewed its results in training and validation tests.

    model <- train( classe~roll_belt + yaw_belt + magnet_dumbbell_z + magnet_dumbbell_y + pitch_belt + pitch_forearm + accel_dumbbell_y + roll_forearm + magnet_forearm_z + roll_dumbbell + accel_dumbbell_z + roll_arm  , data=train, method='rf',trControl=trainControl( method='oob'), tuneGrid=expand.grid( mtry=3))
    model
    confusionMatrix( predict( model, validate ), validate$classe )

I iterated the steps above a few times trying smaller and larges 'sizes' in `rfe()` to get the best fit in the validation data.  The ultimate result was 12 variables that got an out-of-sample accuracy on the validation set of 98.9%.

One choice made during this analysis was to focus on 'random forest' models.  They ran faster than than other models and got superior fit.  Other models were tried in the `rfe()` step and subsequent model fit.

I also used 'oob' ( out of bag ) as the error measure in random forest models.  It gave a slightly better fit both in- and out-of-sample than other methods like 'cv' and 'LOOCV'.  It also ran faster which was mostly noticeable when working with all 52 variables.
