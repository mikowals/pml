# load libraries and data
library( caret );
library( DoMC );
registerDoMC( cores = 2 );
projectTrain <- read.csv( '~/downloads/pml-train.csv' );

#split data into training and test
inTrain <- createDataPartition( projectTrain$classe, p=0.7, list=FALSE);
train <- projectTrain[ inTrain,];
validate <- projectTrain[ -inTrain,];

#build a model and review fit
simpleModel <- train( classe ~ user_name + num_window, data=train, method='rf', trControl=trainControl( method='oob' ), tuneGrid = expand.grid( mtry = 6 ));
simpleModel;
confusionMatrix( predict( simpleModel, validate ), validate$classe );

#create answers
test <- read.csv( '~/downloads/pml-test.csv' );
answers <- predict( simpleModel, test );

#modify trainset for more real world challenge
train <- train [, colSums( is.na( train ) == 0];
train <- train [, colSums(  train == '' ) == 0];
train <- train [, 7:59];

#identify important predictors
ctrl <- rfeControl( functions=rfFuncs, method='oob', index=TRUE );
rfProfile <- rfe( train[,1:52], train[,53], sizes=12 , rfeControl=ctrl );
rfProfile;
predictors( rfProfile );

#build a new model and review fit
model <- train( classe~roll_belt + yaw_belt + magnet_dumbbell_z + magnet_dumbbell_y + pitch_belt + pitch_forearm + accel_dumbbell_y + roll_forearm + magnet_forearm_z + roll_dumbbell + accel_dumbbell_z + roll_arm  , data=train, method='rf',trControl=trainControl( method='oob'), tuneGrid=expand.grid( mtry=3));
model;
confusionMatrix( predict( model, validate ), validate$classe );
