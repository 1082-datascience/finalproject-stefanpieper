# Software User Guide

## Content
This document contains an explaination of how to use the R script (found in the map code). First, the general method is outlined. Next, the data exploration is explained. Thereafter, the code regarding the rotation forest model is explained, including five function (based on the R package rotationForest). Lastly, the Random Forest, glmnet and zeros models are explained.

## 1. General Method (Cross Validation)
The method used for evaluating the performance of the models is 5-fold cross validation. The training data is split into five folds. Every fold is used as a validation set once, while the other four folds are used as the training data. To examine the performance of the best model determined by which model has the lowest average accuracy, the test data from kaggle is used. A prediction is made and uploaded to kaggle in order to determine the performance of the models. The first part of the model reads the data. 


### 2 Corrplot
This sections contains the corrplot2 function, which is a wrapper function of the corrplot function of the corrplot package. It is used here because it is easier to get the right layout for the correlation plot. It can only be run manually. 


## 3. Rotation Forest
The entire rotation forest part of the script consists of five functions and a few lines that call these functions to generate and save and output in csv. For more detailed explainations of how the Rotation Model works, please refer to the power point or pdf file in the doc foler.

### 3.1 rotationForest(x, y, K=round(ncol(x)/3, 0), ntree=10, verbose=FALSE, ...)
This function fits a rotation forest model on the given datasets x and y, using the parameters K and ntree. The output is a list with models, loadings and columnnames. The PCA analysis of each subset is performed using the prcomp function of the stats package. The decision trees are used using the rpart function of the rpart package. The output is a csv file with the prediction, a csv file with the performance (accuracys), and it plots the average accuracys of both training and test set.
 Note that these lines either use the demo, short and long version. The short version only uses two different values for both K and ntree, so that it runs fast. The long version uses a lot of combinations of the two, which makes it take over a day to run. The demo version uses the results of the long version that were produced earlier. 

#### Variables
 * x: independent variables
 * y: response variable (binary)
 * K: amount of subsets of features
 * ntree: amount of decision trees for each subset
 * verbose: if TRUE, progress is printed

#### Output
 models: each individual rpart model trained on each subset
 loadings: the PCA loadings of each subset
 columnnames: the features in x
 
### 3.2 predict.rotationForest(models, newdata, type=c("prob", "class"), all=FALSE, round.results=TRUE)
This function uses the models of the rotationForest function to predict the newdata. Some of the lines are not necessary for this research, but might be needed for later development of the function (this includes the type and all variables).

#### Variables
 * models: output of the randomForest function
 * newdata: data with the same features as used in the randomForest function
 * type: not relevant
 * all: not relevant
 * round.results: if TRUE, binary prediction is given. If FALSE, the probability of pred=1 is given.

#### Output
 vector with predictions
 

### 3.3 cv.rotationForest(x, y, x_test=NULL, y_test=NULL, nfolds=5, type.measure=c("accuracy"), round.results=FALSE, K=round(ncol(x)/3, 0), ntree=10, verbose=FALSE, test=FALSE, ...)
This function performs a k-fold cross validation of the rotationForest model, given the amount of subsets (K) and amount of trees (ntree)

#### Variables
 * x, y, K, ntree and verbose are the same as in the randomForest function (3.1)
 * x_test, y_test: if given, a prediction is made on x_test and its performance is evaluated on y_test
 * nfolds: amount of folds for the k-fold cross validation
 * type.measure: measure to select the best model with
 * round.results: if TRUE, the accuracys in the results are round to two decimal numbers
 * test: if TRUE, the k-fold cross validation splits the data in a training, validation and test set for each fold (instead of only training and validation)
 
#### Output
 * pred.test: prediction on test set if x_test is not NULL or if test is TRUE
 * models: models fitted on each fold
 * performance: data.frame containing the accuracy of the training and validation sets of each fold and the average accuracy over all folds
 
 ### 3.4 vary.rotationForest(x, y, x_test=NULL, y_test=NULL, plot.results=TRUE, nfolds=5, type.measure=c("accuracy"), round.results=FALSE, K=c(round(ncol(x)/3, 0), round(ncol(x)/6, 0), round(ncol(x)/9, 0)), ntree=c(50, 100), verbose=FALSE, test=FALSE, save.file=NULL, ...
 Wrapper function of cv.rotationForest (see 3.3). It performs the cross validation function for each combination of the specified amount of subsets (K) and amount of trees (ntree).
 
 #### Variables
  * x, y, x_test, y_test, nfolds, round.results, type.measure, verbose, test: are the same as for the cv.rotationForest function (see 3.3)
  * K: a vector of values of K to vary in for the cross validation
  * ntree: a vector of values of ntree to vary in for the cross validation
  * plot.results: plots using plot.rotationForest
  * save.file: not relevant
 
 #### Output
 prediction: 
 nfolds: amount of folds in the k-fold cross validation
 best.K: K of the model with the lowest average validation accuracy
 best.ntree: ntree of the model with the lowest average validation accuracy
 best.model: best model (class rotationForest)
 summary: summary of all models
 models: each output of cv.rotationForest
 
 ### 3.5 plot.RotationForest(model, pos="topright", col=NULL)
 Plots the two line graphs for the train and validation set: for with K on the x-axis, average accuracy on the y-axis and a line for each value of ntree.
 
 #### Variables
  * model: output of vary.rotationForest function (see 3.4)
  * pos: position of the legend
  * col: vector of colors for the line
  
  
 ## 4. Random Forest 
 The random forest part of the script selects the best model out of several candidate models (by varying the ntree and replace parameters of the randomForest function of the randomForest package). The vary.randomForest function is a wrapper function that does a similar thing as the vary.rotationForest function (3.4), but it uses the randomForest models with varying parameters (ntree and replace) indstead. It outputs a csv file with the prediction.
 
 
 ## 5. LASSO 
 The LASSO part of the script uses the cv.glmnet function from the glmnet package to determine the best lambda to use in the LASSO model, and output a csv file with a prediction.
 
 
 ## 6. Zero's
 This part of the script just produces a csv prediction file with only zero's, to see if the other models (3, 4 or 5) outperform this NULL-model.

