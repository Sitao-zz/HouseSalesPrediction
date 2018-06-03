# HouseSalesPrediction

The dataset that is selected for Regression Analysis is the residential price in Ames, Iowa. The source of raw data is from Kaggle website (https://www.kaggle.com/c/house-prices-advanced-regression-techniques).
This data source records 79 explanatory variables describing most of aspects of residential houses. The website provides two sets of data:
-	train.csv: 1460 records residential houses’ attributes with actual sale prices
-	test.csv: 1459 records residential houses’ attributes without sale prices
-	sample_submssion: the samples result (sale prices) for the test data is provided by the host

For this exercise, we treat the sample_submission as the actual sale prices and combined the train.csv with test.csv to form the data source for this exercise. Training set and validation set will be extracted out form this combined data source randomly with 3:1 ratio, i.e.:
-	training set: 2189 records
-	validation set: 730 records

## Performance metrics
RMSE and R2 are used as the performance measurement for the regression analysis. Both metrics are from sklearn.metric class.
-	RMSE: sklearn.metric.mean_squared_error
-	R2: sklearn.metric.r2_score.
Unlike the normal R-Squared metric that defines the percentage of explained data and has the range between 0 and 1, R2 score which is provided by sklearn is derived from RMSE and it may have negative results.
The horizontal benchmark line passing through the average of the target values will have the R2 score of 0. If the prediction has a smaller RMSE than this horizontal line, its R2 score will be positive and otherwise negative.

## GRNN Implementation
GRNN uses lazy learning approach for training. Rather than data learning, it just stores training data and uses it for later predictions.
During the prediction, the pattern that is closer to training data will get a higher value while the rest will get relative small values. This is controlled by the RBF, and the sigma parameter is used to adjust the RBF function output. The smaller the sigma value is, the smaller the radius effect of a particular input will be.

### GRNN summary
Advantage:
-	Single pass algorithm, no back propagation is required.
-	High accuracy in estimation, especially when the test data is similar to the training data.
Disadvantage:
-	If the size of the sample becomes very voluminous, the prediction will tend to be more time-consuming.

## MLP Regressor Implementation
MLP Regressor has similar structure as the MLP Classifier, but with following differences:
-	Output layer structure: the output layer only contains one neuron since the regressor is required to predict on the target value.
-	The choice of activation function: linear activation function is preferred for regressor to reflect the actual output from previous layers. If the output data is not normalized or having large skewness, data preprocessing shall be done before the training.
-	Cost function: the quadratic cost functions are usually used for regressor (e.g. RMSE, R2).

### MLP Regressor summary
Advantage:
-	Pre-trained algorithm: minimize the time required for prediction. The algorithm separates the development and its environment. And in the meanwhile, it streamlines the process.
-	High flexibility: allows users to construct large number of models for comparison and select the best one for deployment. E.g. the selected models are all better than the optimal GRNN configuration.
Disadvantage:
-	Difficult for user to optimize: the performance of a certain MLP configuration varies in different training sessions, and the relationship between the MLP configuration and the performance is unclear

## NN Ensembles
Two different Fusion functions are chosen for ensemble
-	mean value
-	median value
Two groups of child models are selected based on their individual RMSE scores on validation dataset.
-	All models
-	Only MLPR models
