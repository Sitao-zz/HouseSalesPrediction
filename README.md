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

## Understandings and Findings
### CNN vs MLPC
The layers perform the learning activities in the CNN are the FC (fully connected) layers, which is effectively the MLP model. However the CNN outperforms the simple MLPC by 5%, reaching 94% accuracy. This difference shows the importance of the data preprocessing.
MLP model has great potential as well as enormous flexibility. However great flexibility leads to great uncertainty and the MLP model can be easily distracted by irrelevant information during training. In the case of image recognition, the raw data capture all the details, but they may not be the determining factor of the entity. Thus adding Convolutional Layer and Pooling layer can greatly filter the input information, and only highlight the pixels as groups. Eventually these pixel groups are treated as features feeding the FC layer. From FC layer’s perspective, the input information is the important features after the filtering, and it has less chance going into wrong direction.

For CNN activation function, usually we choose RELU as it is fast. If the ReLU has a large negative bias, the gradient through it will consistently be 0 and no error signal can ever pass to earlier layers, then we should choose Leaky ReLU to avoid this problem.

### GRNN vs MLPR
GRNN and MLPR are having very different prediction approach. GRNN uses lazy-training and MLPR uses pre-training approach.
GRNN simply stores the input data for future prediction. This makes the model construction very easy. Moreover the parameter can be adjusted is only “std”, and the user can easily choose the value according to the domain knowledge, thus the performance of the model is fully predictable. This model is very useful for data exploration, where fairly good prediction is expected in short amount of time (the best GRNN has the RMSE on validation data of 0.246).

On the contrary, MLPR involve intensive trainings to get a good model. According to Section3.2.3 MLP Regressor and results, the best MLPR can have better prediction than GRNN (the best MLPR has the RMSE on validation data of 0.220), while the worse can far worse than GRNN (the worse MLPR has the RMSE on validation data of 0.427). However once the good MLPR model is developed, the prediction on testing data will be much faster than GRNN, since only the simple math calculations are involved. Thus MLPR is very useful for production model where the calculation performance is considered.

### MLP Classifier vs MLP regressor
MLP is a very flexible model. Models with different configurations are developed for different use cases.
The output data type:
-	Classifier’s output data type are categorical, thus multiple output nodes are required to ensure that no priority is applied between the output classes. 
-	Regressor’s output data is usually a single continuous value, thus output node is 1.
Activation function:
-	Classifiers are required to draw the clear boundaries, thus non-linear can be used to amplify or narrow the differences, e.g. “softmax” function.
-	Regressors are required to have accurate error feedback for backpropagation. To reduce the complexity of the model and increase the learning speed of the model, usually the linear activation functions are used.
Cost function:
-	Classifiers always use “cross-entropy” loss function with “softmax” function together. This normalizes the values given by MLPR into range of (0,1). So effectively the prediction value is always at one side of the true value, which is always below 1. This allows the loss function being simpler without worrying about the sign of the difference.
-	Regressors need to handle more complicated predictions, where predicted values can be higher or lower than the expected one. Thus quadratic cost function is required to prevent the offset between overestimation and underestimation.

### Ensemble performance
Ensemble works well when the individual models have their specialization, e.g. one model is good at classifying class A, and another model is good at classifying class B. Thus if one model is always worse than another one in all classes, ensemble will not generate better performance. In the classification exercise, the ensemble model is better than the MLPC, but worse than CNN. In the regression exercise, the All-in-Ensemble is worse that MLPR-only-Ensemble.

However the accuracy for the ensemble model is much higher than MLPC and CNN at early epochs. After the 1st Epoch the accuracy of each models are: MLPC: 0.5075, CNN: 0.8578, Ensemble: 0.9368. This is might be due to the different specializations of MLPC and CNN during the early trainings, where the CNN is not fully optimized. The ensemble combines the results from MLPC and CNN, compensates one’s weakness with the other’s strength and gives better performance.
