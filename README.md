# Imbalanced-Classs-Problem
Imbalanced Class Problem
I started off with viewing variable distributions of each of the variable. Features 1, 2 11 and 12 were considered categorical while the rest were considered as continuous while features The categorical features were- feature_4, feature_5, feature_6, feature_7, feature_9, feature_10, feature_13, etc. Based on viewing the number of categories and distributions it was decided that the above classification of categorical and continuous variables would be followed. Correlation plots of the continuous variables were plotted. A correlation of 0.4 was observed between feature 11 and feature 12. The continuous variable dataset was standardized. The distribution of all the continuous variables war right skewed. 

 It was observed that the target positive values were roughly 8.5 percent of the total value. The logistic regression algorithm was tried, but it did not give good results. I read some reasons why logistic regression is not recommended for skewed dataset. Tree based methods are used to a greater extent. 
The dataset was split into training and testing set. 80% of the data was kept as training data and the remaining 20% was kept as testing data.

This is an imbalanced class problem. Effort was taken to balance the class by resampling higher amounts from from the sparse class. The library imblearn and SMOTE was used to do this.  The Oversampling was done such that the number sampled from the majority class was equal to its number in the dataset. The minority class was oversampled by 500% (5 times the number of 0’s in the dataset).  
SMOTE oversamples the minority class by creating synthetic examples rather than simple oversampling nearest neighbors is used for creating synthetic data. A line is drawn joining the elements of the minority class and if 200% sampling needs to be used, only two minority class neighbors from the five nearest minority class neighbors are chosen and one sample is generated in the direction of each. Synthetic samples are generated in the following way: Take the difference between the feature vector (sample) under consideration and its nearest neighbor. Multiply this difference by a random number between 0 and 1, and add it to the feature vector under consideration.


SMOTE Under sampling vs SMOTE oversampling
The majority class is under-sampled by randomly removing samples from the majority class population until the minority class becomes some specified percentage of the majority class.
SVM and Random Forest Algorithms were applied to the resampled data. A weight of 1:8 for (0:1) was used in the RF algorithm. This was done to offset the ratio imbalance in the training  dataset. Which was 8:1. Also, a classification threshold probability of 0.11 was applied here. 0.11 is the original N(1’s)/N(0’s) in the original dataset. SVM was applied with both rbf and linear kernels. Both these gave a similar F1 scores and AUC values.
K fold cross-validation was applied on the training dataset. I attempted to implement a gaussian kernel but that did not give as good results as rbf
K fold cross validation was applied with k =5 on the training set. The training dataset was divided into 5 folds. The model was trained and validated. The validation accuracy was over 90% for almost all models.  The accuracy on test set, the un augmented dataset was around 49%.

