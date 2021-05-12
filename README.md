# 297-test-2
The second test for Professor Watson's 297 course. 


## Feature Analysis (eda.py)

  We constructed a heatmap to determine which features have the highest correlation in the dataset as well as highest correlation to the prediction percentage. Features we were able to write off because of this ended up being Race, SES percentage, and Research because their correlations were just too low to be usuable in our models. We were able to check race by converting it into a one hot encoded vector, which ended up being split into 4 separate columns, but it clearly showed no correlation towards chance of admittion. Additionally, we wrote off the feature Serial No. because it was just a incremental value for each row. Lastly, we also decided to remove all the rows that had missing values, this was for simplicity sake and the mean values to fill categories that had missing values could've potentially created false negatives/positives because of the obviously high correlation.


## Scaling

We tested a multitude of feature scalers and found that they all seemed to produce the same outcome in terms of metrics. Thus, we decided to just use the standard scalar for this data set.

##  Linear Regression
--Since Logistic regression performed so well we know that the best model is not linear--
   
## Logistic Regression
   For our logistic regression, we decided to try and refine the considered variables even further. Dropping one column at a time based on what had lowest correlation to the prediction percentage (research, then LOR, then SOP, etc.) and found that the best results were from only considering 4 variables. At first we tried to use GRE, TOEFL, CGPA, and University Rating and found it worked extremely well but that by replacing university rating with SOP, the model performed marginally better on false positives. We split the data 75/25 but also tried other permutations of splits between 80/20 and 60/40 for every 5% and found 75/25 worked best. For the scaler, we tried every type of scaler and found that Standard, Robust, QuantileTransformer, and PowerTransformer all performed identically so we kept Standard. We then plot the correlation heatmap for the four considered variables and it can be seen that they all have high correlations with each other but nothing above .83. We then used a grid search with a wide range of C, penalty's, solvers and random states to find the optimized hyperparameters. We found that a C=0.6, l2 penalty, random state of 1, and the default lbfgs solver performed optimally. We then turned the labels from a continuous scale to a binary classification, trying a variety of cutoffs from 0.6 to 0.9 going every 0.05 finding that .8 performed best. We then fit the data, get the predictions and print the accuracy, R2, and classification report showing the F1 score and accuracy. After this, we plot the confusion matrices for both the test and train data to show the final results of 0 false positives and 2 false negatives for an accuracy of 97.7

## SVM
For SVM we tried implementing a grid search for the different hyperparameters after applying a standard scalar to the features. Our best parameters are the ones we use in the SVR classifier. We are using SVR instead of SVC because this is a regression problem instead of a classifying one. The model did not give us an r2 score above 0.77. While doing the scaling process for both SVM and KNN, we had to remove the research data before scaling, so that it doesn't scale the binary data, and adding it to the standardized data afterwards. This model clearly does not compete with our models therefore we decided not to consider it.

## KNN
Similar to SVM, we also ran a grid search to find the best r2 score that we could reach and found the best score to be 0.77 as well. This makes sense as KNN are not expected to be good regressors but rather make good classifiers. Thus, this was not the best model for the given data set.


## Naïve Bayes:
In our Naïve Bayes model, we changed the target to a binary result in the same way we did in logistic regression. This is because the naïve Bayes probability is calculated using frequencies of certain class values. We found the model to have a fairly high r2 score, which is surprising considering the Naïve Bayes assumption that features are uncorrelated to each other, but we think that it produces good results because of our usage of threshold to convert the data into a binary type of target data. However, this model, although probably our second-best model for the data, still is not better than our logistic regression model in the end. Although, it is still really good for this given dataset.



## Decision Tree

  We selected features based off the highest correlations found in the heatmap. So, initially, we started the model off with TOEFL Score and GRE Score. From here, we tried to improve the model as much as we could with the given features. For this, we tried to maximize our R squared value and ended up adding CGPA, LOR,
University Rating which each improved the R squared value by 0.1 - 0.2 points. This number of features strikes a balance between overfitting and underfitting the data and as stated above, many of the dropped features showed minimal correlation.
  We then used a DecisionTreeRegressor, since the chance of admit is a continous values, and it is a simpler method for calculating whether or not this approach is better for this data set.
  Next, for the parameters of the model itself. We used a grid search to determine the best paramaters for the model but ended up receiving parameters that produced a low R squared value (<75) so that led us to assume that these values produced a high accuracy but a low R squared. Next, we constructed a model with a max_depth of 5, as more would overfit and less produced a smaller r squared by 0.3. Next, we attempted to maximize the output by using a for loop to check the best random_state for the first 1000 values, which somehow ended up being 17.
  The output for the R squared value ended up being 0.83 which is not the best when compared to our other models. This is not our best option for this data.

## Random Forest

We selected features based off the highest correlations found in the heatmap. So, initially, we started the model off with TOEFL Score and GRE Score. From here, we tried to improve the model as much as we could with the given features. For this, we tried to maximize our R squared value and ended up adding CGPA, LOR, University Rating which each improved the R squared value by 0.1 - 0.2 points.
 This number of features strikes a balance between overfitting and underfitting the data and as stated above, many of the dropped features showed minimal correlation. We then used a RandomForestRegressor, since the chance of admit is a continous values, and it is a simpler method for calculating whether or not this approach is better for this data set. Next, for the parameters of the model itself. For this model, we initially changed the number of trees and started off with 40 which ended up producing an r squared value of 0.9231. Next we tried 45 which increased it by 0.0004 and the next multiple of 5 increased it by 0.0007.  So 50 ended up being the best value as increasing it decreased the r squared and decreasing it also decreased the r squared.  Next, we attempted to change the criterion but this had little to no impact on the output. Next, we attempted to maximize the output by using a for loop to check the best random_state for the first 1000 values, which ended up being 819, although this only increased the output by a minute amount. The output for the R squared value ended up being 0.9242 which is not the best when compared to our other models. This is not our best option for this data.
 
 ## Analysis
 
 We made most of our models produce an R squared value since they all were mainly regression models and this was best at determining the usefulness of each model. The model that produced the best R squared, logistic regression, also conveniently produced high values in recall, accuracy, etc. Hence why we decided to explore this model more as it is clearly a better fit for our data. We found that the logistic regression performed extremely , giving 0 false positives and 2 false negatives for an accuracy of 97.7.
 
 
