# Fady Gouda, Griffin Noe, Joe Salerno
# CSCI 297-a
# 10/20/20
# Test 2

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from mlxtend.plotting import heatmap
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import mean_squared_error


# Read the data from the csv, remove the extra spaces from headers, and specify cols to be dropped
data=pd.read_csv('Admission_predict.csv')
data.rename(columns = {'LOR ':'LOR', 'Chance of Admit ':'Chance of Admit'}, inplace=True)

# Drop all data rows that contain a blank (NaN) value
data = data.dropna()


# Make race into one hot encoding
data = pd.get_dummies(data = data, columns = ['Race'])

# Specify the columns to be dropped in the data leaving GRE, TOEFL, CGPA, and SOP
drop_cols = ['Serial No.', 'Chance of Admit', 
             'SES Percentage', 'LOR', 'Research',
             'University Rating', 'Race_Asian', 'Race_african american',
             'Race_latinx', 'Race_white']

# Split the data and drop the specific cols from drop_cols
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(drop_cols,axis=1),
    data['Chance of Admit'],
    test_size=0.25,
    random_state=1)

# Scale the data using standard scalar
sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)

# Create a list that contains the column names of the dataframe
cols=data.drop(drop_cols,axis=1).columns

# Get a correlation matrix and plot it as a heatmap
cm = np.corrcoef(data[cols].values.T)
hm = heatmap(cm, row_names=cols, column_names=cols)
plt.title("Correlation Heatmap of Considered Variables")
plt.show()

# param_grid = {'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 
#               'penalty': ['l2'],
#               'random_state': range(0,20)}

# lg = GridSearchCV(LogisticRegression(), param_grid)
# print(lg.best_params_)

# Instantiate the Logistic Regression with the optimized hyperparameters
lg = LogisticRegression(C=0.6, penalty='l2', random_state=1)

# Turn the labels into a binary classification based on a cutoff of 0.8
y_train_label = [1 if each > 0.8 else 0 for each in y_train]
y_test_label  = [1 if each > 0.8 else 0 for each in y_test]

# Fit the logistic regression to the standardized data and the training label
lg.fit(X_train_std, y_train_label)

# Create the predicted labels from the standardized test data
pred = lg.predict(X_test_std)

# Print the accuracy, R2, and classifiction report
print("Accuracy: %.2f" % (100*accuracy_score(y_test_label, pred)), "%")
print("R2 Score: %.6f" % r2_score(y_test_label, pred))
print(classification_report(y_test_label, pred))

# Plot the confusion matrix for the training and testing data
plot_confusion_matrix(lg, X_test_std, y_test_label)
plt.title("Test Data Confusion Matrix")
plot_confusion_matrix(lg, X_train_std, y_train_label)
plt.title("Train Data Confusion Matrix")
plt.show()