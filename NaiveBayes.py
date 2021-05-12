import numpy as np 
import pandas as pd 
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

data=pd.read_csv('Admission_predict.csv')
data.rename(columns = {'LOR ':'LOR', 'Chance of Admit ':'Chance of Admit'}, inplace=True)

# Drop all data rows that contain a blank (NaN) value
data = data.dropna()


#make race into one hot encoding

data = pd.get_dummies(data = data, columns = ['Race'])

          
drop_cols = ['Serial No.', 'Chance of Admit']

cols = ['TOEFL Score', 'GRE Score', 'CGPA', 'LOR',
        'University Rating']

# Split the data and drop the specific cols from drop_cols
X =  data[cols]
y =  data['Chance of Admit']
naiveBayesModel = GaussianNB()

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)

# Test options and evaluation metric
scoring = 'accuracy'

#Fitting the training set
y_train_label = [1 if each > 0.8 else 0 for each in y_train]
y_test_label  = [1 if each > 0.8 else 0 for each in y_test]

naiveBayesModel.fit(X_train, y_train_label) 

#Predicting for the Test Set
pred_naiveBayesModel = naiveBayesModel.predict(X_test)

#Prediction Probability
prob_pos_naiveBayesModel = naiveBayesModel.predict_proba(X_test)[:, 1]

#Model Performance
#setting performance parameters
kfold = model_selection.KFold(n_splits=10)

#calling the cross validation function
cv_results = model_selection.cross_val_score(GaussianNB(), X_train, y_train_label, cv=kfold, scoring=scoring)

#displaying the mean and standard deviation of the prediction
print("%s: %f %s: (%f)" % ('Naive Bayes accuracy', cv_results.mean(), '\nNaive Bayes StdDev', cv_results.std()))

