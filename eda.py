# Fady Gouda, Griffin Noe, Joe Salerno
# CSCI 297-a
# 10/20/20
# Test 1

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from mlxtend.plotting import heatmap
from sklearn.preprocessing import OneHotEncoder

# Read the data from the csv, remove the extra spaces from headers, and specify cols to be dropped
data=pd.read_csv('Admission_predict.csv')
data.rename(columns = {'LOR ':'LOR', 'Chance of Admit ':'Chance of Admit'}, inplace=True)

# Drop all data rows that contain a blank (NaN) value
data = data.dropna()


#make race into one hot encoding

data = pd.get_dummies(data = data, columns = ['Race'])

data.info()
          
drop_cols = ['Serial No.', 'Chance of Admit']

# Split the data and drop the specific cols from drop_cols
X_train, X_test, y_train, y_test= train_test_split(
    data.drop(drop_cols,axis=1),
    data['Chance of Admit'],
    test_size=0.3,
    random_state=1)

# Scale the data using standard scalar
sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)

# Create a list that contains the column names of the dataframe
cols=data.columns

# Get a correlation matrix and plot it as a heatmap
cm = np.corrcoef(data[cols].values.T)
hm = heatmap(cm, row_names=cols, column_names=cols)
plt.show()
