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
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from mlxtend.plotting import heatmap
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.svm import SVC
from sklearn.svm import SVR

data=pd.read_csv('Admission_predict.csv')
data.rename(columns = {'LOR ':'LOR', 'Chance of Admit ':'Chance of Admit'}, inplace=True)
drop_cols = ['Serial No.', 'SES Percentage','Chance of Admit', 'Race']
data = data.dropna()

X_train, X_test, y_train, y_test= train_test_split(
    data.drop(drop_cols,axis=1),
    data['Chance of Admit'],
    test_size=0.3,
    random_state=6)
X_train_array=X_train.to_numpy()
X_test_array=X_test.to_numpy()

sc=StandardScaler()
sc.fit(X_train_array[:,:-1])
X_train_std=sc.transform(X_train_array[:,:-1])
X_test_std=sc.transform(X_test_array[:,:-1])
X_train_final=np.concatenate((X_train_std,X_train_array[:,-1:]),axis=1)
X_test_final=np.concatenate((X_test_std,X_test_array[:,-1:]),axis=1)
print(X_train_std.size)
print(X_train_final.size)
# cols=['GRE Score',
#       'TOEFL Score',
#       'University Rating',
#       'SOP',
#       'LOR',
#       'CGPA',
#       'Research',
#       'Chance of Admit']



# cm = np.corrcoef(data[cols].values.T)
# hm = heatmap(cm, row_names=cols, column_names=cols)

# plt.show()


# KNN=KNeighborsClassifier(n_neighbors=11,weights='distance',algorithm='auto',leaf_size=10,p=1,metric="minkowski",metric_params=None,n_jobs=None)
# KNN.fit(X_train_final,y_train)
# prediction = KNN.predict(X_test_final)
# def accuracy(y_values,prediction):
#     comparison = tf.equal(prediction, y_values)
#     acc = tf.reduce_mean(tf.cast(comparison, tf.float32))
#     return acc.numpy()


KNN_grid = [
 {'n_neighbors': [1,3,5,7,9,11],'metric':['chebyshev','minkowski','seuclidian'] ,'weights': ['distance', 'uniform'],'algorithm':['auto','ball_tree','kd_tree','brute'],'leaf_size':[10,20,30],'p':[1,2]}
 ]

# Instantiate the grid search with the parameter grid
KNN=GridSearchCV(KNeighborsRegressor(),KNN_grid,verbose=0,scoring='r2')

# Get the KNN with the grid search optimized hyperparameters 
KNN.fit(X_train_final,y_train)

# Feed the scaled test data into the grid search optimized KNN 
pred=KNN.predict(X_test_final)


# Print the accuracies for both of the KNNs
print("KNN r2 score:",r2_score(y_test, pred))

# SVM_grid=[{
#     'C':[1,2,3,4,5,6,7,8,9,10], 'kernel': ['poly','linear','rbf'],'degree':[1,2],'gamma':[1,10,100,0.001,0.01,0.1]
# }]

# SVM= GridSearchCV(SVR(),SVM_grid,verbose=0,scoring='r2')
# SVM.fit(X_train_final,y_train)
# SVM_pred=SVM.predict(X_test_final)
# print("SVM r2 score", r2_score(y_test,SVM_pred))


svc=SVR(C=100000,kernel='rbf',degree=3,gamma=0.000001)
svc.fit(X_train_final,y_train)
svc_pred=svc.predict(X_test_final)
print("SVM r2 score:" ,r2_score(y_test,svc_pred))
