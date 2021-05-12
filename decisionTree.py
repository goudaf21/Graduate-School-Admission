"""
Assignment: #4
Group: Joseph Salerno and Leslie Le
decisionTree.py


"""

from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
# Be sure to plot the tree and explain how the features are used


# Read the data from the csv, remove the extra spaces from headers, and specify cols to be dropped
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
X_train, X_test, y_train, y_test= train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=1)


print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))





print("\t=======Loading Data=======\n\t  Please wait one second\n")

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

    # highlight test examples
    if test_idx:
        # plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='none',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='test set')
        


# ## Building a decision tree



X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))


tree_model = DecisionTreeRegressor(criterion='mse', 
                                    max_depth=5, 
                                    random_state=17)
tree_model.fit(X_train, y_train)

prediction = tree_model.predict(X)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

r2 = r2_score(y, prediction)

#print(classification_report(y,tree_model.predict(X)))

print("Final Loss: ",mean_squared_error(y, prediction))
print("Accuracy (r2 score): ",r2)


#plot_decision_regions(X_combined, y_combined, 
                  #classifier=tree_model, test_idx[267, 356])

dot_data = export_graphviz(tree_model,
                           filled=True, 
                           rounded=True,
                           feature_names=['TOEFL Score', 'GRE Score', 'CGPA', 'LOR',
        'University Rating'],
                           out_file=None) 
tree.plot_tree(tree_model)
#plt.savefig('Machine Learning')
plt.show()
graph = graph_from_dot_data(dot_data) 
graph.write_png('tree.png') 

#graph.write_png('tree.png') 


