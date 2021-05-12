'''
Assignment: #4
Group: Joseph Salerno and Leslie Le
randomForest.py


Creates a random forest algorithmm to determine
if someone will develop diabetes.
Compare the results of the random forest with the decision tree

'''
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
from sklearn import tree
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

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
tree_model = DecisionTreeRegressor(criterion='mse', 
                                    max_depth=6, 
                                    random_state=10)
tree_model.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

# 1 3 8 79 198 253 819
forest = RandomForestRegressor(criterion='mse',
                                n_estimators=50, 
                                random_state=819,
                                n_jobs=2)
forest.fit(X_train, y_train)

#plot_decision_regions(X_combined, y_combined, 
                  #  classifier=forest, test_idx=range(267, 356))

prediction = forest.predict(X)


r2 = r2_score(y, prediction)

#print(classification_report(y,tree_model.predict(X)))

print("Final Loss: ",mean_squared_error(y, prediction))
print("Accuracy (r2 score): ",r2)
#plt.xlabel('Age')
#plt.ylabel('Pregnancies')
#plt.legend(loc='upper left')
#plt.tight_layout()
#plt.savefig('images/03_22.png', dpi=300)
#plt.show()
