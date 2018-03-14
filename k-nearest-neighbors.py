# Kyle Ziegler 
# ML Practice
# Python 3.x
# Description: K nearest neighbors, supervised learning. A 
# value for K chooses the distance from a predicted point to
# determine its classification.

import numpy as np
import pandas 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split 
from sklearn import metrics
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 

dataSet = pandas.read_csv("./Datasets/skulls.csv",delimiter = ",")

def removeColumns(pandasArray, *column):
    return pandasArray.drop(pandasArray.columns[[column]], axis=1).values

def target(numpyArray, targetColumnIndex):
    target_dict = dict()
    target = list()
    count = -1
    for i in range(len(dataSet.values)):
        if dataSet.values[i][targetColumnIndex] not in target_dict:
            count += 1
            target_dict[dataSet.values[i][targetColumnIndex]] = count
        target.append(target_dict[dataSet.values[i][targetColumnIndex]])
    return np.asarray(target)

X = removeColumns(dataSet, 0, 1)
y = target(dataSet, 1)

X_trainSet, X_testSet, y_trainSet, y_testSet = train_test_split(X, y, test_size=.3, random_state=7)

print("X shape: {}".format(X_trainSet.shape))
print("Y shape: {}".format(y_trainSet.shape))

print("X shape: {}".format(X_testSet.shape))
print("Y shape: {}".format(y_testSet.shape))

neigh = KNeighborsClassifier(n_neighbors=1)
neigh23 = KNeighborsClassifier(n_neighbors=23)
neigh90 = KNeighborsClassifier(n_neighbors=90)

neigh.fit(X_trainSet,y_trainSet)
neigh23.fit(X_trainSet,y_trainSet)
neigh90.fit(X_trainSet,y_trainSet)

pred = neigh.predict(X_testSet)
pred23 = neigh23.predict(X_testSet)
pred90 = neigh90.predict(X_testSet)

print("Accuracy: {}".format(metrics.accuracy_score(y_testSet,pred)))
print("Accuracy (neigh23): {}".format(metrics.accuracy_score(y_testSet,pred23)))
print("Accuracy (neigh90): {}".format(metrics.accuracy_score(y_testSet,pred90)))

# Training, testing, and spliting of the diabetes data set

diabetesDataSet = load_diabetes()
linearRegression = LinearRegression()

X = diabetesDataSet.data[:, None, 2]
y = diabetesDataSet.target

X_trainSet, X_testSet, y_trainSet, y_testSet = train_test_split(X, y, test_size=.3, random_state=7)

linearRegression.fit(X_trainSet, y_trainSet)

plt.scatter(X_testSet,y_testSet, color="black")
plt.plot(X_testSet, linearRegression.predict(X_testSet), color="red", linewidth=3)
# Display the plots on graph.  When block is on, code stops executing at this point until the 
# graph is closed.
plt.show(block=True)
