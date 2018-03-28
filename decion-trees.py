# Kyle Ziegler 
# ML Practice
# Python 3.x
# Description: Decision trees are a popular type of algorithm used in machine 
# learning.

## INCOMPLETE ##

import numpy as np 
import pandas 
from sklearn.tree import DecisionTreeClassifier

dataSet = pandas.read_csv("./Datasets/skulls.csv",delimiter = ",")
dataSet[0:5]

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






