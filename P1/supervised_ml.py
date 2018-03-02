# Kyle Ziegler 
# ML Practice
# Python 3.x
# Supervised learning

import numpy as np
import pandas 
from sklearn.neighbors import KNeighborsClassifier

dataSet = pandas.read_csv("skulls.csv",delimiter = ",")

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

neigh1 = KNeighborsClassifier(n_neighbors=1)
neigh7 = KNeighborsClassifier(n_neighbors=7)

neigh1.fit(X,y)
neigh7.fit(X,y)

print("Predtiction: ", neigh1.predict([X[30]]))
print("Predtiction: ", neigh7.predict([X[30]]))

