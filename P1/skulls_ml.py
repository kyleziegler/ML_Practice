# Kyle Ziegler 
# ML Practice
# Python 3.x
# Intro to Supervised ML and the skulls dataset.

import numpy as np
import pandas 
from sklearn.neighbors import KNeighborsClassifier

# Remove the column containing the target name since it doesn't contain numeric values.
# Also remove the column that contains the row number
# axis=1 means we are removing columns instead of rows.
# Function takes in a pandas array and column numbers and returns a numpy array without
# the stated columns
def removeColumns(pandasArray, *column):
    return pandasArray.drop(pandasArray.columns[[column]], axis=1).values

# 1st Input: A numpy array. The numpy array you will use is my_data.values (or my_data.as_matrix())
# 2nd Input: An integer value that represents the target column . (Look at the data again and find 
# which column contains the non-numeric values. This is the target column)

# Use this function to get the response vector y
dataSet = pandas.read_csv("skulls.csv",delimiter = ",")
def targetAndtargetNames(numpyArray, targetColumnIndex):
    target_dict = dict()
    target = list()
    target_names = list()
    count = -1
    for i in range(len(dataSet.values)):
        if dataSet.values[i][targetColumnIndex] not in target_dict:
            count += 1
            target_dict[dataSet.values[i][targetColumnIndex]] = count
        target.append(target_dict[dataSet.values[i][targetColumnIndex]])
    # Since a dictionary is not ordered, we need to order it and output it to a list so the
    # target names will match the target.
    for targetName in sorted(target_dict, key=target_dict.get):
        target_names.append(targetName)
    return np.asarray(target), target_names


workingSet = pandas.read_csv("skulls.csv",delimiter = ",")
#print(workingSet)
#print(type(workingSet))

columns = workingSet.columns
#print(columns)

values = workingSet.values
#print(values)

shape = workingSet.shape
#print(shape)

#Remove columns that are not needed in creating the machine learning problem.
workingSet = removeColumns(workingSet, 0, 1)
#print(workingSet)

target, targetNames = targetAndtargetNames(dataSet, 1)
#print(target,targetNames)

# We've already remove cols, working set is good to go.
X = workingSet
y = target

# Create a K-nearest neighbors model
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X,y)


# Expecting a 2D array here, added another series of brackets around the working set to 
# fix a bug.
print("Predtiction: ", neigh.predict([workingSet[10]]))
#print(workingSet)
print("Actual: " + np.array_str(y[10]))
