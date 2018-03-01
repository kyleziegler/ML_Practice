# Kyle Ziegler 
# ML Practice

from sklearn.datasets import load_digits
from sklearn import svm

# Digits is a dataset made up of 1797 8x8 images
digits = load_digits()

print(type(digits))

print(digits.data)
print(digits.DESCR) # Long description of the dataset
print(digits.target)
print(digits.target_names)

print(digits.data.shape)
print(digits.target.shape)

x = digits.data
y = digits.target

clf = svm.SVC(gamma = 0.001, C = 100)
clf.fit(x,y)

#Array dimension error - expected 2D array, got a 1D
#print('Prediction: '), clf.predict(digits.data[-1])
#print('Actual: '), y[-1]

