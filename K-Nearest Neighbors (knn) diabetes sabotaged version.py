#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 12:52:57 2023

@author: quintenachterberg
"""
#--------------SABOTAGED VERSION------------------

#Import the relevant packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#Data import
rawDF = pd.read_csv('https://raw.githubusercontent.com/HAN-M3DM-Data-Mining/assignments/master/datasets/KNN-diabetes.csv')
rawDF.info()

#------------Data Prep--------------

# Display amount of diagnosis' -> 0 is not diabetes, 1 is diabetes
cntDiag = RawDF['Outcome'].value_counts()


# Percentage/ normalize
propDiag = rawDF['Outcome'].value_counts(normalize=True)
cntDiag


# Looking closer to four differnet characteristics, it can be seen tha Glucose has a much higher range compared to the rest
rawDF[['Glucose', 'BloodPressure', 'Insulin', 'BMI']].show()


# Writing the normalized function
def normalize(x): return((x - min(x)) / (max(x) - min(x))) # Distance of item value - minimum vector value divided by the range of all vector values

testSet1 = np.arange(1,6)
testSet2 = np.arange(1,6) * 10


print(f'testSet1: {testSet1}\n')
print(f'testSet2: {testSet2}\n')
print(f'Normalized testSet1: {normalize(testSet1)}\n')
print(f'Normalized testSet2: {normalize(testSet2)}\n')


# Applying the normalized function on the rawDF set, excluding the 'outcome' column 
excluded = ['Outcome'] # list of columns to exclude
X = rawDF.loc[:, ~propDiag.columns.isin(excluded)]
X = X.apply(normalize, axis=0)


X[['Glucose', 'BloodPressure', 'Insulin', 'BMI']].describe()
# After this, the above four variables have a max of 1


y = rawDF['Outcome'] # 'Outcome' is now our y-axis and the rest of the columns the x-axis
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)

#-----------Modelling and evaluation-------------

# Make predictions on the test set
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)

y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=knn.classes_)
cm

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
disp.plot()

plt.show()


#Accuracy = (all correct / all) = TP + TN / TP + TN + FP + FN
#(133 + 44) / 231 = 177 / 232 = 0.766 or 77% Accuracy

#Misclassification = (all incorrect / all) = FP + FN / TP + TN + FP + FN
#(17 + 37) / 231 = 54 / 232 = 0.232 or 23% Misclassification

