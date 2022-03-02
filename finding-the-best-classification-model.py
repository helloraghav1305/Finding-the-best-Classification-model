# Introduction

# In this project we will be using 'Stroke Prediction Dataset' from kaggle
# We will test various classification models to find out 
# which of them shows the best accuracy.

# Importing basic libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset

df = pd.read_csv('../input/stroke-prediction-dataset/healthcare-dataset-stroke-data.csv');

# This shows the first five rows for all the features of the dataset

df.head()

# We want to keep all the categorical features (like 'gender', 'ever_married' etc)
# in the beginning 

df = df[['id', 'gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status',
         'hypertension', 'stroke', 'age', 'avg_glucose_level', 'bmi', 'heart_disease']]

# This shows the feature 'heart_disease' consists of two values 1 and 0

print(df['heart_disease'].unique())

# We neglect the 'id' feature
# 'X' consists of independent variables
# 'y' consists of dependent variable

X = df.iloc[:, 1:11].values
y = df.iloc[:, -1].values

# This shows a list of features that consist of null values

print(df.columns[df[df.columns].isnull().sum() != 0])

# Replacing the null value for that feature with 
# the mean of all other values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, [9]])
X[:, [9]] = imputer.transform(X[:, [9]])

# Encoding categorical data

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0, 1, 2, 3, 4])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print(X)

print(y)

# Splitting the dataset into training set and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

print(y_train)

print(y_test)

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 18:21] = sc.fit_transform(X_train[:, 18:21])
X_test[:, 18:21] = sc.transform(X_test[:, 18:21])

# Training the classifier with various models

# Logistic Regression

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# SVM

from sklearn.svm import SVC
classifier_svc = SVC(kernel = 'linear', random_state = 0)
classifier_svc.fit(X_train, y_train)

# Kernel SVM

classifier_rbf = SVC(kernel = 'rbf', random_state = 0)
classifier_rbf.fit(X_train, y_train)

# K Nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier_knn.fit(X_train, y_train)

# Decsion Tree

from sklearn.tree import DecisionTreeClassifier
classifier_dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_dt.fit(X_train, y_train)

# Finding the predictions of each model

y_pred = classifier.predict(X_test)

y_pred_svc = classifier_svc.predict(X_test)

y_pred_rbf = classifier_rbf.predict(X_test)

y_pred_knn = classifier_knn.predict(X_test)

y_pred_dt = classifier_dt.predict(X_test)

# Finding the accuracy of each model

from sklearn.metrics import accuracy_score

print("Logistic Regression - " , accuracy_score(y_test, y_pred))

print("SVM - " , accuracy_score(y_test, y_pred_svc))

print("Kernel SVM - ", accuracy_score(y_test, y_pred_rbf))

print("KNN - " , accuracy_score(y_test, y_pred_knn))

print("Decision Tree - ", accuracy_score(y_test, y_pred_dt))

# Conclusion

# The best accuracy turns out to be 94.7% shown
# by Logistic Regression, SVM, and Kernel SVM, followed by 
# KNN which shows an accuracy of 94.4%, and
# Decision Tree which shows an accuracy of 90.5%










