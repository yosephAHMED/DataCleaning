# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 00:53:07 2020

@authors: 
    Ahmed, Yoseph
    Giasi, Nicholas
    Fernando, Nattandige
    
Project:
    Using a dataset from the UCI Machine Learning Repository on whether a patient has heart disease.  
    You will clean the data and then apply a decision tree to the data.
    
Github repo:
    https://github.com/yosephAHMED/DataCleaning
"""

import pandas as pd

# Import the modules needed
import numpy as np # data manipulation
import matplotlib.pyplot as plt # matplotlib is for drawing graphs
import matplotlib.colors as colors
from sklearn.utils import resample # downsample the dataset
from sklearn.model_selection import train_test_split # split data into training and testing sets
from sklearn.preprocessing import scale # scale and center data
from sklearn.svm import SVC # this will make a support vector machine for classificaiton
from sklearn.model_selection import GridSearchCV # this will do cross validation
from sklearn.metrics import confusion_matrix # this creates a confusion matrix
from sklearn.metrics import plot_confusion_matrix # draws a confusion matrix
from sklearn.decomposition import PCA # to perform PCA to plot the data
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import tree

# 1. set up the panda data frame
data = pd.read_csv("processed.cleveland.data", header=1)

# show all columns so that none are abbreviated
pd.set_option('display.max_columns', None)

# 2. print the data before updating the columns
print(data.head(), "\n")

# 3. change the numbers to actual column names in the data
data.columns = ['age',
              'sex',
              'cp',
              'restbp',
              'chol',
              'fbs',
              'restecg',
              'thalach',
              'exang',
              'oldpeak',
              'slope',
              'ca',
              'thal',
              'hd']

# print the data after updating the columns
print(data.head(), "\n")

# 4. determine the datatype of each column
print(data.dtypes, "\n")

# 5. print out unique values for each float64 column
column_values = data[["age", "sex", "cp", "restbp", "chol", "fbs",
                      "restecg", "thalach", "exang", "oldpeak", "slope"]].values.ravel()
unique_values =  pd.unique(column_values)
print("float64 columns: ", unique_values, "\n")

# print out unique values for ca, thal, and hd columns
print("ca column unique vals: ", data['ca'].unique())
print("thal column unique vals: ", data['thal'].unique())
print("hd column unique vals: ", data['hd'].unique(), "\n")

# 8. we see that ca and thal columns contain values that are question marks
# we can determine the number of rows that have question marks via
print("num rows missing data: ", len(data.loc[(data['ca'] == '?') | (data['thal'] == '?')]), "\n")
# output: 6 rows of missing data

# observing the rows that are missing data values
print("rows missing data\n", data.loc[(data['ca'] == '?') | (data['thal'] == '?')])

# 10. count the number of rows in the full dataset
print("\nnumber of rows in full dataset: ", len(data))

# 11. remove the rows with missing values
data_no_missing = data.loc[(data['ca'] != '?') & (data['thal'] != '?')]

# 12. verify that you removed the rows by printing the size of the dataset
print("\nnumber of rows in new dataset: ", len(data_no_missing))

# 13. verify using the unique function that "ca" and "thal" do not have missing values
print("\nunique values for ca\n")
print(data_no_missing['ca'].unique())
print("\nunique values for thal\n")
print(data_no_missing['thal'].unique())

# 14. split the data into dependent and independent variables
# we want to predicty if a patient has heart disease or not, so 'hd' is the data (our y) that we want to predict
# therefore we drop 'hd' from the classifications (our X values)
X = data_no_missing.drop('hd', axis=1).copy() 

# 'hd' is what we want to predict (the y)
y = data_no_missing['hd'].copy()

# 15. print X and y dataframes
print("\nX.head()\n", X.head())
print("\ny.head()\n", y.head())

# 16. verify categorical data
print("sex column unique vals: ", data_no_missing['sex'].unique())
print("cp column unique vals: ", data_no_missing['cp'].unique())
print("fbs column unique vals: ", data_no_missing['fbs'].unique())

print("restecg column unique vals: ", data_no_missing['restecg'].unique())

print("exang column unique vals: ", data_no_missing['exang'].unique())
print("slope column unique vals: ", data_no_missing['slope'].unique())
print("thal column unique vals: ", data_no_missing['thal'].unique())
# output for restecg was 2, 0, 1 but we are supposed to have 1, 2, or 3
# restecg is the only column that needs to be corrected

X_encoded = pd.get_dummies(X, columns=['cp', 'restecg', 'slope', 'thal'])
print("\nX_encoded.head():\n", X_encoded.head())

# 17. making y value 0 or 1
print("\ny.unique(): ", y.unique())
y_not_zero_idx = y > 0
y[y_not_zero_idx] = 1
print("\ny.unique(): ", y.unique())

# 18. decision tree
feature_cols = list(X_encoded)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=1)
# above yields 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion = "entropy", max_depth = 3)
#clf = DecisionTreeClassifier(criterion = "entropy")

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy
print ("\n------------------------------------------------------------\n")
print("Model Accuracy\n")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print ("Precision ", metrics.precision_score(y_test, y_pred))
print ("Recall ", metrics.recall_score(y_test, y_pred ))
print ("f1 score ", metrics.f1_score(y_test, y_pred ))
print("roc-auc score:", metrics.roc_auc_score(y_test, y_pred))

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf,
                   feature_names=feature_cols, 
                   class_names=['0','1'],
                   #class_names=datatext.target_names,
                   filled=True)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=3, random_state=0)
clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print ("\n------------------------------------------------------------\n")
print("Predicted response for test dataset\n")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print ("Precision ", metrics.precision_score(y_test, y_pred))
print ("Recall ", metrics.recall_score(y_test, y_pred ))
print ("f1 score ", metrics.f1_score(y_test, y_pred ))
print("roc-auc score:", metrics.roc_auc_score(y_test, y_pred))

# The tree has four levels including the root node
# For the tree to come to an accurate decision whether or not
# the patient has heart disease, we need to traverse
# a total of 4 levels in the tree. If we don't include the root, 
# the answer is 3

