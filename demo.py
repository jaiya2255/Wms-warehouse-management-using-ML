import pandas as pd
import os

import numpy as np

raw = pd.read_csv(r"C:\Users\NARAYANAN\.spyder-py3\RawData.csv")
raw.head()


from sklearn.preprocessing import LabelEncoder

labelencoder_X = LabelEncoder()

from sklearn.tree import DecisionTreeClassifier 

from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn import metrics

col_names = ['SNo', 'MaterialCode', 'Description', 'O&MStock', 'UoM', 'MatType', 'BinLocation', 'SLoc', 'TotalQuantity','MAPO&M']
raw = pd.read_csv(r"C:\Users\NARAYANAN\.spyder-py3\RawData.csv", header=None, names=col_names)

raw.head()


feature_cols = ['SNo', 'MaterialCode', 'Description', 'O&MStock', 'UoM', 'MatType', 'BinLocation', 'SLoc', 'TotalQuantity','MAPO&M']
X =raw[feature_cols] # Features
Y = raw.SLoc # Target variable

# Split dataset into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1) # 70% training and 30% test

raw[raw.isnull().any(axis=1)]

from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1) # 70% training and 30% test

Y = raw['SLoc']
X = raw.drop(['TotalQuantity'], axis=1)
X = pd.get_dummies(X)
Y = pd.get_dummies(Y)
X.info()
Y.info()

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.model_selection import GridSearchCV


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=2) # 70% training and 30% test

clf = DecisionTreeClassifier()

clf = clf.fit(X_train,Y_train)


Y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))

from sklearn.tree import export_graphviz

from sklearn.externals.six import StringIO  

from IPython.display import Image

import pydotplus

dot_data = StringIO()

from IPython.display import Image  

import pydotplus


dot_data = StringIO()

export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  



