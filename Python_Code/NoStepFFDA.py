# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 23:48:28 2018

@author: NiTiN
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB 

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import style
style.use('ggplot')

import seaborn as sns


#raed CSV File
data = pd.read_csv("C:/Users/RetailAdmin/SpyderProgram/FFDA/PS_20174392719_1491204439457_log.csv")
data.head()
data.info()

#the correlation of the Feature
data.corr()

#the correlation of the Feature heatmap
sns.heatmap(data.corr())

data = data.drop(['step','nameOrig', 'nameDest', 'isFlaggedFraud'], axis = 1)
data.head()

data.describe()

Cols = data[['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']]
X = Cols
y = data['isFraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


#Random Forest
rfc = RandomForestClassifier()
model = rfc.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy_score(y_test, predictions)
print ("Random Forest Accuracy", accuracy_score(y_test,predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test,predictions))

#Logistic Regression
logistic = linear_model.LogisticRegression()
model = logistic.fit(X_train,y_train)
redictions = model.predict(X_test)
accuracy_score(y_test,predictions)
print ("Logistic Regrassion Accuracy", accuracy_score(y_test,predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test,predictions))

#Naive Bayes
gnb = GaussianNB()
model = gnb.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy_score(y_test,predictions)
print ("GaussianNB Accuracy", accuracy_score(y_test,predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test,predictions))


######################@@@@@@@@@@@@@@#######################


###MultinomialNB
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
model = mnb.fit(X_train, y_train)
accuracy_score(y_test,predictions)
print ("Multinomial Accuracy", accuracy_score(y_test,predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test,predictions))

###BernoulliNB
from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
model = bnb.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy_score(y_test,predictions)
print ("BernoulliNB Accuracy", accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test,predictions))

#############################

#DecisionTree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=10)
model = clf.fit(X_train, y_train)
accuracy_score(y_test,predictions)
print ("DecisionTree Accuracy", accuracy_score(y_test,predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test,predictions))

#SVM
from sklearn import svm
sv = svm.SVC(kernel='linear')
model = sv.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy_score(y_test,predictions)
print ("SVM Accuracy", accuracy_score(y_test,predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test,predictions))
