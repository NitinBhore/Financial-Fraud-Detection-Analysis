import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


meta_data = pd.read_csv("C:/Users/dbda/Desktop/input/PS_20174392719_1491204439457_log.csv")
len(meta_data)
del meta_data['nameDest']
del meta_data['nameOrig']
del meta_data['type']

len(meta_data)

meta_data.describe()

Cols = meta_data[['step','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','isFlaggedFraud']]
y = meta_data['isFraud']
X = Cols

#Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
rfc = RandomForestClassifier() #using default values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) #use this random state to match my results only
#training our model
model = rfc.fit(X_train,y_train)
#predicting our labels
predictions = model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test, predictions))
from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)
print ("Random Forest Accuracy", accuracy_score(y_test,predictions))

#Logistic Regression
from sklearn import linear_model
logitic = linear_model.LogisticRegression()
model = logitic.fit(X_train,y_train)
predictions = model.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test, predictions))
accuracy_score(y_test,predictions)
print ("Logistic Regression Accuracy", accuracy_score(y_test,predictions))

#Naive bayes
from sklearn.naive_bayes import GassianNB
gnb=GassianNB()
model = gnb.fit(X_train,y_train)
predictions = model.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test, predictions))
accuracy_score(y_test,predictions)
print ("Naive bayes Accuracy", accuracy_score(y_test,predictions))