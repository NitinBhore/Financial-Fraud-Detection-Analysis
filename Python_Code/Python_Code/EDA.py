# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 00:45:09 2018

@author: NiTiN
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import style
style.use('ggplot')

import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
     
            
#raed CSV File
data = pd.read_csv("C:/Users/RetailAdmin/SpyderProgram/FFDA/PS_20174392719_1491204439457_log.csv")
           
print ("raw data shape is:", data.shape)

print(data.head())

print(data.tail())

print(data.info())  

print(data.describe())
          
data.isnull().any()
            
# count of fraud and not fraud according to 5 types of payment
pd.crosstab(data['type'], data['isFraud'])

#count of transaction according to 5 types of payment
print(data['type'].value_counts())

sns.countplot(x='type',data=data, )
            
data[['nameOrig', 'nameDest']].describe()

#By looking at boxplots for amount, fraudulent activities tend to have larger amounts
plt.figure(figsize=(12,8))
sns.boxplot(hue = 'isFraud', x = 'type', y = 'amount', data = data[data.amount < 1e5])            


#the correlation of the Feature
sns.heatmap(data.corr())
            
            
            