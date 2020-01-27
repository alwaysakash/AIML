# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 16:54:14 2020

@author: kolar
"""

#Data Preprocessing
#import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#import dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:,4].values

#spliting into test and training data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.25, random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#fitting the classifier
from sklearn.naive_bayes import GaussianNB
classifier= GaussianNB()
classifier.fit(X_train,Y_train)

#predicting the test set results
y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(Y_test,y_pred)

