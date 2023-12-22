import numpy as np 

import pandas as pd 

import os, sys

df = pd.read_csv("parkinsons.data") 

df.tail()

df.describe() 

df.info() 

df.shape

features = df.loc[:, df.columns != 'status'].values[:, 1:] 

labels = df.loc[:, 'status'].values

from sklearn.preprocessing import MinMaxScaler 

scaler = MinMaxScaler((-1, 1))

X = scaler.fit_transform(features)

 y = labels

from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test=train_test_split(X, y, test_size=0.30)

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

 model = XGBClassifier()

model.fit(x_train, y_train) 

y_prediction = model.predict(x_test)


print("Accuracy Score is", accuracy_score(y_test, y_prediction) * 100) 