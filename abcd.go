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
s
s
s
s
s
s
d
d
f
f
f
f
g
g

r
r
r
r
r
v
 
 


r
r
r
r
r
r
r
r
r
r
r
r
r
r
r
g
h
h
h
h




t
t
t
t
yt
y
y
t
r
e
e
w
e
r


y
u
j
j
j
j
j
j
j


s
s
s
s
s
dd
d

f
f
f
f
f
r
e
ww
w
w
e
r
t
y
y
u
u



y
u
u
u
i
i

u
y
t
t

2
2
34
5
5
6
6
6

7
8
8
8
9
9
0
0
0
8
7
6
6
5

4
4
4
5
6
6
7
7
8
h
h
h
h
h



6
6
7
7
77
6
5
4
5
6



https://github.com/fjerbi
https://github.com/fjerbi/fjerbi/blob/main/README.md?plain=1

Add files via upload

https://sites.google.com/view/theparkinsons-disease

PARKINSON’S DISEASE
Parkinson Disease

IDENTIFICATION OF PARKINSON’S DISEASE BY VOICE AND SPIRAL DRAWINGS USING MACHINE LEARNING

Identification of parkinson's disease by voice and spiral drawings using machine learning

https://github.com/hemant467/Parkinson-Disease