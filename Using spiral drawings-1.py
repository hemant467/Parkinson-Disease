import numpy as np 

import pandas as pd

import os

import matplotlib.pyplot as plt 

import tensorflow as tf

from tensorflow.keras.utils import to_categorical

 

from tensorflow.keras.preprocessing.image import load_img, img_to_array 

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D 

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.python.ops.numpy_ops import np_utils

 

from sklearn.metrics import classification_report, log_loss, accuracy_score 

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator 

from tensorflow.keras.optimizers import Adam

import warnings 

warnings.filterwarnings("ignore") 

import numpy as np

import cv2


from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dropout,

Flatten, Dense, MaxPool2D

 

from tensorflow.keras.models import Model, Sequential 

from tensorflow.keras.initializers import glorot_uniform 

from tensorflow.keras.optimizers import Adam, SGD 

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

 

from tensorflow.keras.preprocessing.image import ImageDataGenerator 

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report, confusion_matrix 

import seaborn as sns

from tensorflow.keras.regularizers import l2 

dir_sp_train = './parkinsons-drawings/spiral/training' 

dir_sp_test = './parkinsons-drawings/spiral/testing' 

dir_wv_train = './parkinsons-drawings/wave/training' 

dir_wv_test = './parkinsons-drawings/wave/testing'

# Says how many files are present in that particular file/folder

 

Name=[]

 

for file in os.listdir(dir_sp_train): 

Name+=[file]

print(Name) 

N=[]


# To access each item of the sequence with the help of index

 

for i in range(len(Name)):

 

N+=[i]

 

normal_mapping=dict(zip(Name,N)) reverse_mapping=dict(zip(N,Name))

# Tranform maps string value from one to another

 

def mapper(value):

 

return reverse_mapping[value] 

dataset_sp=[]

count=0

 

for file in os.listdir(dir_sp_train):

 

# Combine paths names into one complete path

 

path=os.path.join(dir_sp_train,file) 

for im in os.listdir(path):

image=load_img(os.path.join(path,im),grayscale=False, color_mode='rgb', target_size=(100,100))

image=img_to_array(image)

image=image/255.0

# Adds single item to certain collection types dataset_sp.append([image,count]) 

count=count+1

testset_sp=[] 

count=0

for file in os.listdir(dir_sp_test):


   path=os.path.join(dir_sp_test,file) 

   

for im in os.listdir(path):

image=load_img(os.path.join(path,im),grayscale=False,color_mode='rgb', 

target_size=(100,100))

image=img_to_array(image) 

image=image/255.0 

testset_sp.append([image,count]) 

count=count+1

dataset_wv=[] 

count=0

for file in os.listdir(dir_wv_train): path=os.path.join(dir_wv_train,file) 

for im in os.listdir(path):

image=load_img(os.path.join(path,im), grayscale=False, color_mode='rgb', target_size=(100,100))

image=img_to_array(image) 

image=image/255.0 

dataset_wv.append([image,count]) 

count=count+1

testset_wv=[] 

count=0

for file in os.listdir(dir_wv_test): path=os.path.join(dir_wv_test,file) 

for im in os.listdir(path):

   image=load_img(os.path.join(path,im), grayscale=False, color_mode='rgb', 

   target_size=(100,100))

image=img_to_array(image) 

image=image/255.0 

testset_wv.append([image,count]) 

count=count+1

# Takes in iterables as arguments and return iterators data_sp,labels_sp0=zip(*dataset_sp) 

test_sp,tlabels_sp0=zip(*testset_sp) 

data_wv,labels_wv0=zip(*dataset_wv)
test_wv,tlabels_wv0=zip(*testset_wv)

# Converts an integer to binary matrix labels_sp1=to_categorical(labels_sp0) data_sp=np.array(data_sp) 

labels_sp=np.array(labels_sp1) tlabels_sp1=to_categorical(labels_sp0) test_sp=np.array(test_sp) 

tlabels_sp=np.array(tlabels_sp1) labels_wv1=to_categorical(labels_wv0) data_wv=np.array(data_wv) labels_wv=np.array(labels_wv1) tlabels_wv1=to_categorical(labels_wv0) test_wv=np.array(test_wv) tlabels_wv=np.array(tlabels_wv1)


trainx_sp,testx_sp,trainy_sp,testy_sp=train_test_split(data_sp,labels_sp,test_size=0.2,rand   	om_state=44)

trainx_wv,testx_wv,trainy_wv,testy_wv=train_test_split(data_wv,labels_wv,test_size=0.2,r andom_state=44)

print(trainx_sp.shape) print(testx_sp.shape) print(trainy_sp.shape) print(testy_sp.shape) print(" CNN Model Build")

# To create models layer by layer

 

model= Sequential()

 

# Convolution: Deep learning algorithm specially designed for working images and videos

 

model.add(Conv2D(filters=32, kernel_size=(5,5),activation="relu", 

input_shape=(100,100,3))) #applys conv to images

model.add(MaxPool2D(pool_size=(2,2))) #adjusts the image's pixels

 

# 2nd Convolutional layer

 

model.add(Conv2D(filters=64, kernel_size=(5,5),activation="relu")) model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten()) #returns the copy of array 

# 1st Fully Connected Layer model.add(Dense(units=64,activation="relu")) # Add Output Layer

model.add(Dense(units=2,activation="softmax"))#classification of tasks 

print ("Model Summary")

model.summary() # Print Summary of the Model


 

model.compile(loss='categorical_crossentropy', optimizer= Adam(lr=0.001), metrics=['accuracy'])

# Timer

 

epoch =30

 

# Samples processed before the model is updated

 

batch_size=32

 

print("Start Training", '\n')

 

hist_0= model.fit(trainx_sp,trainy_sp, batch_size=batch_size, epochs=epoch, validation_data=(testx_sp, testy_sp))#provides fit statistics

hist_1= model.fit(trainx_wv,trainy_wv, batch_size=batch_size, epochs=epoch, validation_data=(testx_wv, testy_wv))

print("Training End", '\n') 

model.save ("CNN_Model.h5") figure=plt.figure(figsize=(10,10))

plt.plot(hist_0.history['accuracy'],label='Train_accuracy') plt.plot(hist_0.history['val_accuracy'],label='Test_accuracy') 

plt.title('Model Accuracy')

plt.xlabel('Epochs')
plt.ylabel('Accuracy') 

plt.legend(loc="upper left") 

plt.show() 

figure2=plt.figure(figsize=(10,10))

plt.plot(hist_0.history['loss'],label='Train_loss') plt.plot(hist_0.history['val_loss'],label='Test_loss') 

plt.title('Model Loss')


 

plt.xlabel('Epochs') 

plt.ylabel('Loss') 

plt.legend(loc="upper left") 

plt.show()

figure = plt.figure(figsize=(10,10)) plt.plot(hist_1.history['accuracy'],label='Train_accuracy') plt.plot(hist_1.history['val_accuracy'],label='Test_accuracy') 

plt.title('Model Accuracy')

plt.xlabel('Epochs') 

plt.ylabel('Accuracy') 

plt.legend(loc="upper left") 

plt.show()

figure2 = plt.figure(figsize=(10,10)) 

plt.plot(hist_1.history['loss'], label='Train_loss') plt.plot(hist_1.history['val_loss'], label='Test_loss') 

plt.title('Model Loss')

plt.xlabel('Epochs') 

plt.ylabel('Loss') 

plt.legend(loc="upper left") 

plt.show()

datagen=ImageDataGenerator(horizontal_flip=True,vertical_flip=True,rotation_range=20, zoom_range=0.2,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.1,fill_mod e="nearest")

pretrained_model3=tf.keras.applications.DenseNet201(input_shape=(100,100,3),include_t op=False,weights='imagenet', pooling='avg')


 

pretrained_model3.trainable=False

 

pretrained_model4=tf.keras.applications.DenseNet201(input_shape=(100,100,3),include_t op=False,weights='imagenet', pooling='avg')

pretrained_model4.trainable =False 

inputs3= pretrained_model3.input

x3=tf.keras.layers.Dense(128, activation='relu')(pretrained_model3.output) outputs3=tf.keras.layers.Dense(2, activation='softmax')(x3) model3=tf.keras.Model(inputs=inputs3, outputs=outputs3)

inputs4= pretrained_model4.input

 

x4=tf.keras.layers.Dense(128, activation='relu')(pretrained_model4.output) outputs4=tf.keras.layers.Dense(2, activation='softmax')(x4) 

model4=tf.keras.Model(inputs=inputs4, outputs=outputs4) model3.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy']) model4.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

sp_3=model3.fit(datagen.flow(trainx_sp,trainy_sp,batch_size=32),validation_data=(testx_ sp,testy_sp),epochs=100)

wv_4=model4.fit(datagen.flow(trainx_wv,trainy_wv,batch_size=32),validation_data=(test x_wv,testy_wv),epochs=100)

#Spiral 

y_pred_sp=model3.predict(testx_sp) pred_sp=np.argmax(y_pred_sp,axis=1) 

ground_sp= np.argmax(testy_sp,axis=1)

print(classification_report(ground_sp,pred_sp))

 

# Wave

 

y_pred_wv=model3.predict(testx_wv)


pred_wv=np.argmax(y_pred_wv,axis=1) 


ground_wv= np.argmax(testy_wv,axis=1) 


print(classification_report(ground_wv,pred_wv)) 


get_acc3=sp_3.history['accuracy'] 


value_acc3=sp_3.history['val_accuracy'] 


get_loss3=sp_3.history['loss'] 


validation_loss3=sp_3.history['val_loss'] 


epochs3=range(len(get_acc3))

plt.plot(epochs3,get_acc3,'r', label='Accuracy of Training data') plt.plot(epochs3,value_acc3,'b', label='Accuracy of Validation data') plt.title('Training vs Validation Accuracy -Spiral')

plt.legend(loc=0) 

plt.figure() 

plt.show()

epochs3=range(len(get_loss3)) 

plt.plot(epochs3,get_loss3,'r', label='Loss of Training data')

plt.plot(epochs3,validation_loss3,'b', label='Loss of Validation data') plt.title('Training vs Validation Accuracy -Spiral')

plt.legend(loc=0) 

plt.figure() 

plt.show()

load_img("./parkinsons- drawings/spiral/testing/healthy/V10HE01.png",target_size=(100,100))

image= load_img("./parkinsons- drawings/spiral/testing/healthy/V10HE01.png",target_size=(100,100))

image=img_to_array(image) 

image=image/255.0 

prediction_image=np.array(image)

# New axis position in the expanded array prediction_image=np.expand_dims(image, axis=0) prediction=model3.predict(prediction_image) value=np.argmax(prediction) 

move_name=mapper(value)

print("Prediction is {}.".format(move_name))

 

load_img("./parkinsons- drawings/wave/testing/parkinson/V03PO01.png",target_size=(100,100))

image2= load_img("./parkinsons- drawings/wave/testing/parkinson/V03PO01.png",target_size=(100,100))

image2=img_to_array(image2) 

image2=image2/255.0 

prediction_image2=np.array(image) prediction_image2=np.expand_dims(image, axis=0) prediction2=model4.predict(prediction_image2) value2=np.argmax(prediction2) 

move_name2=mapper(value2)

print("Prediction is {}.".format(move_name2)) print(test_sp.shape) 

prediction_sp=model3.predict(test_sp) 

print(prediction_sp.shape)

PRED_sp=[]


 

for item in prediction_sp:

 

# Process of accessing specific element value_sp=np.argmax(item) 

PRED_sp+=[value_sp]

ANS_sp=tlabels_sp0

 

accuracy_sp= accuracy_score (ANS_sp, PRED_sp) print(accuracy_sp)

print(test_wv.shape) 

prediction_wv=model4.predict(test_wv) print(prediction_wv.shape) 

PRED_wv=[]

for item in prediction_wv: 

value_wv=np.argmax(item) 

PRED_wv+=[value_wv]

ANS_wv=tlabels_wv0

 

accuracy_wv= accuracy_score (ANS_wv, PRED_wv) print(accuracy_wv)