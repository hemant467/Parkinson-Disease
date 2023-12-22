import tkinter as tk

 

from tkinter import filedialog

 

from tkinter.filedialog import askopenfile 

from tkinter import messagebox

from tkinter import ttk


    from tensorflow.keras.preprocessing import image 


import numpy as np

import pandas as pd 

import tensorflow as tf 

import tensorflow.keras

from tensorflow.keras import layers 

from tensorflow.keras import Model

from tensorflow.keras.models import Sequential 

from tensorflow.keras.preprocessing import image

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping 

from tensorflow.keras.models import load_model, Model

from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout model= load_model('CNN_Model.h5')

from PIL import Image, ImageTk 

model= load_model('CNN_Model.h5') 

# Display new window

my_w= tk.Tk()

 

# Size of the window 

my_w.geometry("500x350") 

my_w.title('Parkinson_Detection Model') 

my_font1=('times', 20, 'bold')

l1= tk.Label (my_w, text='Input Image',width =30, font=my_font1) 

l1.grid(row=1, column=1)


 

b1= tk.Button(my_w, text='Upload File', width=20, font= my_font1, command= lambda:upload_file())

b1.grid(row=2, column=1)

def upload_file():

global img

 

f_types= [('png Files','*.png')]

 

filename= filedialog.askopenfilename(filetypes=f_types) 

img = ImageTk.PhotoImage(file= filename)

# Using Button

 

b2= tk.Button(my_w, image=img) 

b2.grid(row=3, column=1) 

print(filename) 

Prediction(filename, model)

def Prediction (img, model): 

classes_dir= ["Healthy", "Parkinson"]

a=image.load_img(img, target_size=(100,100))

 

# Normalizing Image

 

norm_img= image.img_to_array(a)/255 

# Converting Image to Numpy Array

input_arr_img = np.array([norm_img]) 

# Getting Predictions

pred = np.argmax(model.predict(input_arr_img))

 

# Printing Model Prediction

 

print('The predicted Class is', classes_dir[pred])


 

l3= tk.Label(my_w, text =(classes_dir[pred]),width=30, font= my_font1)

 

# Keep the window open

 

my_w.mainloop()