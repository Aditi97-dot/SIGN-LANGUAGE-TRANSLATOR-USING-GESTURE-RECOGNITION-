import numpy as np
import pandas as pd
import tensorflow as tf
import os

import cv2
import matplotlib.pyplot as plt

import random
from tqdm import tqdm

path = "C:/Users/mukun/Downloads/Image_PRO2/ImagePro"
files = os.listdir(path)

files.sort()
print(files)

#image and its lable
image_array = []
label_array = []

for i in tqdm(range(len(files))):
 sub_files = os.listdir(path+"/"+files[i])
 #print(len(sub_files))
 for j in range(len(sub_files)):
 file_path = path+"/"+files[i]+"/"+sub_files[j]
 #CV2 read image
 image = cv2.imread(file_path)
 #resize 96*96
 image = cv2.resize(image,(96,96))
 if random.randint(1,10) > 6:
 image = cv2.flip(image,1)
 #color blackwhite
 #image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
 #color black to RGB
 #image=cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
 image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 image_array.append(image)

 label_array.append(i)
 
 image_array = np.array(image_array)
label_array = np.array(label_array,dtype="float")

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(image_array,label_array,test_size=0.2)

del image_array,label_array
import gc
gc.collect()

from keras import layers,callbacks,utils,applications,optimizers
from keras.models import Sequential, Model, load_model

model = Sequential()
pretrained_model = tf.keras.applications.EfficientNetB3(input_shape=(96,96,3),include_top = False)
model.add(pretrained_model)

model.add(layers.GlobalAveragePooling2D())

model.add(layers.Dropout(0.3))
model.add(layers.Dense(1))
model.build(input_shape=(None,96,96,2))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

ckg_path = "trained_model/model"
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath= ckg_path, monitor="val_mae",mode="auto",save_best_only=True,save_weight_only=True)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
factor = 0.9, monitor = "val_mae", mode = "auto", cooldown = 0, patience = 5, verbose =1, min_lr= 1e-6)

Epoch = 50
Batch_size = 32

history  = model.fit(X_train,Y_train,validation_data=(X_test,Y_test),batch_size = Batch_size, epochs= Epoch, callbacks=[model_checkpoint,reduce_lr])

results = model.evaluate(X_test,Y_test, batch_size=32)

model.load_weights(ckg_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model=converter.convert()

with open("model2withflips10echB4.tflite","wb") as f:
 f.write(tflite_model)

import math

prediction_val = model.predict(X_test,batch_size=32)
print(prediction_val)
print(Y_test)

pre = prediction_val.tolist()
y_tes = Y_test.tolist()

correct = 0
wrong = 0
for i in range(len(pre)):
 if round(pre[i][0]) == y_tes[i]:
 correct += 1
 else:
 wrong += 1
a = correct/len(pre)*100
b = wrong/len(pre)*100
print(a,b)

import matplotlib.pyplot as plt
left = [1, 2]
height = [b,a]
tick_label = ['Wrong','Correct']
for index, value in enumerate(height):
 plt.text(value, index,
 str(value))
plt.barh(left, height, tick_label = tick_label, color = ['red','green'])
plt.ylabel('Predictions')
plt.xlabel('Scale 0-100')
plt.title('Predictions Of Model')
plt.figure(figsize=(10, 10))
plt.show()
