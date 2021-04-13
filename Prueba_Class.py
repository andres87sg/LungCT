# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 07:49:57 2021

@author: Andres
"""

#%%

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Input

from tensorflow.keras import layers, models


import math
import albumentations as A
import matplotlib.pyplot as plt

#%%


train_path = 'C:/Users/Andres/Desktop/CTClassif/train/'
mask_path = 'C:/Users/Andres/Desktop/CTClassif/mask/'

train_datagen = ImageDataGenerator(rescale=1./255)
mask_datagen = ImageDataGenerator(rescale=1./255)

#%%
batch_size = 16
target_size=(512//16, 512//16)

seed = 1 # Provide the same seed and keyword argument
image_generator = train_datagen.flow_from_directory(train_path,target_size=target_size,class_mode=None,seed=1)
mask_generator = train_datagen.flow_from_directory(mask_path,target_size=target_size,class_mode=None,seed=1)

image_validation = train_datagen.flow_from_directory(mask_path,target_size=target_size,class_mode=None,seed=1)

steps = image_generator.n//image_generator.batch_size

train_gen = zip(image_generator,mask_generator)

#%%

for _ in range(5):
    img = image_generator.next()
    mask = mask_generator.next()
    
    print(img.shape)
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.imshow(img[0])
    plt.subplot(1,2,2)
    plt.imshow(mask[0])
    plt.axis('off')
    plt.show()
    
#%%

def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p

def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling2D((2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c
    
#%%

image_size = 512//16

def UNet():
    f = [16, 32, 64, 128, 256]
    inputs = keras.layers.Input((512//16, 512//16, 3))
    
    p0 = inputs
    c1, p1 = down_block(p0, f[0]) #128 -> 64
    c2, p2 = down_block(p1, f[1]) #64 -> 32
    c3, p3 = down_block(p2, f[2]) #32 -> 16
    c4, p4 = down_block(p3, f[3]) #16->8
    
    bn = bottleneck(p4, f[4])
    
    u1 = up_block(bn, c4, f[3]) #8 -> 16
    u2 = up_block(u1, c3, f[2]) #16 -> 32
    u3 = up_block(u2, c2, f[1]) #32 -> 64
    u4 = up_block(u3, c1, f[0]) #64 -> 128
    
    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
    model = keras.models.Model(inputs, outputs)
    return model

model = UNet()
#model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])

model.summary()

#%%
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

#%% 

history = model.fit(image_generator, epochs=2, 
                    validation_data=mask_generator,
                                   )

#%%

