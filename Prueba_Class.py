# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 07:49:57 2021

@author: Andres
"""

#%%

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


import math
import albumentations as A
import matplotlib.pyplot as plt

#%%


train_path = 'C:/Users/Andres/Desktop/imexhs/Lung/Prueba/CT/'
mask_path = 'C:/Users/Andres/Desktop/imexhs/Lung/Prueba/mask/'

train_datagen = ImageDataGenerator(rescale=1./255)
mask_datagen = ImageDataGenerator(rescale=1./255)




batch_size = 16
target_size=(256, 256)

seed = 1 # Provide the same seed and keyword argument
image_generator = train_datagen.flow_from_directory(train_path,target_size=target_size,class_mode=None,seed=1)
mask_generator = train_datagen.flow_from_directory(mask_path,target_size=target_size,class_mode=None,seed=1)

steps = image_generator.n//image_generator.batch_size


train_generator = zip(image_generator,mask_generator)

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
