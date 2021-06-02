# -*- coding: utf-8 -*-
"""
Created on Thu May 27 13:39:27 2021

@author: Andres
"""

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras import Input,layers, models
from tensorflow.keras.layers import Conv2DTranspose,Dropout,Conv2D,BatchNormalization, Activation,MaxPooling2D
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

import math
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import PIL
import pandas as pd

import timeit as timeit
from timeit import timeit

import scipy as sp
from scipy.stats import skew,kurtosis

import skimage
from skimage.feature import greycomatrix, greycoprops

#%%

path = 'C:/Users/Andres/Desktop/CovidImages2/Training/CT2/CT/'
pathmask = 'C:/Users/Andres/Desktop/CovidImages2/Training/Mask/Mask/'

listfiles = os.listdir(path)
listfilesmask = os.listdir(pathmask)

i=10

im_name = listfiles[i] # Gray level
im_namemask = listfilesmask[i] # Segmentation mask

# Graylevel image (array)
im_or=cv2.imread(path+im_name)
im_array=im_or[:,:,0]
grtr_mask=cv2.imread(pathmask+im_namemask)

#grtr_mask=grtr_mask[:,:,0]

#mask=np.int16(grtr_mask>0)

#%% Cropping mask

kernel = np.ones((10, 10), np.uint8)
cropmask = cv2.erode(mask, kernel)

im_or=im_or[:,:,0]*cropmask
grtr_mask = grtr_mask[:,:,0]*cropmask


#%%

# Creating kernel
kernel = np.ones((10, 10), np.uint8
  
# Using cv2.erode() method 
imagemask = cv2.erode(mask, kernel)
grtr_mask2 = grtr_mask*imagemask



kk=im_or[im_or>0]

plt.hist(kk,20)
plt.figure()
plt.imshow(imagemask)



