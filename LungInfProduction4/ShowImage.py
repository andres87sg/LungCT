#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 18:52:52 2021

@author: usuario
"""

import os
import random

import math
import albumentations as A
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use("ggplot")
#%matplotlib inline

import tensorflow as tf

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from tensorflow.keras.layers import Lambda, RepeatVector, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPool2D
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

import tensorflow.keras.backend as T

#%%
#experiment_name = 'LungInf_SF2_Filt64_25052021'
scale_factor = 4
n_filters = 32
image_size = 512
batch_size = 32

img_width =  np.uint16(image_size/scale_factor)
img_height = np.uint16(image_size/scale_factor)
img_channels = 1
color_mode = "rgb"



#%%
def conv_block(tensor, nfilters, size=3, padding='same', initializer="he_normal"):
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def deconv_block(tensor, residual, nfilters, size=3, padding='same', strides=(2, 2)):
    y = Conv2DTranspose(nfilters, kernel_size=(size, size), strides=strides, padding=padding)(tensor)
    y = tf.concat([y, residual], axis=3)
    y = conv_block(y, nfilters)
    return y


def Unet(img_height, img_width, nclasses, filters):
# down
    input_layer = Input(shape=(img_height, img_width, 3), name='image_input')
    conv1 = conv_block(input_layer, nfilters=filters)
    conv1_out = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(conv1_out, nfilters=filters*2)
    conv2_out = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(conv2_out, nfilters=filters*4)
    conv3_out = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block(conv3_out, nfilters=filters*8)
    conv4_out = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv4_out = Dropout(0.5)(conv4_out)
    conv5 = conv_block(conv4_out, nfilters=filters*16)
    conv5 = Dropout(0.5,name='BOTTLENECK')(conv5)
# up
    deconv6 = deconv_block(conv5, residual=conv4, nfilters=filters*8)
    deconv6 = Dropout(0.5)(deconv6)
    deconv7 = deconv_block(deconv6, residual=conv3, nfilters=filters*4)
    deconv7 = Dropout(0.5)(deconv7) 
    deconv8 = deconv_block(deconv7, residual=conv2, nfilters=filters*2)
    deconv9 = deconv_block(deconv8, residual=conv1, nfilters=filters)
# output
    output_layer = Conv2D(filters=1, kernel_size=(1, 1))(deconv9)
    output_layer = BatchNormalization()(output_layer)
    output_layer = Activation('sigmoid')(output_layer)

    model = Model(inputs=input_layer, outputs=output_layer, name='Unet')
    return model

nclasses=4

model = Unet(512//scale_factor,512//scale_factor, nclasses, filters=32)

model.summary()
a=0


#%%
import math
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
#from tqdm import tqdm # Progress bar

import tensorflow as tf
import tensorflow.keras as keras

#model = keras.models.load_model('/home/usuario/Documentos/GBM/Experimentos/PCSegmModel.h5')

#%%

#model.load_weights('/home/usuario/Documentos/GBM/Experimentos/LungSegmModel.h5')
model.load_weights('C:/Users/Andres/Desktop/InfSegmModelCompletoPython.h5')
   
    
    
    
    #%%

import cv2 as cv2
path = 'C:/Users/Andres/Desktop/NuevoLungInfDataset/CT2/'
test_path = 'C:/Users/Andres/Desktop/NuevoLungInfDataset/Mask2/'


listfiles = sorted(os.listdir(path))
mask_listfiles = sorted(os.listdir(test_path))

dicescore = []
accuracy = []
sensitivity = []
specificity = []
f1score = []

IoUmetric = []

## Input image to model must be 128x128 therefore 512/4
#scale = 8
imsize = 128

#del pred_mask3, pred_mask

for i in range(3,4):
#for i in range(31,39):
#for i in tqdm(range(100)):
#for i in range(2,3):

  # List of files
  mask_im_name = mask_listfiles[i]
  im_name = listfiles[i]
       
# Groundtruth image (array)
  mask_array=cv2.imread(test_path+mask_im_name)   # Mask image
  im_array = cv2.imread(path+im_name)               # Graylevel image
  
  kk=mask_array.copy()
  
  mask_array2=mask_array.copy()
  ll=mask_array2[:,:,0]>0
  
  #pred_mask3=ll+pred_mask_res
  
  #im_gray = im_array.copy()
  im_gray = im_array
  
  #mask_array.shape()
  # Groundtruth mask Image resize
  mask_array=cv2.resize(mask_array,(imsize,imsize),interpolation = cv2.INTER_AREA)
  print(np.unique(kk))
  ## Input image to model must be 128x128 therefore 512/4
  scale = 8
  
  # Image resize must resize (Model input 128 x 128)
  im_array=cv2.resize(im_array,(imsize,imsize),
                      interpolation = cv2.INTER_NEAREST)
  im_array=im_array/255
  
  # Adding one dimension to array
  img_array = np.expand_dims(im_array,axis=[0])
  # Generate image prediction
  pred_mask = model.predict(img_array)
  
  #zzz=pred_mask.copy()
  zzz=pred_mask[0,:,:,0]
  
  # Image mask as (NxMx1) array
  pred_mask = pred_mask[0,:,:,0]
  pred_mask = np.uint16(np.round(pred_mask>0.99))
  
  # Resize image to 512x512x1
  pred_mask = cv2.resize(pred_mask,(imsize,imsize), 
                      interpolation = cv2.INTER_NEAREST)
  
  
  
  true_mask = np.uint16(mask_array[:,:,0])//255
  
  plt.figure()
  plt.subplot(1,2,1)
  plt.imshow(mask_array[:,:,0],cmap='gray')
  plt.axis('off')
  plt.title('Ground truth')
  plt.subplot(1,2,2)
  plt.imshow(pred_mask3,cmap='gray')
  plt.title('Prediction')
  plt.axis('off')
  
  intersectmask = true_mask & pred_mask
  
  #sumintersectmask = np.sum(intersectmask)
  
  sumpredtrue = np.sum(true_mask)+np.sum(pred_mask)
  
  if sumpredtrue != 0:
        
      dice = 2*np.sum(intersectmask)/(np.sum(true_mask)+np.sum(pred_mask)+0.001)
  
      dicescore.append(dice)
  
  true_mask_flat = true_mask.flatten()
  pred_mask_flat = pred_mask.flatten()
  
  p = np.sum(true_mask_flat)
  n = np.sum(np.logical_not(true_mask_flat))
  tp = np.sum(true_mask_flat & pred_mask_flat)
  fp = np.sum(np.logical_not(true_mask_flat) & pred_mask_flat)
  tn = np.sum(np.logical_not(true_mask_flat) & np.logical_not(pred_mask_flat))
  fn = np.sum(true_mask_flat & np.logical_not(pred_mask_flat))
  
  acc = (tp+tn)/(p+n)
  sens = tp/(tp+fn+0.01) # Cuidado BUG!
  spec = tn/(tn+fp)

  IoU = (tp+0.00001)/(tp+fp+fn+0.00001)

  #f1 = 2*tp/(2*tp+fp+fn)
  
  IoUmetric.append(IoU)
  accuracy.append(acc)
  sensitivity.append(sens)
  specificity.append(spec)
    #f1score.append(f1)

# Metrics

dicescore = np.array(dicescore)
meandice = np.mean(dicescore)
stddice = np.std(dicescore)

accuracy = np.array(accuracy)
meanacc = np.mean(accuracy)
stdacc = np.std(accuracy)

IoU = np.array(IoUmetric)
meanIoU = np.mean(IoUmetric)
stdIoU = np.std(IoUmetric)

# f1sco = np.array(f1score)
# meanf1 = np.mean(f1sco)
# stdf1 = np.std(f1sco)

print('------------------------')    
print('Mean Dice: '+str(meandice))
print('Std Dice: '+str(stddice))
print('------------------------')
print('Mean Acc: '+str(meanacc))
print('Std Acc: '+str(stdacc))
print('------------------------')
print('------------------------')
print('Mean IoU: '+str(meanIoU))
print('Std IoU: '+str(stdIoU))
print('------------------------')



import skimage.segmentation
from skimage.segmentation import mark_boundaries


path = 'C:/Users/Andres/Desktop/NuevoLungInfDataset/CT/'
path2 = 'C:/Users/Andres/Desktop/NuevoLungInfDataset/CT2/'

im_array = cv2.imread(path+im_name)
im_array2 = cv2.imread(path2+im_name)

pred_mask_res=cv2.resize(pred_mask,(512,512),
                    interpolation = cv2.INTER_NEAREST)  




plt.figure()
plt.axis('off')
plt.imshow(mark_boundaries(im_array,pred_mask_res,color=(1, 0, 0)))
