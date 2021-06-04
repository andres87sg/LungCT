# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 08:11:54 2021

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

import tqdm

import scipy as sp

from os import listdir
from os.path import isfile, join

#%%
path = 'C:/Users/Andres/Desktop/CovidImages2/Testing/CT2/CT/'
pathmask = 'C:/Users/Andres/Desktop/CovidImages2/Testing/Mask/Mask/'

listfiles = os.listdir(path)
listfilesmask = os.listdir(pathmask)

statslist=[]

i=20

im_name = listfiles[i] # Gray level
im_namemask = listfilesmask[i] # Segmentation mask

# Graylevel image (array)
im_or=cv2.imread(path+im_name)
im_array=im_or[:,:,0]
grtr_mask=cv2.imread(pathmask+im_namemask)

mask=np.int16(grtr_mask[:,:,0]>0)

kernel = np.ones((10, 10), np.uint8)
cropmask = cv2.erode(mask, kernel)

im_or=im_or[:,:,0]*cropmask
grtr_mask = grtr_mask[:,:,0]*cropmask

plt.imshow(im_or)
plt.imshow(grtr_mask)

#%%

pp1=np.zeros((512,512,3))
pp1[:,:,0]=im_or
pp1[:,:,1]=im_or
pp1[:,:,2]=im_or

pixel_values = np.float32(im_or.reshape((-1,1)))
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

flags = cv.KMEANS_RANDOM_CENTERS

k=3
compactness,labels,centers = cv.kmeans(pixel_values,k,None,criteria,10,flags)

centers = np.uint8(centers)
labels = labels.flatten()

segmented_image_vector = centers[labels.flatten()]

segmented_image = segmented_image_vector.reshape(im_or.shape)
# show the image
plt.imshow(segmented_image,cmap='gray')
plt.show()

#%%

masked_image = np.copy(pp1)
masked_image = masked_image.reshape((-1, 3))
cluster = 2
masked_image[labels == cluster] = [0, 0, 0]
masked_image = masked_image.reshape(pp1.shape)
plt.imshow(masked_image)
plt.show()

#%%

import cv2
import numpy as np

label_list=np.unique(segmented_image)
pa1=np.int16(segmented_image==label_list[2])

kernel = np.ones((3,3), np.uint8)
closing = cv2.morphologyEx(pa1, cv2.MORPH_OPEN, kernel)
plt.imshow(closing,cmap='gray')

#%%

roi = np.where(closing == 1)
#roi = np.where(grtr_mask == 255)
plt.imshow(im_or,cmap='gray')
plt.scatter(roi[1],roi[0],marker='*')

for i in range(np.shape(roi)[1]):
    value=im_or[roi[0][i],roi[1][i]]
    #print(value)

np.shape(roi)[1]
dist=5
kk=[]

for k in range(np.shape(roi)[1]):
    c=roi[1][k],roi[0][k]
    #zlz=im_or[c[0]-dist:c[0]+dist+1,c[1]-dist:c[1]+dist+1]
    #zlz=grtr_mask[c[1]-dist:c[1]+dist,c[0]-dist:c[0]+dist]
    zlz=im_or[c[1]-dist:c[1]+dist,c[0]-dist:c[0]+dist]
    data=zlz.flatten()
    mean_gl = np.mean(data)
    med_gl  = np.median(data)
    std_gl  = np.std(data)
    kurt_gl = sp.stats.kurtosis(data)
    skew_gl = sp.stats.skew(data)
    
    kk.append([mean_gl,med_gl,std_gl,kurt_gl,skew_gl])
    
X=np.array(kk)    
features_matrix=X.copy()
scaler = preprocessing.StandardScaler().fit(features_matrix)
features_matrix_scal=scaler.transform(features_matrix)
X=features_matrix_scal.copy()
#y=true_labels


    # plt.figure()
    # plt.imshow(zlz)
    # plt.scatter(20,20,marker='*')

#%%
# plt.figure()
# plt.imshow(im_or,cmap='gray')
# plt.scatter(roi[1][0:10],roi[0][0:10])

import pickle
import joblib

pkl_filename = "KNNmodel.pkl"

joblib_model = joblib.load(pkl_filename)

# with open(pkl_filename, 'rb') as file:
#     loaded_model = pickle.load(pkl_filename)
#%%

predicted_label=joblib_model.predict(X)


y=roi[0][predicted_label==2]
x=roi[1][predicted_label==2]

# plt.figure()
# plt.imshow(im_or,cmap='gray')
# plt.plot(x,y,'.r')

unodos=np.zeros((np.shape(im_or)[0],np.shape(im_or)[1]))
unodos[y,x]=1
# for i in range(len(x)):
#     unodos[y[i],x[i]]=1
           
plt.imshow(unodos,cmap='gray')

#%%

def graylevelstatistics(data):
    mean_gl = np.mean(data)
    med_gl  = np.median(data)
    std_gl  = np.std(data)
    kurt_gl = sp.stats.kurtosis(data)
    skew_gl = sp.stats.skew(data)
    return mean_gl,med_gl,std_gl,kurt_gl,skew_gl
    # class_gl= data_class[1]
    # statist = [class_gl,mean_gl,med_gl,std_gl,kurt_gl,skew_gl]
    # statslist.append(statist)
     
    
    # classnames=['class','mean','med','std','skew','kurt']
    
    # df = pd.DataFrame(statslist, columns = classnames)
    # df.head()

