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
import pandas as pd

import timeit as timeit
from timeit import timeit

import scipy as sp
from scipy.stats import skew,kurtosis

#%%

path = 'C:/Users/Andres/Desktop/CovidImages2/Testing/CT2/CT/'
pathmask = 'C:/Users/Andres/Desktop/CovidImages2/Testing/Mask/Mask/'

listfiles = os.listdir(path)
listfilesmask = os.listdir(pathmask)

i=20

im_name = listfiles[i] # Gray level
im_namemask = listfilesmask[i] # Segmentation mask

# Graylevel image (array)
im_or=cv2.imread(path+im_name)
im_array=im_or[:,:,0]
grtr_mask=cv2.imread(pathmask+im_namemask)
grtr_mask=grtr_mask[:,:,0]

#%%
winsize=10

th=0.95
area_th=(winsize**2)*th

[width,heigth,x]=np.shape(im_or)

col = np.int16(width/winsize)
row = np.int16(heigth/winsize)
grtr_mask2=np.zeros([row,col]) # Mask in pixel size

k=0
coord=[]


for col_ind in range(col):
        for row_ind in range(row):
            patch_bw = grtr_mask[winsize*row_ind:winsize*row_ind+winsize,
                             winsize*col_ind:winsize*col_ind+winsize]
            
            patch_bw_th = np.int16(np.array(patch_bw>0))
            patch_area = np.sum(patch_bw_th)
            
            if patch_area>np.int16(area_th):
                
                #grtr_mask[row_ind][col_ind]=1
                label=np.max(patch_bw)
                coord.append([row_ind,col_ind,label])  

                #grtr_mask2[row_ind][col_ind]=1
                grtr_mask2[row_ind][col_ind]=np.max(patch_bw)
                
plt.imshow(grtr_mask2,cmap='gray')

#%%
winsize=30
import cv2
img=np.zeros([512,512])

# for col_ind in range(col):
#         for row_ind in range(row):
#             for i in range(512):
#                 plt.plot(i,3)
for i in range(row):
    kk=cv2.line(img, (0,winsize*(i+1)),(512-1,winsize*(i+1)),(1, 0, 0), 1)
for i in range(col):
    kk=cv2.line(img, (winsize*(i+1),0),(winsize*(i+1),512-1),(1, 0, 0), 1)
#kk=cv2.line(img, (200,0),(200,512),(255, 0, 0), 2)
#kk=cv2.line(img, (300,0),(300,512),(255, 0, 0), 2)
#kk=cv2.line(img, (400,0),(400,512),(255, 0, 0), 2)

plt.imshow(kk,cmap='gray')                
#cv2.imshow(img,kk

#%%

from PIL import Image, ImageDraw
from skimage import io, color

overlapimg=color.label2rgb(im_array,kk,
                      colors=[(0,0,255),(0,0,255)],
                      alpha=0.0015, bg_label=0, bg_color=None)  


plt.imshow(overlapimg)




#%%
import scipy as sp
from scipy.stats import skew,kurtosis

statslist=[]

for i in range(len(coord)):
    [row_ind,col_ind,label]=coord[i]
    patch=im_or[winsize*row_ind:winsize*row_ind+winsize,
                winsize*col_ind:winsize*col_ind+winsize]
    
    plt.imshow(patch,cmap='gray')
    
    patch=np.ndarray.flatten(patch) # Convierte una matrix en un vector
    
    mean_gl = np.mean(patch)
    med_gl  = np.median(patch)
    std_gl  = np.std(patch)
    kurt_gl = sp.stats.kurtosis(patch)
    skew_gl = sp.stats.skew(patch)
    class_gl= np.int16(label)
    
    statist = [class_gl,mean_gl,med_gl,std_gl,kurt_gl,skew_gl]
    statslist.append(statist)
    
classnames=['class','mean','med','std','skew','kurt']

df = pd.DataFrame(statslist, columns = classnames)
df.head()

#%%

is_one=df.loc[:,'class']==63
dfclass_one=df.loc[is_one]


is_two=df.loc[:,'class']==127
dfclass_two=df.loc[is_two]

is_three=df.loc[:,'class']==255
dfclass_three=df.loc[is_three]

#%%
x1=dfclass_one.iloc[:,4]
y1=dfclass_one.iloc[:,3]

x2=dfclass_two.iloc[:,4]
y2=dfclass_two.iloc[:,3]

x3=dfclass_three.iloc[:,4]
y3=dfclass_three.iloc[:,3]


plt.scatter(x1,y1,marker='.')
plt.scatter(x2,y2,marker='.')
plt.scatter(x3,y3,marker='.')
plt.title('mean vs median')
plt.xlabel('mean')
plt.ylabel('median')





#%%
plt.imshow(grtr_mask2,cmap='gray')
for i in range(500):
    [y,x]=coord[i]
    
    plt.plot(x,y,c='r',marker='*')
    





