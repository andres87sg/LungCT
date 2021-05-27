# -*- coding: utf-8 -*-
"""
Created on Wed May 26 09:29:32 2021

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

#%%

import scipy as sp
from scipy.stats import skew,kurtosis


path = 'C:/Users/Andres/Desktop/CovidImages2/Testing/CT2/CT/'
pathmask = 'C:/Users/Andres/Desktop/CovidImages2/Testing/Mask/Mask/'

listfiles = os.listdir(path)
listfilesmask = os.listdir(pathmask)


statslist=[]
classes=[63,127,255]

# NUmber of classes
for j in range(0,3):

    # Muestra todas las imágenes de la carpeta
    for i in range(len(listfiles)):
    
        im_name = listfiles[i] # Gray level
        im_namemask = listfilesmask[i] # Segmentation mask
    
    # Graylevel image (array)
        im_or=cv2.imread(path+im_name)
        im_array=im_or[:,:,0]
        grtr_mask=cv2.imread(pathmask+im_namemask)
        
        #gl graylevel
        
        dataclass=np.array(im_or[grtr_mask==np.int16(classes[j])])
        mean_gl = np.mean(dataclass)
        med_gl  = np.median(dataclass)
        std_gl  = np.std(dataclass)
        kurt_gl = sp.stats.kurtosis(dataclass)
        skew_gl = sp.stats.skew(dataclass)
        class_gl= np.int16(classes[j])
        
        stats = [class_gl,mean_gl,med_gl,std_gl,kurt_gl,skew_gl]
        statslist.append(stats)
        
        
#%%
    
classnames=['class','mean','med','std','skew','kurt']

df = pd.DataFrame(statslist, columns = classnames)
    
#a=df.iloc[:,0].values[~np.isnan(df.iloc[:,0].values)]
#b=df.iloc[:,1].values[~np.isnan(df.iloc[:,1].values)]

print(df)
#%%%

#[~np.isnan(df.iloc[:,i].values)]

x1=df.iloc[0:226,1].values[~np.isnan(df.iloc[0:226,1].values)] # Mean
y1=df.iloc[0:226,2].values[~np.isnan(df.iloc[0:226,2].values)] # Median
z1=df.iloc[0:226,3].values[~np.isnan(df.iloc[0:226,3].values)] # Desv es

x2=df.iloc[227:226+227,1].values[~np.isnan(df.iloc[227:226+227,1].values)]
y2=df.iloc[227:226+227,2].values[~np.isnan(df.iloc[227:226+227,2].values)]
z2=df.iloc[227:226+227,3].values[~np.isnan(df.iloc[227:226+227,3].values)]

x3=df.iloc[226+227+1:226+227+227,1].values[~np.isnan(df.iloc[226+227+1:226+227+227,1].values)]
y3=df.iloc[226+227+1:226+227+227,2].values[~np.isnan(df.iloc[226+227+1:226+227+227,2].values)]
z3=df.iloc[226+227+1:226+227+227,3].values[~np.isnan(df.iloc[226+227+1:226+227+227,3].values)]


plt.scatter(x1,y1,marker='.')
plt.scatter(x2,y2,marker='.')
plt.scatter(x3,y3,marker='.')
plt.title('mean vs median')
plt.xlabel('mean')
plt.ylabel('median')

#%%
imbw_a=np.zeros([512,512])
imbw_b=np.zeros([512,512])

maxim=np.round(np.mean(x3)+np.std(x3))
minim=np.round(np.mean(x3)-np.std(x3))

imbw_a[im_array>minim]=1
imbw_b[im_array<maxim]=1

pp=imbw_a*imbw_b
plt.imshow(pp, cmap='gray')





#%% Experimento aparte
i=20

im_name = listfiles[i] # Gray level
im_namemask = listfilesmask[i] # Segmentation mask

# Graylevel image (array)
im_or=cv2.imread(path+im_name)
im_array=im_or[:,:,0]
grtr_mask=cv2.imread(pathmask+im_namemask)





#%%%
"""


mean,median,std,skew,kurt
"""

# Read ground truth Mask image (array)
 
    
    # x1.append(np.mean(dataclass1))
    # y1.append(np.median(dataclass1))
    # dataclass2=np.array(im_or[grtr_mask==127])
    # x2.append(np.mean(dataclass2))
    # y2.append(np.median(dataclass2))

plt.figure()
plt.scatter(x1,y1,c='b')
plt.scatter(x2,y2,c='r')
plt.scatter(np.mean(x1),np.mean(y1),c='g',marker='D')
plt.scatter(np.mean(x2),np.mean(y2),c='g',marker='D')
plt.show()    
#%%
"""
Extrae los datos que hay dentro de las máscaras d-e segmentación
"""
dataclass1=np.array(im_or[grtr_mask==63])
dataclass2=np.array(im_or[grtr_mask==127])
dataclass3=np.array(im_or[grtr_mask==255])


#%%



