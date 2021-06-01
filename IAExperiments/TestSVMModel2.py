# -*- coding: utf-8 -*-
"""
Created on Mon May 31 15:35:16 2021

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

import scipy as sp
from scipy.stats import skew,kurtosis

import tqdm
import skimage
from skimage.feature import greycomatrix, greycoprops
#%%

path = 'C:/Users/Andres/Desktop/CovidImages2/Testing/CT2/CT/'
pathmask = 'C:/Users/Andres/Desktop/CovidImages2/Testing/Mask/Mask/'

statslist=[]

listfiles = os.listdir(path)
listfilesmask = os.listdir(pathmask)

listfiles.sort()
listfilesmask.sort()

#for i in range(len(listfiles)):
#for i in tqdm.tqdm(range(1,10)):
for i in tqdm.tqdm(range(len(listfiles))):

    im_name = listfiles[i] # Gray level
    im_namemask = listfilesmask[i] # Segmentation mask

    # Graylevel image (array)
    im_or=cv2.imread(path+im_name)
    im_array=im_or[:,:,0]
    grtr_mask=cv2.imread(pathmask+im_namemask)
    grtr_mask=grtr_mask[:,:,0]
    mask=np.int16(grtr_mask>0)
    
    winsize=10
    xmin, ymin, w, h = cv2.boundingRect(np.uint8(mask))
    
    # Redondeo
    xmin=np.uint(np.round(xmin/winsize)*winsize)
    ymin=np.uint(np.round(ymin/winsize)*winsize)
    
    xmax=np.uint((xmin+w)/winsize+1)*winsize
    ymax=np.uint((ymin+h)/winsize+1)*winsize

    # Nuevas imagenes con el tamaño del bounding box
    grtr_mask = grtr_mask[ymin:ymax,xmin:xmax]
    im_array = im_array[ymin:ymax,xmin:xmax]
    im_or = im_or[ymin:ymax,xmin:xmax]
    
    th=0.95
    area_th=(winsize**2)*th
    
    [heigth,width,x]=np.shape(im_or)
    
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
                    
                    [mode_patch,b]=sp.stats.mode(np.ndarray.flatten(patch_bw))
                    
                    grtr_mask2[row_ind][col_ind]=np.int16(mode_patch[0])
                    
    # plt.figure()
    # plt.imshow(grtr_mask2,cmap='gray')
    # plt.axis('off')
    
    for i in range(len(coord)):
        [row_ind,col_ind,label]=coord[i]
        patch=im_or[winsize*row_ind:winsize*row_ind+winsize,
                    winsize*col_ind:winsize*col_ind+winsize]
        
        patch=patch[:,:,0]
    
        '''
        Haralick Texture    
        '''
        glcm = skimage.feature.greycomatrix(patch, distances=[1], angles=[0], levels=256,
                            symmetric=True, normed=True)
        
        contrast = greycoprops(glcm, 'contrast')
        homogene = greycoprops(glcm, 'homogeneity')
        dissimil = greycoprops(glcm, 'dissimilarity')
        energy   = greycoprops(glcm, 'energy')
        
        contr = contrast[0,0]
        homog = homogene[0,0]
        dissi = dissimil[0,0]
        energ = energy[0,0]
        
        
        #plt.imshow(patch,cmap='gray')
        
        patch=np.ndarray.flatten(patch) # Convierte una matrix en un vector
        
        mean_gl = np.mean(patch)
        med_gl  = np.median(patch)
        std_gl  = np.std(patch)
        kurt_gl = sp.stats.kurtosis(patch)
        skew_gl = sp.stats.skew(patch)
        class_gl= np.int16(label)
        
        statist = [class_gl,mean_gl,med_gl,std_gl,kurt_gl,skew_gl,
                   contr,homog,dissi,energ]
        statslist.append(statist)
        
    classnames=['class','mean','med','std','skew','kurt',
                'contr','homog','dissi','energ'                
                ]
    
    df = pd.DataFrame(statslist, columns = classnames)
    df.head()
        
#%%

#%% Load and save
#df.to_csv ('test_dataframe.csv', index = False, header=True)
df = pd.read_csv('test_dataframe.csv')


#%%

is_one=df.loc[:,'class']==63
dfclass_one=df.loc[is_one]

is_two=df.loc[:,'class']==127
dfclass_two=df.loc[is_two]

is_three=df.loc[:,'class']==255
dfclass_three=df.loc[is_three]

class_one=dfclass_one.iloc[:,1:9].values
class_two=dfclass_two.iloc[:,1:9].values
class_three=dfclass_three.iloc[:,1:9].values

label_one = dfclass_one.iloc[:,0].values
label_two = dfclass_two.iloc[:,0].values
label_three = dfclass_three.iloc[:,0].values


features_matrix=np.concatenate((class_one,
                                class_two,
                                class_three
                                ),axis=0)
labels = np.concatenate((label_one,label_two,label_three),axis=0)

#%%

from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

scaler = preprocessing.StandardScaler().fit(features_matrix)
features_matrix_scal=scaler.transform(features_matrix)
X=features_matrix_scal.copy()
y=np.zeros(np.shape(labels))



y[labels==63]=0
y[labels==127]=1
y[labels==255]=2


#%%
#zz1=features_matrix[0:1000]

#predicted=loaded_model.predict(features_matrix)



#%%

import pickle
import joblib

pkl_filename = "pickle_model.pkl"

with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

#%%

predicted_label = pickle_model.predict(X)
true_label=y.copy()

#%%
from sklearn.metrics import confusion_matrix, classification_report

print("Classification report")
print(confusion_matrix(true_label, predicted_label))
print(classification_report(true_label, predicted_label, digits=3))