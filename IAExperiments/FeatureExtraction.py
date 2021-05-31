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

import scipy as sp
from scipy.stats import skew,kurtosis

#%%

path = 'C:/Users/Andres/Desktop/CovidImages2/Training/CT2/CT/'
pathmask = 'C:/Users/Andres/Desktop/CovidImages2/Training/Mask/Mask/'

statslist=[]

listfiles = os.listdir(path)
listfilesmask = os.listdir(pathmask)

for i in range(len(listfiles)):

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
        
#        plt.imshow(patch,cmap='gray')
        
        patch=np.ndarray.flatten(patch) # Convierte una matrix en un vector
        
        mean_gl = np.mean(patch)
        med_gl  = np.median(patch)
        std_gl  = np.std(patch)
        kurt_gl = sp.stats.kurtosis(patch)
        skew_gl = sp.stats.skew(patch)
        class_gl= np.int16(label)
        
        statist = [class_gl,mean_gl,med_gl,std_gl,kurt_gl,skew_gl]
        statslist.append(statist)
        
#%%

classnames=['class','mean','med','std','skew','kurt']

df = pd.DataFrame(statslist, columns = classnames)
df.head()

is_one=df.loc[:,'class']==63
dfclass_one=df.loc[is_one]

is_two=df.loc[:,'class']==127
dfclass_two=df.loc[is_two]

is_three=df.loc[:,'class']==255
dfclass_three=df.loc[is_three]

class_one=dfclass_one.iloc[:,1:6].values
class_two=dfclass_two.iloc[:,1:6].values
class_three=dfclass_three.iloc[:,1:6].values

label_one = dfclass_one.iloc[:,0].values
label_two = dfclass_two.iloc[:,0].values
label_three = dfclass_three.iloc[:,0].values


features_matrix=np.concatenate((class_one,class_two,class_three),axis=0)
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
X=features_matrix.copy()
y=labels

#%%

mm=[]
scores=[]
k=0
listaclf=[]


kf = KFold(n_splits=10,shuffle=True)
for train, test in kf.split(X):
    k=k+1
    print(k)
    #print('Train: %s | test: %s' % (train, test))
    clf = svm.SVC(kernel='linear', C=1).fit(X[train], y[train])
    mm.append(clf)
    listaclf.append(k)
    #print(clf)
    sco=clf.score(X[test],y[test])
    scores.append(sco)
    #print(sco)
    #scores = cross_val_score(clf, X[test], y[test], cv=5)
    
scores==np.max(scores)

zmn=listaclf*(scores==np.max(scores))
bestindclf=np.max(zmn)-1
model=mm[bestindclf]

#%%
import pickle

filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

#%%

loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, Y_test)
print(result)

#%%






    