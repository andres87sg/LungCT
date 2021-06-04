# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 08:11:54 2021

@author: Andres
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2 as cv
import PIL
import pandas as pd

import scipy as sp
from scipy.stats import skew,kurtosis

import skimage

import tqdm

from os import listdir
from os.path import isfile, join


from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

#from sklearn import preprocessing

#%%

import pickle
import joblib

model_filename = 'KNNmodel.pkl'
clf_model = joblib.load(model_filename)

path = 'C:/Users/Andres/Desktop/CovidImages2/Testing/CT2/CT/'
pathmask = 'C:/Users/Andres/Desktop/CovidImages2/Testing/Mask/Mask/'

listfiles = os.listdir(path)
listfilesmask = os.listdir(pathmask)

statslist=[]

i=23

im_name = listfiles[i] # Gray level
im_namemask = listfilesmask[i] # Segmentation mask

# Graylevel image (array)
im_or=cv.imread(path+im_name)
im_array=im_or[:,:,0]
grtr_mask=cv.imread(pathmask+im_namemask)

mask=np.int16(grtr_mask[:,:,0]>0)

kernel = np.ones((10, 10), np.uint8)
cropmask = cv.erode(mask, kernel)

im_or = im_or[:,:,0]*cropmask
grtr_mask = grtr_mask[:,:,0]*cropmask

# plt.figure()
# plt.imshow(im_or,cmap='gray')
# plt.figure()
# plt.imshow(grtr_mask,cmap='gray')

#%% Cluster regions

def kmeanscluster(im_or):
    
    pixel_values = np.float32(im_or.reshape((-1,1)))
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    flags = cv.KMEANS_RANDOM_CENTERS
    
    k=3 # Background, Lung, Consolidation and GGO
    compactness,labels,centers = cv.kmeans(pixel_values,k,None,criteria,10,flags)
    
    centers = np.uint8(centers)
    labels = labels.flatten()
    
    segmented_image_vector = centers[labels.flatten()]
    segmented_image = segmented_image_vector.reshape(im_or.shape)
    
    return segmented_image
    

segmented_image=kmeanscluster(im_or)


# # show the image
# plt.imshow(segmented_image,cmap='gray')
# plt.show()

#%%

clusterlabels = np.unique(segmented_image)

lungmask = segmented_image==clusterlabels[1]

lunginfmask=np.int16(segmented_image==clusterlabels[2])

kernel = np.ones((3,3), np.uint8)
closing = cv.morphologyEx(lunginfmask, cv.MORPH_OPEN, kernel)
plt.imshow(closing,cmap='gray')

lunginfmask = closing.copy()

# Region of interest
roi = np.where(lunginfmask == 1)

plt.imshow(im_or,cmap='gray')
plt.plot(roi[1],roi[0],'.r')

#%%

def feature_extraction(im_or,roi):
    dist=5
    statislist=[]
    # Slicing window
    for k in range(np.shape(roi)[1]):
        #print(k)
        c=roi[1][k],roi[0][k]
        zlz=im_or[c[1]-dist:c[1]+dist,c[0]-dist:c[0]+dist]
        data=zlz.flatten()
        mean_gl = np.mean(data)
        med_gl  = np.median(data)
        std_gl  = np.std(data)
        kurt_gl = sp.stats.kurtosis(data)
        skew_gl = sp.stats.skew(data)        
        statslist.append([mean_gl,med_gl,std_gl,kurt_gl,skew_gl])

    featurematrix = np.array(statslist)    

    return featurematrix

featurematrix=feature_extraction(im_or,roi)

#%% data normalization

scaler = preprocessing.StandardScaler().fit(featurematrix)
featurematrix_norm=scaler.transform(featurematrix)

#%%

predicted_label=clf_model.predict(featurematrix_norm)


#%%
def predmask(roi,predicted_label,label):
    predcoordy=roi[0][predicted_label==label]
    predcoordx=roi[1][predicted_label==label]
    
    predictedmask=np.zeros((np.shape(im_or)[0],np.shape(im_or)[1]))
    predictedmask[predcoordy,predcoordx]=label+1
    return predictedmask
# for i in range(len(x)):
#     unodos[y[i],x[i]]=1
           
ggomask=predmask(roi,predicted_label,1)
conmask=predmask(roi,predicted_label,2)
#%%

plt.imshow(ggomask,cmap='gray')
plt.imshow(conmask,cmap='gray')

plt.imshow(conmask+ggomask+lungmask)

