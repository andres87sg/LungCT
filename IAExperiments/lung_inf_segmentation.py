# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 08:11:54 2021
Definitve version
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

import pickle
import joblib

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

def feature_extraction(im_or,roi,subsample):
    dist=5
    statslist=[]
    
    #subsample=1
    
    xcoord=roi[0][::subsample]
    ycoord=roi[1][::subsample]
    
    # Slicing window
    for k in range(np.shape(xcoord)[0]):
        #print(k)
        c=ycoord[k],xcoord[k]
        data=(im_or[c[1]-dist:c[1]+dist,c[0]-dist:c[0]+dist]).flatten()
        mean_gl = np.mean(data)
        med_gl  = np.median(data)
        std_gl  = np.std(data)
        kurt_gl = sp.stats.kurtosis(data)
        skew_gl = sp.stats.skew(data)        
        statslist.append([mean_gl,med_gl,std_gl,kurt_gl,skew_gl])

    featurematrix = np.array(statslist)    

    return featurematrix

def predmask(roi,subsample,predicted_label,label):
    
    #subsample=1   
    predcoordy=roi[0][::subsample][predicted_label==label]
    predcoordx=roi[1][::subsample][predicted_label==label]
    predictedmask=np.zeros((np.shape(im_or)[0],np.shape(im_or)[1]))
    predictedmask[predcoordy,predcoordx]=label+1
    
    return predictedmask


def lunginfectionsegmentation(im_or):

    segmented_image=kmeanscluster(im_or)    
    clusterlabels = np.unique(segmented_image)    
    lungmask = segmented_image==clusterlabels[1]    
    lunginfmask=np.int16(segmented_image==clusterlabels[2])
    
    kernel = np.ones((3,3), np.uint8)
    imopen = cv.morphologyEx(lunginfmask, cv.MORPH_OPEN, kernel)    
    lunginfmask = imopen.copy()
    
    # Region of interest
    roi = np.where(lunginfmask == 1)
    
    subsample=3
    
    featurematrix=feature_extraction(im_or,roi,subsample)
    scaler = preprocessing.StandardScaler().fit(featurematrix)
    featurematrix_norm=scaler.transform(featurematrix)
    
    predicted_label = clf_model.predict(featurematrix_norm)
    
    ggomask=predmask(roi,subsample,predicted_label,1)
    conmask=predmask(roi,subsample,predicted_label,2)
      
    kernel = np.ones((subsample,subsample), np.uint8)
    ggomask_close = cv.morphologyEx(ggomask, cv.MORPH_CLOSE, kernel)   
    conmask_close = cv.morphologyEx(conmask, cv.MORPH_CLOSE, kernel)   
    
    
    #lunginfmask = conmask2+ggomask2+lungmask
    lunginfmask = conmask_close+ggomask_close
    lunginfmask[lunginfmask>3]=0

    return lunginfmask


def regionsegmentation(im_or):
    im_array=im_or[:,:,0]
    
   
    grtr_mask=cv.imread(pathmask+im_namemask)
    
   
    mask=np.int16(grtr_mask[:,:,0]>0)
    
    kernel = np.ones((5, 5), np.uint8)
    cropmask = cv.erode(mask, kernel)
    
    im_or = im_or[:,:,0]*cropmask
    grtr_mask = grtr_mask[:,:,0]*cropmask
    
    final_mask=lunginfectionsegmentation(im_or)
    
    return final_mask

    
#%%

# Load classification model
model_filename = 'KNNmodel.pkl'
clf_model = joblib.load(model_filename)

path = 'C:/Users/Andres/Desktop/CovidImages2/Testing/CT2/CT/'
pathmask = 'C:/Users/Andres/Desktop/CovidImages2/Testing/Mask/Mask/'

listfiles = os.listdir(path)
listfilesmask = os.listdir(pathmask)

#for i in range(len(listfiles)):
for i in range(1,30):
    
    im_name = listfiles[i] # Gray level
    im_namemask = listfilesmask[i] # Segmentation mask
    
    # Graylevel image (array)
    im_or=cv.imread(path+im_name)
    grtr_mask=cv.imread(im_namemask)
    
    scale=1
    im_or2=cv.resize(im_or,(512//scale,512//scale), 
                        interpolation = cv.INTER_AREA)
    
    final_mask=regionsegmentation(im_or2)
       
    plt.figure()
    plt.subplot(1,2,2)
    plt.imshow(final_mask,cmap='gray')
    plt.title('segmentation')
    plt.axis('off')
    plt.subplot(1,2,1)
    plt.imshow(im_or2,cmap='gray')
    plt.title('Im or')
    plt.axis('off')




