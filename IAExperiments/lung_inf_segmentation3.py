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

#%%

def GetPrepareImage(img_in):
    
    im_array=img_in[:,:,0]
    mask=np.zeros((np.shape(img_in)[0],np.shape(img_in)[1]))
    mask[im_array>0]=1
    kernel = np.ones((3, 3), np.uint8)
    cropmask = cv.erode(mask, kernel)
    img_out = img_in[:,:,0]*cropmask
    
    return img_out

def GetFeatureExtraction(scaled_im_or,scaled_segmented_image):
    
    roi = np.where(scaled_segmented_image>0)
    xcoord=roi[0]
    ycoord=roi[1]
    
    dist=5
    statslist=[]
    
    for k in range(np.shape(xcoord)[0]):
    
        c=ycoord[k],xcoord[k]
        data=(scaled_im_or[c[1]-dist:c[1]+dist,c[0]-dist:c[0]+dist]).flatten()
        mean_gl = np.mean(data)
        med_gl  = np.median(data)
        std_gl  = np.std(data)
        kurt_gl = sp.stats.kurtosis(data)
        skew_gl = sp.stats.skew(data)        
        statslist.append([mean_gl,med_gl,std_gl,kurt_gl,skew_gl])

    featurematrix = np.array(statslist) 
    
    return featurematrix

def GetClusteredMask(img_in,scale):
    
    pixel_values = np.float32(img_in.reshape((-1,1)))
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    flags = cv.KMEANS_RANDOM_CENTERS
    k=3 # Background, Lung, Consolidation and GGO
    compactness,labels,centers = cv.kmeans(pixel_values,k,None,criteria,10,flags)
    centers = np.uint8(centers)
    labels = labels.flatten()
    
    segmented_image_vector = centers[labels.flatten()]
    segmented_image = segmented_image_vector.reshape(img_in.shape)  
    
    scaled_im_or=cv.resize(img_in,(512//scale,512//scale), 
            interpolation = cv.INTER_AREA)
    
    scaled_segmented_image=cv.resize(segmented_image,(512//scale,512//scale), 
            interpolation = cv.INTER_AREA)
    
    return scaled_im_or,scaled_segmented_image

def GetPrediction(featurematrix):
    scaler = preprocessing.StandardScaler().fit(featurematrix)
    featurematrix_norm=scaler.transform(featurematrix)

    predicted_label = clf_model.predict(featurematrix_norm)
    return predicted_label

def GetPredictedMask(im_or,subsample,predicted_label,label):
    
    roi = np.where(im_or>0)
    #subsample=1   
    predcoordy=roi[0][::subsample][predicted_label==label]
    predcoordx=roi[1][::subsample][predicted_label==label]
    predictedmask=np.zeros((np.shape(im_or)[0],np.shape(im_or)[1]))
    predictedmask[predcoordy,predcoordx]=label+1
    
    return predictedmask

def GetLungInfSegmentation(scl_img_or,predicted_label):
    
    subsample=1
    lngmask=GetPredictedMask(scl_img_or,subsample,predicted_label,0)
    ggomask=GetPredictedMask(scl_img_or,subsample,predicted_label,1)
    conmask=GetPredictedMask(scl_img_or,subsample,predicted_label,2)
    
    kernel = np.ones((2, 2), np.uint8)
    lngmask = cv.morphologyEx(lngmask, cv.MORPH_OPEN, kernel)    
    ggomask = cv.morphologyEx(ggomask, cv.MORPH_OPEN, kernel)    
    conmask = cv.morphologyEx(conmask, cv.MORPH_OPEN, kernel) 
    
    lunginfmask = ggomask+conmask   
    
    lunginfmask[lunginfmask>3]=0
    
    resizedlunginfmask=cv.resize(lunginfmask,(512,512),interpolation = cv.INTER_AREA)
    # final_mask = cv.GaussianBlur(resizedlunginfmask, (0,0), sigmaX=1, sigmaY=1, borderType = cv.BORDER_DEFAULT)
    # final_mask = np.round(final_mask)
    final_mask=resizedlunginfmask.copy()
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
for i in range(10,11):
    
    im_name = listfiles[i] # Gray level
    im_namemask = listfilesmask[i] # Segmentation mask
    
    # Graylevel image (array)
    im_or=cv.imread(path+im_name)
    
    im_or=GetPrepareImage(im_or)
    scl_img_or,scl_segm_img=GetClusteredMask(im_or,scale=2)
    featurematrix = GetFeatureExtraction(scl_img_or,scl_segm_img)
    predicted_label = GetPrediction(featurematrix)
    final_mask = GetLungInfSegmentation(scl_img_or,predicted_label)
       
    plt.figure()
    plt.imshow(final_mask,cmap='gray')