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

def feature_extraction(im_or,roi):
    dist=5
    statslist=[]
    # Slicing window
    for k in range(np.shape(roi)[1]):
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

def predmask(roi,predicted_label,label):
    predcoordy=roi[0][predicted_label==label]
    predcoordx=roi[1][predicted_label==label]
    predictedmask=np.zeros((np.shape(im_or)[0],np.shape(im_or)[1]))
    predictedmask[predcoordy,predcoordx]=label+1
    
    return predictedmask


def lunginfectionsegmentation(im_or):

    segmented_image=kmeanscluster(im_or)    
    clusterlabels = np.unique(segmented_image)    
    lungmask = segmented_image==clusterlabels[1]    
    lunginfmask=np.int16(segmented_image==clusterlabels[2])
    
    kernel = np.ones((3,3), np.uint8)
    closing = cv.morphologyEx(lunginfmask, cv.MORPH_OPEN, kernel)    
    lunginfmask = closing.copy()
    
    # Region of interest
    roi = np.where(lunginfmask == 1)
    
    featurematrix=feature_extraction(im_or,roi)
    scaler = preprocessing.StandardScaler().fit(featurematrix)
    featurematrix_norm=scaler.transform(featurematrix)
    
    predicted_label = clf_model.predict(featurematrix_norm)
    ggomask=predmask(roi,predicted_label,1)
    conmask=predmask(roi,predicted_label,2)
    lunginfmask = conmask+ggomask+lungmask
    
    return lunginfmask

    
#%%

# Load classification model
model_filename = 'KNNmodel.pkl'
clf_model = joblib.load(model_filename)

path = 'C:/Users/Andres/Desktop/CovidImages2/Testing/CT2/CT/'
pathmask = 'C:/Users/Andres/Desktop/CovidImages2/Testing/Mask/Mask/'

listfiles = os.listdir(path)
listfilesmask = os.listdir(pathmask)

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

final_mask=lunginfectionsegmentation(im_or)

plt.imshow(final_mask,cmap='gray')



