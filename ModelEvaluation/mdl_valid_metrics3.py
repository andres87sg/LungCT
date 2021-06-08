# -*- coding: utf-8 -*-
"""
Compute Validation metrics
Jaccard Index

@author: Andres Sandino

https://stackoverflow.com/questions/53248099/keras-image-segmentation-using-grayscale-masks-and-imagedatagenerator-class

"""

import tensorflow as tf
import tensorflow.keras as keras

# from tensorflow.keras import Input,layers, models
# from tensorflow.keras.layers import Conv2DTranspose,Dropout,Conv2D,BatchNormalization, Activation,MaxPooling2D
# from tensorflow.keras import Model
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

import math
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2 as cv
import pandas as pd

import timeit as timeit
from timeit import timeit

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


#%% Loading model 
#modelpath = 'C:/Users/Andres/Desktop/CTClassif/LungInf_SF2_Filt64_Python_25052021.h5'
#model = keras.models.load_model(modelpath)

#%%

def imoverlay(img,predimg,coloredge):
    
    
    #predictedrgb=np.zeros((512,512,3))
    
    # Upsample image to 512x512
    predmask = cv.resize(predimg,(512,512), interpolation = cv.INTER_AREA)
    predmask = predmask*255
    
    # Upsample image to 512x512    
    img = cv.resize(img,(512,512), interpolation = cv.INTER_AREA)
    
    overlayimg = img.copy()
    overlayimg[predmask == 255] = coloredge
    
    return overlayimg

# Convert image in a tensor
def getprepareimg(im_array,scale):
    
    # Resize image (Input array to segmentation model)
    im_array=cv.resize(im_array,(512//scale,512//scale), 
                        interpolation = cv.INTER_AREA)
    
    # Image gray level normalization
    im_array=im_array/np.max(im_array)
    
    # Adding one dimension to array
    im_array_out = np.expand_dims(im_array,axis=[0])
    
    return im_array_out

# Convert gray mask to color mask
def getcolormask(inputmask):
    
    # Labels in the image    
    lab=np.unique(inputmask)
    
    # Image size
    [w,l] = np.shape(inputmask)
    
    # 3-Channel image
    colormask = np.zeros([w,l,3])
    
    # Gray level image
    graymask = np.zeros([w,l]) 
    
    # Color label (black, green, red, blue)
    colorlabel=([0,0,0],[0,255,0],[255,0,0],[0,0,255]) # Colors
    graylabel=[0,1,2,3] # Gray leves
    
    # Replace values in the image 
    for lab in lab:
        colormask[inputmask==lab]=colorlabel[np.int16(lab)]
        graymask[inputmask==lab]=graylabel[np.int16(lab)]
    
    # Mask in color
    colormask=np.int16(colormask)
    
    # Mask in graylevel
    graymask=np.int16(graymask)
    
    
    return colormask,graymask

# Jaccard Index
def jaccarindex(grtr_mask,pred_mask,label):
    
    grtr=np.zeros([512,512])
    
    # Choose pixels corresponding to label
    grtr[grtr_mask==label]=1
    
    pred=np.zeros([512,512])
    pred[pred_mask==label]=1
    
    # Intersection
    inter= np.sum(grtr*pred>=1)
    #print(inter)
    
    # Union
    union=np.sum(grtr+pred>=1)
    #print(union)
    jaccard=inter/union
    
    return jaccard

#%% Main 

path = 'C:/Users/Andres/Desktop/CovidImages2/Testing/CT2/CT/'
pathmask = 'C:/Users/Andres/Desktop/CovidImages2/Testing/Mask/Mask/'


#destpath = 'C:/Users/Andres/Desktop/CovidImages/Testing/CT2/CT/'
listfiles = os.listdir(path)
listfilesmask = os.listdir(pathmask)


colormat=np.zeros([512,512])

grtr_mask=[] #Groundtruth mask
classes = 4
scale = 2
i=1
jaccard_df=[] #Jaccard index dataframe
model_filename = 'KNNmodel.pkl'
clf_model = joblib.load(model_filename)

#for i in range(15,21):
for i in range(len(listfiles)):
    
    # List of files
    im_name = listfiles[i] # Gray level
    im_namemask = listfilesmask[i] # Segmentation mask
    
    # Graylevel image (array)
    im_or=cv.imread(path+im_name)
    im_array=im_or


    # Read ground truth Mask image (array)
    grtr_mask=cv.imread(pathmask+im_namemask)
    
    # Convert RGB mask to Grayscale
    grtr_mask = grtr_mask[:,:,0] 
    grtr_mask = np.round(grtr_mask/255*classes)
    grtr_mask[grtr_mask==4]=3 # Recuerde que asigno 3 a los valores 4 (Opcional)
    grtr_mask2 = grtr_mask
    
    
    #scale =1
    #input_img_mdl = getprepareimg(im_array,scale)


    pred_mask=regionsegmentation(im_or)
    
    # a=0
    pred_maskmulti = pred_mask.copy()


    # pred_maskmulti=np.round(pred_mask*classes)
    # pred_maskmulti=pred_maskmulti-1 #Classes: 0,1,2,3

    # # Resize predicted mask
    pred_mask = cv.resize(pred_maskmulti,(512,512), 
                          interpolation = cv.INTER_AREA)
    
    # Convert gray mask to color mask    
    col_predmask,gray_predmask = getcolormask(pred_mask)
    col_grtrmask,gray_grtrmask = getcolormask(grtr_mask)
    
    label_list = []
    label_list = np.unique(grtr_mask.tolist() + pred_mask.tolist())
    jaccard_list=[np.nan,np.nan,np.nan,np.nan] # Empty list
  
    
    for label_list in label_list:
        index = int(label_list)
        #print(index)
        jaccard_list[index]=jaccarindex(grtr_mask,pred_mask,label_list)
        
        
        #print(jaccard_list[index])
    
    jaccard_df.append(jaccard_list)
    jack=jaccarindex(grtr_mask,pred_mask,2)
    print(jack)
    
    jack=jaccarindex(grtr_mask,pred_mask,3)
    print(jack)

    
    # plt.figure()
    # plt.subplot(1,3,1)
    # plt.imshow(im_array,cmap='gray')
    # plt.axis('off')
    # plt.title('Gray Level')
    
    # plt.subplot(1,3,2)
    # plt.imshow(col_grtrmask,cmap='gray')
    # plt.axis('off')  
    # plt.title('Groundtruth')
    
    
    # plt.subplot(1,3,3)    
    # plt.imshow(col_predmask,cmap='gray')
    # plt.axis('off')
    # plt.title('Predicted')
    # plt.show()

#%% Show validation metrics (Jaccard Index (mean,std) )

'''
Bkg: Background
Lung: Healty lung
GGO: Ground-glass opacity
Cons: Consolidation
'''
classnames=['Bkg','Lung','GGO','Cons']

df = pd.DataFrame(jaccard_df, columns = classnames)

for i in range(4):
    meanvalue=np.round(
        # selecciona valores de la columna sin tomar NAN
        np.mean(df.iloc[:,i].values[~np.isnan(df.iloc[:,i].values)]),3
        )
    stdvalue=np.round(
        np.std(df.iloc[:,i].values[~np.isnan(df.iloc[:,i].values)]),4
        )
    print(classnames[i]+ ': ' + str(meanvalue) + ' +- ' + str(stdvalue))

#%% Step 2

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
    
    xcoord=roi[0][::subsample]
    ycoord=roi[1][::subsample]
    
    # Slicing window
    for k in range(np.shape(xcoord)[0]):
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
    lunginfmask = conmask_close+ggomask_close+lungmask
    lunginfmask[lunginfmask>3]=0

    return lunginfmask

def regionsegmentation(im_or):
    im_array=im_or[:,:,0]
    grtr_mask=cv.imread(pathmask+im_namemask)
    
    mask=np.int16(grtr_mask[:,:,0]>0)
    
    kernel = np.ones((1, 1), np.uint8)
    cropmask = cv.erode(mask, kernel)
    
    im_or = im_or[:,:,0]*cropmask
    grtr_mask = grtr_mask[:,:,0]*cropmask
    
    final_mask=lunginfectionsegmentation(im_or)
    
    return final_mask