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
import cv2
import pandas as pd

import timeit as timeit
from timeit import timeit


#%% Loading model 
modelpath = 'C:/Users/Andres/Desktop/CTClassif/LungInf_SF2_Filt64_Python_25052021.h5'
model = keras.models.load_model(modelpath)

#%%

def imoverlay(img,predimg,coloredge):
    
    
    #predictedrgb=np.zeros((512,512,3))
    
    # Upsample image to 512x512
    predmask = cv2.resize(predimg,(512,512), interpolation = cv2.INTER_AREA)
    predmask = predmask*255
    
    # Upsample image to 512x512    
    img = cv2.resize(img,(512,512), interpolation = cv2.INTER_AREA)
    
    overlayimg = img.copy()
    overlayimg[predmask == 255] = coloredge
    
    return overlayimg

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

jaccard_df=[] #Jaccard index dataframe

#for i in range(1,3):
for i in range(len(listfiles)):
    
    # List of files
    im_name = listfiles[i] # Gray level
    im_namemask = listfilesmask[i] # Segmentation mask
    
    # Graylevel image (array)
    im_or=cv2.imread(path+im_name)
    im_array=im_or
    
    # Read ground truth Mask image (array)
    grtr_mask=cv2.imread(pathmask+im_namemask)
    
    # Convert RGB mask to Grayscale
    grtr_mask = grtr_mask[:,:,0] 
    grtr_mask = np.round(grtr_mask/255*classes)
    grtr_mask[grtr_mask==4]=3 # Recuerde que asigno 3 a los valores 4 (Opcional)
    grtr_mask2 =grtr_mask
       
    
    input_img_mdl = getprepareimg(im_array,scale)
    
    # Generate image prediction
    pred_mask = model.predict(input_img_mdl)
    
    # Image mask as (NxMx1) array
    pred_mask = pred_mask[0,:,:,0]

    pred_maskmulti=np.round(pred_mask*classes)
    pred_maskmulti=pred_maskmulti-1 #Classes: 0,1,2,3

    # Resize predicted mask
    pred_mask = cv2.resize(pred_maskmulti,(512,512), 
                          interpolation = cv2.INTER_AREA)
    
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
    # jack=jaccarindex(grtr_mask,pred_mask,2)
    # print(jack)
    
    # jack=jaccarindex(grtr_mask,pred_mask,3)
    # print(jack)

    
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

# Convert image in a tensor
def getprepareimg(im_array,scale):
    
    # Resize image (Input array to segmentation model)
    im_array=cv2.resize(im_array,(512//scale,512//scale), 
                        interpolation = cv2.INTER_AREA)
    
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
# def jaccarindex(grtr_mask,pred_mask,label):
    
#     grtr=np.zeros([512,512])
    
#     # Choose pixels corresponding to label
#     grtr[grtr_mask==label]=1
    
#     pred=np.zeros([512,512])
#     pred[pred_mask==label]=1
    
#     # Intersection
#     inter= np.sum(grtr*pred>=1)
#     #print(inter)
    
#     # Union
#     union=np.sum(grtr+pred>=1)
#     #print(union)
#     jaccard=inter/union
    
#     return jaccard

def jaccarindex(grtr_mask,pred_mask,label):
    
    grtr=np.zeros([512,512])
    
    # # Choose pixels corresponding to label
    grtr[grtr_mask==label]=1
    
    pred=np.zeros([512,512])
    pred[pred_mask==label]=1
    
    # # Intersection
    # inter= np.sum(grtr*pred>=1)
    # #print(inter)
    
    # # Union
    # union=np.sum(grtr+pred>=1)
    #print(union)
    
    true=np.float64(np.sum(grtr.flatten()*pred.flatten()))
    total=np.float64(np.sum(grtr.flatten()))
    
    # print(true)
    # print(total)
    
    jaccard=np.float64(true/total)
    
    return jaccard


#%%
#for i in range(len(listfiles)):ç
timeit()
from time import time
start_time = time()

for i in range(10000):
    print(i)
    
    

# Take the original function's return value.

# Calculate the elapsed time.
elapsed_time = time() - start_time 

#%%

im_name = 'P0002_Im0064.png'

path = 'C:/Users/Andres/Desktop/CovidImages/Testing/CT/CT/'
imCT=cv2.imread(path+im_name)

from skimage import io, color

a=io.imshow(color.label2rgb(pred_mask,imCT,
                          colors=[(0,0,0),(255,0,0),(0,0,255)],
                          alpha=0.0015, bg_label=0, bg_color=None))
plt.axis('off')

#https://stackoverflow.com/questions/57576686/how-to-overlay-segmented-image-on-top-of-main-image-in-python

