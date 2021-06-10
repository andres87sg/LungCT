"""
Created on  12/05/2021
Modified on 12/05/2021
@author: Andres Sandino

"""
#%%
# Install gdcm 
import pydicom as dicom
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import os
from time import time

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from skimage import io, color

from tensorflow.keras import Input,layers, models
from tensorflow.keras.layers import Conv2DTranspose,Dropout,Conv2D,BatchNormalization, Activation,MaxPooling2D
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

#%%
import pydicom as dicom
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import os

#%%

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


#%% Function

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


def dcm_convert(dcm_dir,dcmfilename,WL,WW): 
   
    img_path = dcm_dir+dcmfilename

    # Dicom image read
    dcm_img = dicom.dcmread(img_path)   # Read dicom
    instance_number=dcm_img.InstanceNumber # Dicom instace number
    
    # Convert dicom image to pixel array
    img_array = dcm_img.pixel_array
    
    # Tranform matrix to HU
    hu_img = transform_to_hu(dcm_img,img_array)
    
    # Compute an image in a window (Lung Window)
    window_img = window_img_transf(hu_img,WL,WW)
    
    #w,l = np.shape(window_img)

    #window_img = cv.resize(window_img,(w,l), 
    #                    interpolation = cv.INTER_AREA)
    
    window_img= cv.cvtColor(window_img,cv.COLOR_GRAY2RGB)

    return window_img, instance_number
    
    #dcm_image.WindowCenter
    #dcm_image.WindowWidth

# Transform dcm to HU (Hounsfield Units)
def transform_to_hu(medical_image, image):
    intercept = medical_image.RescaleIntercept
    #print('intercept')
    #print(str(intercept))
    slope = medical_image.RescaleSlope
    hu_image = image * slope + intercept
    return hu_image

# Transform HU image to Window Image
def window_img_transf(image, win_center, win_width):
    
    img_min = win_center - win_width // 2
    img_max = win_center + win_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    
    # Gray level bias correction -> from [0 to maxgraylevel]
    window_image_c=window_image+np.abs(img_min)
    
    # Image gray level [0 255]
    window_image_gl=np.uint16((window_image_c/np.max(window_image_c))*255)
    window_image_gl=np.uint8(window_image_gl)
        
    return window_image_gl


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

def predmask(im_or,roi,subsample,predicted_label,label):
    
    #subsample=1   
    predcoordy=roi[0][::subsample][predicted_label==label]
    predcoordx=roi[1][::subsample][predicted_label==label]
    predictedmask=np.zeros((np.shape(im_or)[0],np.shape(im_or)[1]))
    predictedmask[predcoordy,predcoordx]=label+1
    
    return predictedmask


def lunginfectionsegmentation(im_or,clf_model):

    
    
    segmented_image=kmeanscluster(im_or)    
    clusterlabels = np.unique(segmented_image)  
    #print(len(clusterlabels))
    
    if len(clusterlabels)>1:
    
        lungmask = segmented_image==clusterlabels[1]    
        
        if len(clusterlabels)>2:
            lunginfmask=np.int16(segmented_image==clusterlabels[2])
            
            # kernel = np.ones((3,3), np.uint8)
            # imopen = cv.morphologyEx(lunginfmask, cv.MORPH_OPEN, kernel)    
            # lunginfmask = imopen.copy()
            
            # Region of interest
            roi = np.where(lunginfmask == 1)
            
            subsample=3
            
            featurematrix=feature_extraction(im_or,roi,subsample)
            scaler = preprocessing.StandardScaler().fit(featurematrix)
            featurematrix_norm=scaler.transform(featurematrix)
            
            predicted_label = clf_model.predict(featurematrix_norm)
            
            ggomask=predmask(im_or,roi,subsample,predicted_label,1)
            conmask=predmask(im_or,roi,subsample,predicted_label,2)
              
            kernel = np.ones((subsample,subsample), np.uint8)
            ggomask_close = cv.morphologyEx(ggomask, cv.MORPH_CLOSE, kernel)   
            conmask_close = cv.morphologyEx(conmask, cv.MORPH_CLOSE, kernel)   
            
            
            #lunginfmask = conmask2+ggomask2+lungmask
            lunginfmask = conmask_close+ggomask_close
            lunginfmask[lunginfmask>3]=0
        
        else:
           lunginfmask=segmented_image.copy()
           lunginfmask[lunginfmask>0]=1
    
    else:
        lunginfmask=segmented_image.copy()
        lunginfmask[lunginfmask>0]=1

    return lunginfmask


def regionsegmentation(im_or,clf_model):
    
    im_array=im_or[:,:,0]
    
    mask=np.zeros((np.shape(im_or)[0],np.shape(im_or)[1]))
    
    mask[im_array>0]=1
    
    kernel = np.ones((3, 3), np.uint8)
    cropmask = cv.erode(mask, kernel)
    
    im_or = im_or[:,:,0]*cropmask

    
    final_mask=lunginfectionsegmentation(im_or,clf_model)
    
    return final_mask

#%%

#case='22474FA3'
#case='22474FA3'
patient_no = 8

origpath = 'C:/Users/Andres/Desktop/imexhs/Lung/dicomimage/Torax/AF935CEE/'

#origpath = 'C:/Users/Andres/Desktop/imexhs/Lung/dicomimage/Torax/'+ case +'/' 
listfiles = os.listdir(origpath)
#patient = 'patient1a'

#%%
#model_multiclass=load_model('C:/Users/Andres/Desktop/CTClassif/LungInf_SF2_Filt64_Python.h5')
model_lungsegmentation=load_model('C:/Users/Andres/Desktop/CTClassif/lng_seg_mdl.h5')



#%%

pp=[]


# Start counting
start_time = time() 

dcmimages=[]
maskimages=[]

model_filename = 'KNNmodel.pkl'
clf_model = joblib.load(model_filename)

#for i in range(len(listfiles)):
for i in range(100,101):

#for i in range(284,285):
    
    """
    Step 1 -> Prepare dcm image into Lung windows (WW=-500, WW=1500)
    """
    
    dcmfilename = listfiles[i]
    
    # Convert .dcm file to graylevel image in window
    # Lung window (WL=-500, WW=1500)
   
    WL_WW=[-500,1500]
    
    # Convert dcm image to image in Lung Window, output:
    # normalized image, instance number
    [norm_img, ins_num] = dcm_convert(origpath,dcmfilename,WL_WW[0],WL_WW[1])  
    [wn,ln,ch]=norm_img.shape
    
    # Resize Image
    im_dcm=cv.resize(norm_img,(512,512), 
                        interpolation = cv.INTER_AREA)    
    # Append CT slices
    #dcmimages.append(im_dcm)
    dcmimages.append(norm_img[:,:,0])

    """
    Step 2 -> Lungs segmentation
    """
    scale = 4
    # Prepare input image as input in the model 
    inputimg=cv.resize(im_dcm,(512//scale,512//scale), 
                        interpolation = cv.INTER_AREA)
      
    # Input image normalization imnorm = im/max(im)
    norminputimg=inputimg/np.max(inputimg)
    
    # Adding one dimension to array
    norm_inputimg = np.expand_dims(norminputimg,axis=[0])
    
    # Predicted mask
    pred_mask = model_lungsegmentation.predict(norm_inputimg)
    
    # Image mask as (NxMx1) array
    pred_mask = pred_mask[0,:,:,0]
    #pred_mask[pred_mask==0]=1
    
    # Resize predicted mask
    pred_mask = np.round(cv.resize(pred_mask,(512,512), 
                       interpolation = cv.INTER_AREA))
    
    # Lungs segmentation
    segmentedlungs=im_dcm.copy()
    segmentedlungs[pred_mask==0]=0
    
    """
    Step 3 -> Segment consolidation and ground-glass opacity segmentation
    """
    scale = 4
    
    segmentedlungs=cv.resize(segmentedlungs,(512//scale,512//scale), 
                    interpolation = cv.INTER_AREA)

    pred_maskmulti=regionsegmentation(segmentedlungs,clf_model)
    
    # Predicted mask : color and gray
    col_predmask,gray_predmask = getcolormask(pred_maskmulti)
    
    # Resize segmentation mask (wn,ln from original image [norm_img])
    pred_colmaskfinal = cv.resize(col_predmask,(ln,wn), 
                           interpolation = cv.INTER_AREA)
    
    pred_graymaskfinal = cv.resize(gray_predmask,(ln,wn), 
                       interpolation = cv.INTER_AREA)
    
    maskimages.append(pred_graymaskfinal)
    
# Finish counting
elapsed_time = time() - start_time 
print(elapsed_time)

minutes=np.round(np.floor(elapsed_time/60),0)
seconds=np.round((elapsed_time/60-minutes)*60,0)
print(str(minutes)+' minutes '+ str(seconds) + ' seconds ')



#%% Show results
    
from PIL import Image, ImageDraw
from skimage import io, color



for i in range(1,len(dcmimages)-1):
#for i in range(1,90):
    
    # overlapimg=color.label2rgb(maskimages[i],dcmimages[i],
    #                       colors=[(0,0,0),(255,0,0),(0,0,255)],
    #                       alpha=0.0015, bg_label=0, bg_color=None)

    overlapimg=color.label2rgb(maskimages[i],dcmimages[i],
                          colors=[(0,0,0),(0,0,255),(255,0,0)],
                          alpha=0.0015, bg_label=0, bg_color=None)    
    
        
    overlapimg[:,:,0]=overlapimg[:,:,0]*255
    overlapimg[:,:,1]=overlapimg[:,:,1]*255
    overlapimg[:,:,2]=overlapimg[:,:,2]*255
    
    
    im1 = cv.imwrite("over.jpg",overlapimg)
    im2 = Image.open('over.jpg')
    
    plt.show()
    plt.imshow(im2)
    pp.append(im2)
    plt.axis("off")

    # Color label (black, green, red, blue)
    # colorlabel=([0,0,0],[0,255,0],[255,0,0],[0,0,255]) # Colors

#%%

# pp[0].save('C:/Users/Andres/Desktop/109BB5EC.gif',
#                save_all=True, append_images=pp[1:], optimize=False, duration=200, loop=0)



