import pydicom as dicom
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import os

import scipy as sp
from scipy.stats import skew,kurtosis
import pandas as pd

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

from LungInfectionConstantManager import imgnormsize, inputimgCNNscale, SEsize

def dcm_size(dcm_img):
    img_array = dcm_img.pixel_array
    #(dcm_heigth,dcm_length)=np.shape(img_array)
    dcm_size=np.shape(img_array)
    return dcm_size

def dcm_imresize(imginput,dcm_heigth,dcm_length):
    
     imgoutput=cv.resize(imginput,(dcm_length,dcm_heigth), 
                        interpolation = cv.INTER_NEAREST) 
     return imgoutput
    
def dcm_convert(dcm_img,WL,WW): 
   
    #dcm_img: Imagen Dicom
    #WW: WindowsWidth
    #WH: WindowsHeigth
       
    # Convert dicom image to pixel array
    img_array = dcm_img.pixel_array
    instance_number=dcm_img.InstanceNumber
    
    # Tranform matrix to HU
    hu_img = transform_to_hu(dcm_img,img_array)
    
    # Compute an image in a window (Lung Window)
    window_img = window_img_transf(hu_img,WL,WW)
    
    window_img = cv.cvtColor(window_img,cv.COLOR_GRAY2RGB)
    
    window_img=cv.resize(window_img,(imgnormsize[0],imgnormsize[1]), 
                        interpolation = cv.INTER_AREA) 
    

    return window_img, instance_number

def getprepareimgCNN(inputimg):
    imgscale = inputimgCNNscale
    # Prepare input image as input in the model 
    inputimg=cv.resize(inputimg,(imgnormsize[0]//imgscale,
                               imgnormsize[1]//imgscale),
                       interpolation = cv.INTER_AREA)
      
    # Input image normalization imnorm = im/max(im)
    norminputimg=inputimg/np.max(inputimg)
    
    # Adding one dimension to array
    nn_inputimg = np.expand_dims(norminputimg,axis=[0])
    
    return nn_inputimg

    
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

def GetPrepareImage(img_in):
    
    #im_array=img_in
    mask=np.zeros((np.shape(img_in)[0],np.shape(img_in)[1]))
    # mask[im_array>0]=1
    mask[img_in>0]=1
    kernel = np.ones((3, 3), np.uint8)
    cropmask = cv.erode(mask, kernel)
    img_out = img_in*cropmask
    
    return img_out

def getlungsegmentation(inputimg,predictedmask):

    predictedmaskresize = np.round( cv.resize(predictedmask[0,:,:,0],
                                    (imgnormsize[0],imgnormsize[1]),
                                    interpolation = cv.INTER_AREA)
                                    )
    
    kernel = np.ones((SEsize,SEsize), np.uint8)
    cropmask = cv.erode(predictedmaskresize, kernel)
    
    outputimg = inputimg[:,:,0]*cropmask
    return outputimg


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

def GetPrediction(featurematrix,clf_model):
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
    
    lunginfmask = lngmask+ggomask+conmask   
    
    lunginfmask[lunginfmask>3]=0
    
    resizedlunginfmask=cv.resize(lunginfmask,(512,512),interpolation = cv.INTER_AREA)
    # final_mask = cv.GaussianBlur(resizedlunginfmask, (0,0), sigmaX=1, sigmaY=1, borderType = cv.BORDER_DEFAULT)
    # final_mask = np.round(final_mask)
    final_mask=resizedlunginfmask.copy()
    return final_mask





