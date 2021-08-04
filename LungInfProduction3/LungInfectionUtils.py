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

def getprepareimgCNN(inputimg,imgscale):
     
    inputimg=cv.resize(inputimg,(imgnormsize[0]//imgscale,
                            imgnormsize[1]//imgscale),
                    interpolation = cv.INTER_NEAREST)
      
    # Input image normalization imnorm = im/max(im)
    norminputimg=inputimg/255
    
    # Adding one dimension to array
    nn_inputimg = np.expand_dims(norminputimg,axis=[0])
    
    return nn_inputimg
    
def getlungsegmentation(inputimg,predictedmask):

    predictedmaskresize = np.round( cv.resize(predictedmask[0,:,:,0],
                                    (imgnormsize[0],imgnormsize[1]),
                                    interpolation = cv.INTER_AREA)
                                    )
    
    kernel = np.ones((SEsize,SEsize), np.uint8)
    cropmask = cv.erode(predictedmaskresize, kernel)
    
    #outputimg = inputimg[:,:,0]*cropmask
    return outputimg

def getsmoothmask(Mask):
    
    ResizedMask = cv.resize(Mask,(imgnormsize[0],imgnormsize[1]),
                           interpolation = cv.INTER_AREA)
    BlurredMask = cv.GaussianBlur(ResizedMask, (9,9), 5)
    ModifiedMask = np.uint16(BlurredMask>0.5)
    
    return ModifiedMask

def getsmoothmask2(Mask,ksize,sigma,th):
    
    ResizedMask = cv.resize(Mask,(imgnormsize[0],imgnormsize[1]),
                           interpolation = cv.INTER_AREA)
    BlurredMask = cv.GaussianBlur(ResizedMask, ksize, sigma)
    ModifiedMask = np.uint16(BlurredMask>th)
    
    return ModifiedMask

def getRoImask(Mask,th1,th2):
    
    MaskTh1=Mask<th2
    MaskTh2=Mask>th1
    
    RoIMask = MaskTh1 & MaskTh2
    
    return RoIMask

