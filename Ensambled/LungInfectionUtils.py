# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 07:48:10 2021

@author: Andres
"""

import pydicom as dicom
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import os

from LungInfectionConstantManager import imgnormsize, nn_image_scale

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
    
    window_img=cv.resize(norm_img,(imgnormsize[0],imgnormsize[1]), 
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

def get_prepareimgCNN(inputimg,nn_image_scale):
    #scale_factor = 4
    # Prepare input image as input in the model 
    inputimg=cv.resize(inputimg,(imgnormsize[0]//nn_image_scale,
                               imgnormsize[1]//nn_image_scale),
                       interpolation = cv.INTER_AREA)
      
    # Input image normalization imnorm = im/max(im)
    norminputimg=inputimg/np.max(inputimg)
    
    # Adding one dimension to array
    nn_inputimg = np.expand_dims(norminputimg,axis=[0])
    
    return nn_inputimg
    


