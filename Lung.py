# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 07:45:07 2021

@author: Andres Sandino

Esta versión es con VTK
"""
#%%
 
import pydicom as dicom
#import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

#%% Main program

origpath = 'C:/Users/Andres/Desktop/dicomimage/Patient3/' 
listfiles = os.listdir(origpath)
destpath = 'C:/Users/Andres/Desktop/imexhs/Lung/converted3/'

for i in range(len(listfiles)):
#for i in [1]:
#    print(listfiles[i])
    dcmfilename = listfiles[i]
    norm_img = dcm_convert(origpath,dcmfilename)    
    #plt.figure()
    #plt.imshow(norm_img,cmap='gray')
    #plt.axis('off')
    imgformat = '.png'
    image_dest = destpath + dcmfilename + imgformat
    cv2.imwrite(image_dest, norm_img)
    
    
#%%
# Open Image
def dcm_convert(dcm_dir,dcmfilename):
   
    #filename = '60920DF2'
    img_path = dcm_dir+dcmfilename
    #print(img_path)
    
    
    dcm_img = dicom.dcmread(img_path)
    
    # Convert dicom image to pixel array
    img_array = dcm_img.pixel_array
    #print('img Lista')
    
    # Window (Lung Cancer L=-500, W=1500)
    WL=-500
    WW=1500
    
    # Tranform matrix to HU
    hu_img = transform_to_hu(dcm_img,img_array)
    
    # Lung window image
    lungwin_img = window_image(hu_img,WL,WW)
    
    # Normalizing image
    norm_img=cv2.normalize(lungwin_img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    norm_img=np.uint8(norm_img)
    
    return norm_img
    
#dcm_image.WindowCenter
#dcm_image.WindowWidth


def transform_to_hu(medical_image, image):
    intercept = medical_image.RescaleIntercept
    slope = medical_image.RescaleSlope
    hu_image = image * slope + intercept
    return hu_image

def window_image(image, window_center, window_width):
    img_min = window_center - window_width // 2
    #print('min '+str(img_min))
    img_max = window_center + window_width // 2
    #print('max '+str(img_max))
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    return window_image
    
