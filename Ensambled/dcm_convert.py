# -*- coding: utf-8 -*-
"""
Created on Wed May 12 15:37:00 2021

@author: Andres
"""

#%% Define functions

def principal():
    
    import pydicom as dicom
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np
    import os
    
    
    # Convert .dcm file to graylevel image in window
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