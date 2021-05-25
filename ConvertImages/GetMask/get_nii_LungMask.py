# -*- coding: utf-8 -*-
"""
Created on Apr 7 2021
Modified on May 05 2021

@author: Andres Sandino

Convert "nii" image format in "png" in Lung WW=-500,WL=1500
"""
#%%

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

import nibabel as nib

# Patient number
patient_no = 1

# Origin path and filename
path = 'C:/Users/Andres/Desktop/CTAnotado/resultados/Dr Alvarado/'
filename = 'maskEstudio1.nii'

# Dest path
destpath = 'C:/Users/Andres/Desktop/CovidImages/Mask/' 

# Load Image
img = nib.load(path+filename)
img = img.get_fdata()

# Image format
imgformat = '.png'

array=np.asarray(img)

#%%
[width,length,numslices]=np.shape(array)
[m,n,t]=np.shape(array)


#for i in range(numslices):
for i in range(35,40):
    
    #print(i)
    # List is flipped
    a=numslices-1-i
    slide = array[:,:,a] 
    
 
    #Labeling files    
    filename='P'+str(patient_no).zfill(4)+'_Im'+str(numslices-a).zfill(4)+'_mask'+imgformat
    print(filename)
    
    # Image rotation 90°, later flip 180°
    im2=np.rot90(slide)
    # for i in range(4):
    #     im2=np.rot90(im2)
    
    i#m3=im2.copy()    
    im3=np.fliplr(im2)
    
    norm_img=cv2.normalize(im3, None, alpha = 0, 
                           beta = 255, 
                           norm_type = cv2.NORM_MINMAX, 
                           dtype = cv2.CV_32F)
    
    norm_img=np.uint8(norm_img)
    
    cv2.imwrite(destpath+filename, norm_img)
    
    #plt.figure()
    #plt.axis('off')
    #plt.imshow(norm_img,cmap="gray")
    #plt.title('slide'+str(t-a))
    

