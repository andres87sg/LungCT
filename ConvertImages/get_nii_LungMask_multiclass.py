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
patient_no = 41

# bkg=0, class1=1, class2=2
# Include background as a class
classes=3

# Origin path and filename
path = 'C:/Users/Andres/Desktop/CTAnotado/resultados/Dr Vargas/'
filename = 'maskEstudio41.nii'

# Dest path
destpath = 'C:/Users/Andres/Desktop/CovidImages/Mask/' 

# Load Image
img = nib.load(path+filename)
img = img.get_fdata()

# Image format
imgformat = '.png'

im_array=np.array(img)

#%%
[width,length,numslices]=np.shape(im_array)
[m,n,t]=np.shape(im_array)

for i in range(numslices):
#for i in range(39,42):
    
    #print(i)
    # List is flipped
    a=numslices-1-i
    slide = im_array[:,:,a] 
    
    for ind_class in range(classes):         
        num_class=np.round(255/(classes-1)*ind_class)        
        slide[slide==ind_class]=num_class

    # Image rotation 90°, later flip 180°
    im_rot=np.rot90(slide)
    # for i in range(4):
    #     im2=np.rot90(im2)
    
    im_flip=np.fliplr(im_rot)
    
    sum_pix=np.sum(slide)
    #print(sum_pix)
    
    if sum_pix>0:    
    #Labeling files    
        filename='P'+str(patient_no).zfill(4)+'_Im'+str(numslices-a).zfill(4)+'_mask'+imgformat
        print(filename)
        cv2.imwrite(destpath+filename,im_flip)


