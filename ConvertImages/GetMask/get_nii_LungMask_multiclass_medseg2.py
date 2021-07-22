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
patient_no = 109

# bkg=0, class1=1, class2=2
# Include background as a class
classes=3

name = 'Estudio'+str(patient_no)

# Origin path and filename

# CT Image
path = 'C:/Users/Andres/Desktop/CTAnotadoMedSeg/CTnii/'
filename = name + '.nii'

# Mask Image
path_mask = 'C:/Users/Andres/Desktop/CTAnotadoMedSeg/Masknii/'
filename_mask = 'mask'+ name + '.nii'

# Dest path
destpath = 'C:/Users/Andres/Desktop/CTAnotadoMedSeg/CT_png/'
destpath_mask = 'C:/Users/Andres/Desktop/CTAnotadoMedSeg/Mask_png/'

# Load Images
img = nib.load(path+filename)
img = img.get_fdata()

img_mask = nib.load(path_mask+filename_mask)
img_mask = img_mask.get_fdata()

# Image format
imgformat = '.png'

# Convert *.nii in array
im_mask_array=np.array(img_mask)
im_array=np.array(img)
    
#%%
def nii2png(im_array,numslices,indslic,patient_no,destpath):
        # Recuerde que está al revés la numeración
    img_array = img[:,:,numslices-1-indslic]
    
    # Image rotation 
    im_rot=img_array
    
    for _ in range(3):
        im_rot=np.rot90(im_rot)
        
    im_rot=np.fliplr(im_rot)
           
    # Window Width (WW) and Window Lenght (WW) in CT
    L=-500
    W=1500    

    # Image transformed into CT window 
    im_out=window_img_transf(im_rot,L,W)
    
    # Image normalization Graylevel->[0,255]
    norm_img=cv2.normalize(im_out, None, 
                           alpha = 0, 
                           beta = 255, 
                           norm_type = cv2.NORM_MINMAX, 
                           dtype = cv2.CV_32F)
    
    # Image number
    im_num=indslic+1
            
    # Dest filename
    destfilename='P'+str(patient_no).zfill(4)+'_Im'+str(im_num).zfill(4)+imgformat
        
    # Save image
    cv2.imwrite(destpath+destfilename, norm_img)
    #print(destfilename)

def createmask(im_array,numslices,i,classes,patient_no,destpath_mask):    
    
    #print(i)
    # List is flipped
    a=numslices-1-i
    slide = im_array[:,:,a] 
      
    
    num_classes=3
    class_ggo=2
    class_con=3
    
    slide[slide==1]=np.int16(255*class_ggo/num_classes)
    slide[slide==2]=np.int16(255*class_con/num_classes)
    
        
    im_rot = slide
    
    # Image rotation 90°, later flip 180°
    for _ in range(3):
        im_rot=np.rot90(im_rot)
    #im_rot = slide

    #im_rot=slide
    #im_flip=im_rot
    #im_flip=im_rot
    im_flip=np.fliplr(im_rot)
    
    # Esto es para seleccionar solamente los cortes segmentados    
    sum_pix=np.sum(slide)
    #print(sum_pix)
    
    flag = 0    
    if sum_pix>0:    
    #Labeling files    
        filename='P'+str(patient_no).zfill(4)+'_Im'+str(numslices-a).zfill(4)+'_mask'+imgformat
        #print(filename)
        cv2.imwrite(destpath_mask+filename,im_flip)
        flag=1
    
    return flag

def window_img_transf(image, win_center, win_width):
    
    img_min = win_center - win_width // 2
    img_max = win_center + win_width // 2
    window_image = image
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    
    # Gray level bias correction -> from [0 to maxgraylevel]
    window_image_c=window_image+np.abs(img_min)
    
    # Image gray level [0 255]
    window_image_gl=np.uint16((window_image_c/np.max(window_image_c))*255)
    
    # Image in selected WW and WL
    window_image_gl=np.uint8(window_image_gl)
        
    return window_image_gl

#%% Main

[width,length,numslices]=np.shape(im_array)
#[m,n,t]=np.shape(im_array)

for ind in range(numslices):
#for ind in range(41,42):
    
    flag=createmask(im_mask_array,numslices,ind,classes,patient_no,destpath_mask)
    #print(str(flag))
    
    if flag == 1:
        
        # Convert image
        nii2png(im_array,numslices,ind,patient_no,destpath)
    
print('The process has ended')