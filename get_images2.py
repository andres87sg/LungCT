# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 14:55:07 2021

@author: Andres
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import segmentation, color

#%%

path_mask = 'C:/Users/Andres/Desktop/CTPulmon/LNG/Val/Mask/Mask_png/'
listfiles_mask = sorted(os.listdir(path_mask))

path = 'C:/Users/Andres/Desktop/CTPulmon/LNG/Val/CT/CT_png/'
listfiles = sorted(os.listdir(path))


destpath_mask = 'C:/Users/Andres/Desktop/CTPulmon/LNG/Val/Mask_M/Mask_png/'

#%%
    
    # List of files
#mask_im_name = mask_listfiles_mask[i]
#im_name = listfiles_mask[i]
#for i in range(len(listfiles_mask)):
for i in range(20,21):

    # path_mask='C:/Users/Andres/Desktop/imexhs/Lung/dicomimage/Torax/dcm2png/mask_train/'
    # filename='P0001_Im0'+str(i)+'_mask.png'
    # Groundtruth image (array)
    
    maskfilename = listfiles_mask[i]
    filename = listfiles[i]
    
    im_array=cv2.imread(path + listfiles[i]) 
    
    superpixels = convertim(im_array,2)
    plt.imshow(superpixels,cmap='gray')
    
    
    mask_array=cv2.imread(path_mask + maskfilename)   # Mask image
    print(listfiles_mask[i])
    
    scale=1
    mask_array=cv2.resize(mask_array,(512//scale,512//scale),interpolation = cv2.INTER_AREA)
    mask_array=np.round(mask_array)
    
    # Structiring element (Disk , r=5 )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    
      
    #open_img = cv2.morphologyEx(mask_array, cv2.MORPH_CLOSE, kernel) 
    #close_img = cv2.morphologyEx(open_img, cv2.MORPH_CLOSE, kernel)  
    
    close_img = cv2.morphologyEx(mask_array, cv2.MORPH_CLOSE, kernel)    
    # open_img = cv2.morphologyEx(close_img, cv2.MORPH_CLOSE, kernel)
    
    # Structiring element (Disk , r=5 )    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))    
    mask_array_modif = cv2.dilate(close_img,kernel,iterations = 1)
    mask_array_modif = cv2.morphologyEx(mask_array_modif, cv2.MORPH_OPEN, kernel)
    
    mask_array_modif2 = mask_array_modif.copy()
    mask_array_modif2 = np.uint8(mask_array_modif2)
    
    # plt.subplot(131)
    # plt.imshow(im_array,cmap='gray')
    # plt.axis('off')
    # plt.subplot(132)
    # plt.imshow(mask_array,cmap='gray')
    # plt.axis('off')
    # plt.subplot(133)
    # plt.imshow(mask_array_modif2,cmap='gray')
    # plt.axis('off')
    # plt.show()
    
    norm_img=cv2.normalize(mask_array_modif2, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    norm_img=np.uint8(norm_img)
    
    #cv2.imwrite(destpath_mask+maskfilename, norm_img)

#im_array = cv2.imread(path_mask+im_name)               # Graylevel image
#im_gray = im_array.copy()

#%%

def convertim(im_array,scale):

    im_reduc=cv2.resize(im_array,(512//scale,512//scale),interpolation = cv2.INTER_AREA)
    
    im_segments = segmentation.slic(im_array, compactness=20, n_segments=500)
    
    superpixels = color.label2rgb(im_segments, im_array, kind='avg')
    
    return superpixels


