# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 08:25:49 2021

@author: Andres
"""
#%%

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

import nibabel as nib

path='C:/Users/Andres/Desktop/dicomimage/Mask_60920DAE.nii'
img = nib.load(path)
img = img.get_data()
print(img.shape)

(h, w) = img.shape[:2]

center = (w / 2, h / 2)

array=np.asarray(img)

[m,n,t]=np.shape(array)


#%%

for i in range(t):
    
    print(i)
    # List is flipped
    a=t-1-i
    
    #print('slide'+str(a))
    im1 = array[:,:,a] 
    
    path2='C:/Users/Andres/Desktop/imexhs/Lung/mask/'
    file='im'+str(t-a)+'.png'
    
        
    
    # Image rotation 90°, later flip 180°
    im2=np.rot90(im1)
    for i in range(2):
        im2=np.rot90(im2)
        
    im3=np.fliplr(im2)
    
    norm_img=cv2.normalize(im3, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    norm_img=np.uint8(norm_img)
    
    cv2.imwrite(path2+file, norm_img)
    
    #plt.figure()
    #plt.axis('off')
    #plt.imshow(im3,cmap="gray")
    #plt.title('slide'+str(t-a))
    

