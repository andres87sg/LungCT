"""
Created on Apr 14 2021
Modified on May 05 2021

@author: Andres Sandino
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
import nibabel as nib

#%%

# Patient number
patient_no = 25

# Origin path
path = 'C:/Users/Andres/Downloads/Estudio15.nii'

# Dest path
destpath = 'C:/Users/Andres/Desktop/imexhs/Lung/dicomimage/Torax/dcm2png/nuevos_casos_test/' 

# Load Image
img = nib.load(path)
img = img.get_fdata()

# Image format
imgformat = '.png'

#%%


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

#%%

def main():
    [m,n,numslices]=img.shape
    
    # Recuerde que está al revés la numeración
    
    for i in range(numslices):
        
        
        
        img_array = img[:,:,numslices-1-i]
        
        im_rot=img_array.copy()
        
        for _ in range(1):
            im_rot=np.rot90(im_rot)
               
        #a=numslices-1-i
        
    
        L=-500
        W=1500    
    
        im_out=window_img_transf(im_rot,L,W)
        
        norm_img=cv2.normalize(im_out, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        
        imnum=i+1
        
        filename='P'+str(patient_no).zfill(4)+'_Im'+str(imnum).zfill(4)+'_mask'+imgformat
        
        print(filename)
        
        cv2.imwrite(destpath+filename, norm_img)
    

#%%
if __name__ == "__main__":
    main()

print('The process has ended')
