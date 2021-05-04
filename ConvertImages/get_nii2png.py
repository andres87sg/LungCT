"""
Created on Apr 14 2021
Modified on May 05 2021

@author: Andres Sandino

Convert "nii" image format in "png" in Lung WW=-500,WL=1500
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
import nibabel as nib

#%%

# Patient number
patient_no = 1

# Origin path and filename
path = 'C:/Users/Andres/Desktop/CTAnotado/imagenes/Dr Alvarado/'
filename = 'Estudio1.nii'

# Dest path
destpath = 'C:/Users/Andres/Desktop/CovidImages/CT/' 

# Load Image
img = nib.load(path+filename)
img = img.get_fdata()

# Image format
imgformat = '.png'

#%% Main (Convert nii images in png)

def main():
    
    # Size width,lenght and number of slices 
    [width,length,numslices]=img.shape
    
    # Every slide is converted in png
    for indslic in range(numslices):
         
        # Recuerde que está al revés la numeración
        img_array = img[:,:,numslices-1-indslic]
        
        # Image rotation 
        im_rot=img_array
        for _ in range(1):
            im_rot=np.rot90(im_rot)
               
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
    
#%%

# Transform HU image to Window Image
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

#%%
if __name__ == "__main__":
    main()

print('The process has ended')
