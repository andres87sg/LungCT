# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 08:15:42 2021

@author: Andres

Convierte la máscara ggo y cons (infec) en una máscara unificada
en el que bkg=0 e infec=1

"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


import os
from os.path import isfile, join
import tqdm


path = 'C:/Users/Andres/Desktop/CovidImages2/Training/CT2/CT/'
pathmask = 'C:/Users/Andres/Desktop/CovidImages2/Training/Mask/Mask/'

destpath = 'C:/Users/Andres/Desktop/CovidImages2/Training/Mask2/'

listfiles = os.listdir(path)
listfilesmask = os.listdir(pathmask)
statslist=[]

num_classes = 3
maxgraylevel = 255
classlabel_lung = 1

for i in tqdm.tqdm(range(len(listfiles))):
   
    im_name = listfiles[i] # Gray level
    im_namemask = listfilesmask[i] # Segmentation mask
    
    im_or=cv.imread(join(path,im_name))
    im_array=im_or[:,:,0]
    grtr_mask=cv.imread(join(pathmask,im_namemask))
    
    
    grtr_mask2=np.int16((grtr_mask/maxgraylevel)*num_classes)
    
    grtr_mask2[grtr_mask2>1]=255
    grtr_mask2[grtr_mask2==1]=127
    grtr_mask2[grtr_mask2<classlabel_lung]=0
    
    cv.imwrite(destpath+im_name, grtr_mask2)
    
#%%

path = 'C:/Users/Andres/Desktop/CovidImages2/Testing/CT2/CT/'
pathmask = 'C:/Users/Andres/Desktop/LungInfDataset/Testing/Mask2/Mask/'

#destpath = 'C:/Users/Andres/Desktop/CovidImages2/Testing/Mask2/'

listfiles = os.listdir(path)
listfilesmask = os.listdir(pathmask)
statslist=[]

num_classes = 3
maxgraylevel = 255
classlabel_lung = 1

for i in tqdm.tqdm(range(len(listfiles))):
    im_name = listfiles[i] # Gray level
    im_namemask = listfilesmask[i] # Segmentation mask
    im_or=cv.imread(join(path,im_name))
    im_array=im_or[:,:,0]
    grtr_mask=cv.imread(join(pathmask,im_namemask))
    
    print(np.unique(grtr_mask))
    