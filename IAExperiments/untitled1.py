# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 10:06:19 2021

@author: Andres
"""

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
import os

import cv2 as cv
import numpy as np
# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required = True, help = "Path to the image")
# args = vars(ap.parse_args())
# # load the image and convert it to a floating point data type
# image = img_as_float(io.imread(args["image"]))
# loop over the number of segments

path = 'C:/Users/Andres/Desktop/CovidImages2/CTMedSeg2/'
pathmask = 'C:/Users/Andres/Desktop/CovidImages2/MaskMedSeg2/'




listfiles = os.listdir(path)
listfilesmask = os.listdir(pathmask)


i=30

im_name = listfiles[i] # Gray level
im_namemask = listfilesmask[i] # Segmentation mask

# Graylevel image (array)
im_or=cv.imread(path+im_name)
im_array=im_or[:,:,0]
grtr_mask=cv.imread(pathmask+im_namemask)

# mask=np.int16(grtr_mask[:,:,0]>0)

# kernel = np.ones((10, 10), np.uint8)
# cropmask = cv2.erode(mask, kernel)


# im_or=im_or[:,:,0]*cropmask
# grtr_mask = grtr_mask[:,:,0]*cropmask

plt.imshow(im_or)

#%%
# kk=np.zeros((512,512,3))

# kk[:,:,0]=im_or
# kk[:,:,1]=im_or
# kk[:,:,2]=im_or
mask=np.zeros((512,512))

kk=im_or.copy()

for numSegments in (100, 200, 900):
	# apply SLIC and extract (approximately) the supplied number
	# of segments
	segments = slic(kk, n_segments = numSegments, sigma = 3)
	# show the output of SLIC
	fig = plt.figure("Superpixels -- %d segments" % (numSegments))
	ax = fig.add_subplot(1, 1, 1)
	ax.imshow(mark_boundaries(kk, segments))
	plt.axis("off")
# show the plots
plt.show()

#%%
im_or2=im_or[:,:,0]


for i in range(np.max(np.unique(segments))):
    
    prom=np.mean(im_or[segments==i])
    #print(prom)
    
    if prom>20:
        mask[segments==i]=np.int16(i)
        
plt.imshow(mask,cmap='jet')
    
SuperPixList = np.int16(np.unique(mask))

#%%
for SPind in range(1,len(SuperPixList)):
    print(np.mean(im_or2[mask==SuperPixList[SPind]]))


# mask==listica[2]


# zzz=np.int16(pp[:,:,0])*im_or2

# plt.imshow(zzz)

