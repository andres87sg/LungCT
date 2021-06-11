# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 08:33:05 2021

@author: Andres
"""

import os
import pydicom as dicom
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import load_model

from LungInfectionUtils import dcm_convert
from LungInfectionConstantManager import WinLength,WinWidth

# patient_no = 8

origpath = 'C:/Users/Andres/Desktop/imexhs/Lung/dicomimage/Torax/AF935CEE/'

listfiles = os.listdir(origpath)
i=10
dcmfilename = listfiles[i]

dcm_img = dicom.dcmread(origpath+dcmfilename)

model_lungsegmentation=load_model('C:/Users/Andres/Desktop/CTClassif/lng_seg_mdl.h5')
#%% Preprocessing

# Convert dcm image to image in Lung Window, output:
# normalized image, instance number


[norm_img, ins_num] = dcm_convert(dcm_img,WinLength,WinWidth) 

zz=get_prepareimgCNN(norm_img,nn_image_scale)

pred_mask = model_lungsegmentation.predict(norm_inputimg)


plt.imshow(norm_img)
plt.axis('off')

#%%


