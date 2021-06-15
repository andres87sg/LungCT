# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 08:33:05 2021

@author: Andres
"""

import os
import pydicom as dicom
import matplotlib.pyplot as plt
import numpy as np
import pickle
import joblib


from tensorflow.keras.models import load_model

from LungInfectionUtils import dcm_convert,getlungsegmentation,getprepareimgCNN
from LungInfectionUtils import lunginfectionsegmentation
from LungInfectionUtils import dcm_size

from LungInfectionConstantManager import WinLength,WinWidth

#%%

# patient_no = 8

origpath = 'C:/Users/Andres/Desktop/imexhs/Lung/dicomimage/Torax/AF935CEE/'
model_lungsegmentation=load_model('C:/Users/Andres/Desktop/CTClassif/lng_seg_mdl.h5')
model_filename = 'KNNmodel.pkl'
clf_model = joblib.load(model_filename)

listfiles = os.listdir(origpath)
i=10
dcmfilename = listfiles[i]
#%%

dcm_img = dicom.dcmread(origpath+dcmfilename)
dcmlen,dcmwid = dcm_size(dcm_img)

#Preprocessing
[norm_img, ins_num] = dcm_convert(dcm_img,WinLength,WinWidth) 

inputCNNimg=getprepareimgCNN(norm_img)
predictedmask = model_lungsegmentation.predict(inputCNNimg)

lungsegmentationimg = getlungsegmentation(norm_img,predictedmask)

pred_maskmulti=lunginfectionsegmentation(lungsegmentationimg ,clf_model)

plt.imshow(pred_maskmulti)




