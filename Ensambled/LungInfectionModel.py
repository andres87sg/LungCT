# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 08:56:54 2021

@author: Andres
"""

from os import path
import numpy as np

import cv2 as cv
import os

from LungInfectionUtils import dcm_convert,getlungsegmentation,getprepareimgCNN
from LungInfectionUtils import lunginfectionsegmentation
from LungInfectionUtils import dcm_size,dcm_imresize
from LungInfectionConstantManager import WinLength,WinWidth


class LungInfectionModel():

    def __init__(self):
        
       a=0
        
    
    def run_preprocessing(self, dcm_img):
        
        [norm_img, ins_num] = dcm_convert(dcm_img,WinLength,WinWidth)    
        dcm_originalsize = dcm_size(dcm_img)
        
        return norm_img, ins_num, dcm_originalsize

    def run_prediction(self,norm_img,targetsize):
        inputCNNimg=getprepareimgCNN(norm_img)
        predictedmask = model_lungsegmentation.predict(inputCNNimg)
        lungsegmentationimg = getlungsegmentation(norm_img,predictedmask)
        pred_maskmulti=lunginfectionsegmentation(lungsegmentationimg,clf_model)
        pred_maskmulti_res=np.round(dcm_imresize(pred_maskmulti,
                                                 targetsize[0],
                                                 targetsize[1]))
        
        return pred_maskmulti_res

    def run_evaluation(self):
        pass

    def run_training(self):
        pass

#%%
import os
import pydicom as dicom
import matplotlib.pyplot as plt
import numpy as np
import pickle
import joblib


from tensorflow.keras.models import load_model

origpath = 'C:/Users/Andres/Desktop/imexhs/Lung/dicomimage/Torax/AF935CEE/'
model_lungsegmentation=load_model('C:/Users/Andres/Desktop/CTClassif/lng_seg_mdl.h5')
model_filename = 'KNNmodel.pkl'
clf_model = joblib.load(model_filename)

origpath = 'C:/Users/Andres/Desktop/imexhs/Lung/dicomimage/Torax/AF935CEE/'
listfiles = os.listdir(origpath)
i=40
dcmfilename = listfiles[i]


dcm_img = dicom.dcmread(origpath+dcmfilename)

mdl=LungInfectionModel()
[norm_img, ins_num,dcm_size]=mdl.run_preprocessing(dcm_img)
pred_mask=mdl.run_prediction(norm_img,targetsize=dcm_size)