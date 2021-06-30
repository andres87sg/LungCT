
from os import path
import numpy as np
import matplotlib.pyplot as plt


import cv2 as cv
import os
import pydicom as dicom

from LungInfectionUtils import dcm_size, dcm_imresize, dcm_convert, getprepareimgCNN, getlungsegmentation
from LungInfectionUtils import transform_to_hu, window_img_transf
from LungInfectionUtils import GetPrepareImage, GetFeatureExtraction
from LungInfectionUtils import GetClusteredMask, GetPrediction, GetPredictedMask,GetLungInfSegmentation

from LungInfectionConstantManager import WinLength,WinWidth
from AbstractProducts import load_mdl_lungsegmentation,load_mdl_infsegmentation

#%%

class LungInfectionModel():

    def __init__(self,mdl1,mdl2):
        
        self.mdl1=mdl1
        self.mdl2=mdl2
      
    
    def run_preprocessing(self, dcm_img):
        
        [norm_img, ins_num] = dcm_convert(dcm_img,WinLength,WinWidth)    
        dcm_originalsize = dcm_size(dcm_img)
        
        return norm_img, ins_num, dcm_originalsize
    
    def run_prediction(self,norm_img,targetsize):
        
        inputCNNimg=getprepareimgCNN(norm_img)
        predictedmask = self.mdl1.predict(inputCNNimg)
    
        lungsegmentationimg = getlungsegmentation(norm_img,predictedmask)
        aa = GetPrepareImage(lungsegmentationimg)
        
        
        scl_img_or,scl_segm_img=GetClusteredMask(aa,scale=2)
        featurematrix = GetFeatureExtraction(scl_img_or,scl_segm_img)
        
        if featurematrix.size!=0:
        
            predicted_label = GetPrediction(featurematrix,self.mdl2)
            pred_maskmulti = GetLungInfSegmentation(scl_img_or,predicted_label)
            
            pred_maskmulti_res=np.round(dcm_imresize(pred_maskmulti,
                                                      targetsize[0],
                                                      targetsize[1]))
        else:
            pred_maskmulti_res=np.zeros((targetsize[0],targetsize[1]))
            
            
        
        return lungsegmentationimg

    def run_evaluation(self):
        pass

    def run_training(self):
        pass

#%% Prueba 

origpath = 'C:/Users/Andres/Desktop/imexhs/Lung/dicomimage/Torax/AF935CEE/'
listfiles = os.listdir(origpath)

mdl=LungInfectionModel(load_mdl_lungsegmentation(),load_mdl_infsegmentation())

from time import time
start_time = time()

for i in range(20,21):
    dcmfilename = listfiles[i]
    
    dcm_img = dicom.dcmread(origpath+dcmfilename)
    
    [norm_img, ins_num,dcm_originalsize]=mdl.run_preprocessing(dcm_img)
    pred_mask2=mdl.run_prediction(norm_img,dcm_originalsize)
    
    plt.figure()
    plt.title('Image No '+ str(i))
    plt.imshow(pred_mask,cmap='gray')
    plt.axis('off')
    
elapsed_time = time() - start_time 
print(elapsed_time)

minutes=np.round(np.floor(elapsed_time/60),0)
seconds=np.round((elapsed_time/60-minutes)*60,0)
print(str(minutes)+' minutes '+ str(seconds) + ' seconds ')