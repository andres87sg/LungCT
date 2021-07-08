
from os import path
import numpy as np
import matplotlib.pyplot as plt


import cv2 as cv
import os
import pydicom as dicom

from LungInfectionUtils import dcm_convert,getlungsegmentation,getprepareimgCNN
from LungInfectionUtils import lunginfectionsegmentation
from LungInfectionUtils import dcm_size,dcm_imresize
from LungInfectionConstantManager import WinLength,WinWidth
from AbstractProducts import load_mdl_lungsegmentation,load_mdl_infsegmentation
from seg_utils import create_segmentations

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
        pred_maskmulti=lunginfectionsegmentation(lungsegmentationimg,
                                                 self.mdl2)
        
        
        pred_maskmulti_res=np.round(dcm_imresize(pred_maskmulti,
                                                  targetsize[0],
                                                  targetsize[1]))
        
        lngmask=np.round(dcm_imresize(predictedmask[0,:,:,0],
                                          targetsize[0],
                                          targetsize[1]))
        
        ggomask=np.int16(pred_maskmulti_res==2)
        conmask=np.int16(pred_maskmulti_res==3)
        
        kernel = np.ones((10,10), np.uint8)
        
        ggomask2=cv.morphologyEx(ggomask, cv.MORPH_OPEN, kernel)
        conmask2=cv.morphologyEx(conmask, cv.MORPH_OPEN, kernel)
        
        pred_maskmulti_res=lngmask+ggomask2+2*conmask2
                
        return pred_maskmulti_res

    def run_evaluation(self):
        pass

    def run_training(self):
        pass

#%% Prueba 
origpath = 'C:/Users/Andres/Desktop/SementacionesDicom/Patient/'
#origpath = 'C:/Users/Andres/Desktop/imexhs/Lung/dicomimage/Torax/109BB5EC/'
listfiles = os.listdir(origpath)

mdl=LungInfectionModel(load_mdl_lungsegmentation(),load_mdl_infsegmentation())

segmentation=[]

from time import time
start_time = time() 

for i in range(len(listfiles)):
    dcmfilename = listfiles[i]
    
    dcm_img = dicom.dcmread(origpath+dcmfilename)
    
    [norm_img, ins_num,dcm_originalsize]=mdl.run_preprocessing(dcm_img)
    pred_mask=mdl.run_prediction(norm_img,dcm_originalsize)
    #plt.figure()
    #plt.imshow(pred_mask,cmap='gray')
    #plt.axis('off')
    
    
    segmentation.append(pred_mask)

segmentation=np.array(segmentation,dtype=np.uint8)


#%%
def extract_mask(mask, value):
    array_mask = mask.copy()
    array_mask = np.array(array_mask == value, dtype=np.uint8)
    return array_mask

lung_mask = extract_mask(segmentation, 1)
ground_glass_mask = extract_mask(segmentation, 2)
consolidation_mask = extract_mask(segmentation, 3)

#%%
    
elapsed_time = time() - start_time 
print(elapsed_time)

minutes=np.round(np.floor(elapsed_time/60),0)
seconds=np.round((elapsed_time/60-minutes)*60,0)
print(str(minutes)+' minutes '+ str(seconds) + ' seconds ')


#%%
metadata = "meta.json"

dest_folder='C:/Users/Andres/Desktop/SementacionesDicom/'

#%%

create_segmentations([lung_mask,ground_glass_mask,consolidation_mask],metadata,origpath,dest_folder)



