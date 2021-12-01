
from os import path
import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv
import os
import pydicom as dicom

from LungInfectionUtils import dcm_convert,getprepareimgCNN
from LungInfectionUtils import dcm_size,dcm_imresize
from LungInfectionUtils import getsmoothmask,getRoImask
from LungInfectionConstantManager import WinLength,WinWidth,imgnormsize
from AbstractProducts import load_mdl_lungsegmentation,load_mdl_infsegmentation
from seg_utils import create_segmentations

#%%
class LungInfectionModel():

    def __init__(self,mdl1,mdl2):
        
        self.mdl1=mdl1
        self.mdl2=mdl2
        
    def run_preprocessing(self, dcm_img):
        
        # Preprocessing step:
        [norm_img, ins_num] = dcm_convert(dcm_img,WinLength,WinWidth)    
        dcm_originalsize = dcm_size(dcm_img)
        
        return norm_img, ins_num, dcm_originalsize
         
    def run_prediction(self,inputimg,targetsize):
        
        # Decrease scale may reduce computational cost
        scale=4

        # Normalize and add one dimension to array
        inputCNNimg=getprepareimgCNN(inputimg,scale)
        
        # Lung segmentation (lng)
        LngSegmentatioMask = self.mdl1.predict(inputCNNimg)
        
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3)) 

        # Lung segmentation mask must be an MxNx1 array
        cropmask = cv.morphologyEx(np.round(LngSegmentatioMask[0,:,:,0]), 
                                   cv.MORPH_CLOSE, kernel)        

        cropmask = cv.morphologyEx(np.round(cropmask), cv.MORPH_ERODE, kernel)

        # Function to smooth the segmentation mask
        LngSegmentatioMask = getsmoothmask(cropmask)
        
        # Usually the image size is 512x512
        CroppedLng = np.zeros((imgnormsize[0],imgnormsize[1],3))
        
        # Selecciona la RoI (Lng)
        for i in range(3):
            CroppedLng[:,:,i] = inputimg[:,:,i]*LngSegmentatioMask
            
        scale=4
        LngCNNimg=getprepareimgCNN(CroppedLng,scale)
        
        # Segmentación de vidrio esmerilado y consolidacion (ggo + cons)
        PredictedLngInfMask = self.mdl2.predict(LngCNNimg)
        PredictedLngInf = np.squeeze(PredictedLngInfMask,axis=0)
        PredictedLngInfMask = np.argmax(PredictedLngInf,axis=-1)

        # El tamaño es de 128x128 pix
        LngMask = np.zeros((np.shape(PredictedLngInfMask)[0],
                            np.shape(PredictedLngInfMask)[1]))
        
        # Mascara de segmentación de Pulmón 
        LngMask[PredictedLngInfMask!=2]=1
        LngMask = getsmoothmask(LngMask)
        
        # Mascara de segmentación ggo+con
        LngInfMask=np.uint16(PredictedLngInf[:,:,0]>0.5)
        LngInfMask = getsmoothmask(LngInfMask)

        # Selecciona la RoI (ggo+cons)        
        CroppedLngInf = LngInfMask*inputimg[:,:,0]
        
        # Lung
        lngRoIMask = getRoImask(CroppedLngInf,60,90)
        lngRoIMask = getsmoothmask(np.int16(lngRoIMask))
        
        # Ggo
        ggoRoIMask = getRoImask(CroppedLngInf,90,170)
        ggoRoIMask = getsmoothmask(np.int16(ggoRoIMask))
        
        # Cons
        conRoIMask = getRoImask(CroppedLngInf,170,255)
        conRoIMask = getsmoothmask(np.int16(conRoIMask))

        PredictedMaskMulti=LngMask.copy()
        
        for mask,label in zip((lngRoIMask,ggoRoIMask,conRoIMask),range(1,4)):
            PredictedMaskMulti[mask==1]=label
            
        PredictedMaskMulti=cv.resize(PredictedMaskMulti,
                                     (targetsize[1],targetsize[0]),
                                     interpolation = cv.INTER_AREA)

        return PredictedMaskMulti

    def run_evaluation(self):
        pass

    def run_training(self):
        pass

#%% 

"""
Prueba del modelo de segmentación de ggo + cons

"""

origpath = 'C:/Users/Andres/Desktop/SementacionesDicom/Patient4/'
listfiles = os.listdir(origpath)

mdl=LungInfectionModel(load_mdl_lungsegmentation(),load_mdl_infsegmentation())

segmentation=[]

from time import time
start_time = time() 

for i in range(len(listfiles)):
#for i in range(50,51):
    
    dcmfilename = listfiles[i]
    
    dcm_img = dicom.dcmread(origpath+dcmfilename)
    
    [norm_img, ins_num,dcm_originalsize]=mdl.run_preprocessing(dcm_img)
    pred_mask=mdl.run_prediction(norm_img,dcm_originalsize)
    
   
    imor_res=cv.resize(norm_img,(dcm_originalsize[1],dcm_originalsize[0]),
              interpolation = cv.INTER_AREA)
    
    
    
    plt.show()
    plt.subplot(1,2,1)
    plt.imshow(pred_mask,cmap='gray')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(imor_res,cmap='gray')
    plt.axis('off')
    print('Instace number: '+ str(i))
    
#     segmentation.append(pred_mask)

# segmentation=np.array(segmentation,dtype=np.uint8)

elapsed_time = time() - start_time 
print(elapsed_time)

minutes=np.round(np.floor(elapsed_time/60),0)
seconds=np.round((elapsed_time/60-minutes)*60,0)
print(str(minutes)+' minutes '+ str(seconds) + ' seconds ')

#%%
def extract_mask(mask, value):
    array_mask = mask.copy()
    array_mask = np.array(array_mask == value, dtype=np.uint8)
    return array_mask

lung_mask = extract_mask(segmentation, 1)
ground_glass_mask = extract_mask(segmentation, 2)
consolidation_mask = extract_mask(segmentation, 3)

#%%
    
metadata = "meta.json"

dest_folder='C:/Users/Andres/Desktop/SementacionesDicom/'

create_segmentations([lung_mask,ground_glass_mask,consolidation_mask],
                     metadata,origpath,dest_folder)
