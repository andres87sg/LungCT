
from os import path
import numpy as np
import matplotlib.pyplot as plt


import cv2 as cv
import os
import pydicom as dicom

from LungInfectionUtils import dcm_convert,getlungsegmentation,getprepareimgCNN
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
        
        [norm_img, ins_num] = dcm_convert(dcm_img,WinLength,WinWidth)    
        dcm_originalsize = dcm_size(dcm_img)
        
        return norm_img, ins_num, dcm_originalsize
    
    # def getsmoothmask(self,Mask):
        
    #     ResizedMask = cv.resize(Mask,(imgnormsize[0],imgnormsize[1]),
    #                            interpolation = cv.INTER_AREA)
    #     BlurredMask = cv.GaussianBlur(ResizedMask, (9,9), 5)
    #     ModifiedMask = np.uint16(BlurredMask>0.5)
        
    #     return ModifiedMask
    
    # def getRoImask(self,Mask,th1,th2):
        
    #     MaskTh1=Mask<th2
    #     MaskTh2=Mask>th1
        
    #     RoIMask = MaskTh1 & MaskTh2
        
    #     return RoIMask

        
    def run_prediction(self,norm_img,targetsize):
        
        inputCNNimg=getprepareimgCNN(norm_img,4)
        LngSegmentatioMask = self.mdl1.predict(inputCNNimg)
        
        LngSegmentatioMask = getsmoothmask(LngSegmentatioMask[0,:,:,0])
        LngSegmentatioImg = norm_img[:,:,0]*LngSegmentatioMask
        
        # TAmaño de la imagen 512x512
        LngSegmentatioImgRGB = np.zeros((imgnormsize[0],imgnormsize[1],3))
        
        for ind in range(3):
            LngSegmentatioImgRGB[:,:,ind] = LngSegmentatioImg/255
            
        scale=4
        LngCNNimg=getprepareimgCNN(LngSegmentatioImgRGB,scale)
        
        PredictedLngInfMask = self.mdl2.predict(LngCNNimg)
        
        PredictedLngInf = np.squeeze(PredictedLngInfMask,axis=0)
        PredictedLngInfMask = np.argmax(PredictedLngInf,axis=-1)

        # El tamaño es de 128x128 pix
        LngMask = np.zeros((np.shape(PredictedLngInfMask)[0],
                            np.shape(PredictedLngInfMask)[1]))
        
        # Mascara de segmentación de Pulmón 
        LngMask[PredictedLngInfMask!=2]=1

        LngInfMask=np.uint16(PredictedLngInf[:,:,0]>0.5)

        LngMask = getsmoothmask(LngMask)
        LngInfMask = getsmoothmask(LngInfMask)
        
        
        
        CroppedLngInf = LngInfMask*norm_img[:,:,0]
        
        lngMask = getRoImask(CroppedLngInf,60,90)
        ggoMask = getRoImask(CroppedLngInf,90,170)
        conMask = getRoImask(CroppedLngInf,170,255)

        lng=getsmoothmask(np.int16(lngMask))
        ggo=getsmoothmask(np.int16(ggoMask))
        con=getsmoothmask(np.int16(conMask))
        
        PredictedMaskMulti=LngMask.copy()
        
        for mask,label in zip((lng,ggo,con),range(1,4)):
            PredictedMaskMulti[mask==1]=label

                
        return PredictedMaskMulti

    def run_evaluation(self):
        pass

    def run_training(self):
        pass

#%% Prueba 
origpath = 'C:/Users/Andres/Desktop/SementacionesDicom/Patient2/'
#origpath = 'C:/Users/Andres/Desktop/imexhs/Lung/dicomimage/Torax/109BB5EC/'
listfiles = os.listdir(origpath)

mdl=LungInfectionModel(load_mdl_lungsegmentation(),load_mdl_infsegmentation())

segmentation=[]

from time import time
start_time = time() 



#for i in range(len(listfiles)):
for i in range(50,51):
    
    dcmfilename = listfiles[i]
    
    dcm_img = dicom.dcmread(origpath+dcmfilename)
    
    [norm_img, ins_num,dcm_originalsize]=mdl.run_preprocessing(dcm_img)
    pred_mask=mdl.run_prediction(norm_img,dcm_originalsize)
    
    im1=np.zeros((512,512,3))
    for p in range(3):
        #im1[:,:,p]=pred_mask/3
        im1[:,:,p]=pred_mask
    
    
    #cv.resize(pred_maskmulti_res,(targetsize[0],targetsize[1]),interpolation = cv.INTER_AREA)
    imout=cv.resize(im1,(dcm_originalsize[1],dcm_originalsize[0]),
              interpolation = cv.INTER_AREA)
    
    imor_res=cv.resize(norm_img,(dcm_originalsize[1],dcm_originalsize[0]),
              interpolation = cv.INTER_AREA)
    
    imout2=np.round(imout[:,:,0]*3)
    
    
    plt.show()
    plt.subplot(1,2,1)
    plt.imshow(imout2,cmap='gray')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(imor_res,cmap='gray')
    plt.axis('off')
    print('Instace number: '+ str(i))
    
#     segmentation.append(pred_mask)

# segmentation=np.array(segmentation,dtype=np.uint8)


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





#%%



