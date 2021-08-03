
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
    
    def getsmoothmask(self,pred_mask):
        
        pred_mask2 = cv.resize(pred_mask,(512,512),interpolation = cv.INTER_AREA)
        
        # pred_mask3 = np.uint16(pred_mask2>=0.5)
        
        MASK3 = cv.GaussianBlur(pred_mask2, (9,9), 5)
        
        pred_mask4 = np.uint16(MASK3>=0.5)
        
        return pred_mask4

        
    def run_prediction(self,norm_img,targetsize):
        
        inputCNNimg=getprepareimgCNN(norm_img,4)
        predictedmask = self.mdl1.predict(inputCNNimg)
        lungsegmentationimg = getlungsegmentation(norm_img,predictedmask)
        im1=np.zeros((512,512,3))
        
        for i in range(3):
            im1[:,:,i]=lungsegmentationimg/255
        
        scale=4
        lungseg=getprepareimgCNN(im1,scale)
        
        
        
        predictedlunginfmask = self.mdl2.predict(lungseg)
        
        pred_maskmulti = predictedlunginfmask[0,:,:,:]
        
        zz=np.argmax(pred_maskmulti,axis=-1)
        
        # El tamaño es de 128x128 pix
        lngmask = np.zeros((np.shape(zz)[0],np.shape(zz)[1]))
        
        lngmask[zz!=2]=1
        
        lnginfmask=np.uint16(pred_maskmulti[:,:,0]>0.1)

        lngmask2=self.getsmoothmask(lngmask)
        lunginfmask2=self.getsmoothmask(lnginfmask)
        
        a=0
        
        
        
        pred_maskmulti_res=lngmask2+lunginfmask2


        """
        pred_maskmulti=np.round(pred_maskmulti*4)-1
        
        lung=pred_maskmulti==1
        ggo=pred_maskmulti==2
        cons=pred_maskmulti==3
        
        pred_maskmulti_res=lung+2*ggo+3*cons
        """
        
                        
        return pred_maskmulti_res

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



