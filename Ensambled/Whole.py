"""
Created on  12/05/2021
Modified on 12/05/2021
@author: Andres Sandino

"""
#%%
 
import pydicom as dicom
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os



import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from skimage import io, color

from tensorflow.keras import Input,layers, models
from tensorflow.keras.layers import Conv2DTranspose,Dropout,Conv2D,BatchNormalization, Activation,MaxPooling2D
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

#%%

case='22474FA3'
patient_no = 8

origpath = 'C:/Users/Andres/Desktop/imexhs/Lung/dicomimage/Torax/'+ case +'/' 
listfiles = os.listdir(origpath)
destpath = 'C:/Users/Andres/Desktop/imexhs/Lung/dicomimage/Torax/dcm2png/val_dcm/'
#patient = 'patient1a'

model_multiclass=load_model('multiclass_seg_mdl.h5')
model_lungsegmentation=load_model('lng_seg_mdl.h5')

#%%

dcmimages=[]

#for i in range(len(listfiles)):
for i in range(1,60):

    dcmfilename = listfiles[i]
    
    # Convert .dcm file to graylevel image in window
    # Lung window (L=-500, W=1500)
   
    L=-500
    W=1500
        
    [norm_img, ins_num] = dcm_convert(origpath,dcmfilename,L,W)  
    
    im_array=cv2.resize(norm_img,(512,512), 
                    interpolation = cv2.INTER_AREA)
    
    dcmimages.append(im_array)
    
    scale = 4

    im1=cv2.cvtColor(im_array,cv2.COLOR_GRAY2RGB)

    im_array=cv2.resize(im1,(512//scale,512//scale), 
                        interpolation = cv2.INTER_AREA)
      
    # Image gray level normalization
    im_array=im_array/255
    
    # Adding one dimension to array
    img_array2 = np.expand_dims(im_array,axis=[0])

    pred_mask = model_lungsegmentation.predict(img_array2)
    
    # Image mask as (NxMx1) array
    pred_mask = pred_mask[0,:,:,0]
    
    pred_mask = cv2.resize(pred_mask,(512,512), 
                       interpolation = cv2.INTER_AREA)

    croppedlungs=im1[:,:,0]*pred_mask

    
    lungs_array=cv2.resize(croppedlungs,(512//4,512//4), 
                        interpolation = cv2.INTER_AREA)
    
    lungs_array=cv2.cvtColor(lungs_array,cv2.COLOR_GRAY2RGB)
    
    lungs_array=lungs_array/255


    classes=4    

    lungs_array_tensor = np.expand_dims(lungs_array,axis=[0])
    segm_pred = model_multiclass.predict(lungs_array_tensor)
    segm_mask = segm_pred[0,:,:,0]
    
    pred_maskmulti=np.round(segm_mask*classes)
    pred_maskmulti=pred_maskmulti-1 #Classes: 0,1,2,3

    # Resize predicted mask
    # pred_mask = cv2.resize(pred_maskmulti,(512,512), 
    #                       interpolation = cv2.INTER_AREA)
    
    col_predmask,gray_predmask = getcolormask(pred_maskmulti)
    
    pred_maskfinal = cv2.resize(col_predmask,(512,512), 
                           interpolation = cv2.INTER_AREA)
    
    pred_graymaskfinal = cv2.resize(gray_predmask,(512,512), 
                       interpolation = cv2.INTER_AREA)
    

    plt.figure()
    io.imshow(color.label2rgb(pred_graymaskfinal,im1[:,:,0],
                          colors=[(0,0,0),(255,0,0),(0,0,255)],
                          alpha=0.0015, bg_label=0, bg_color=None))
    plt.axis('off')


#%%

# Convert gray mask to color mask
def getcolormask(inputmask):
    
    # Labels in the image    
    lab=np.unique(inputmask)
    
    # Image size
    [w,l] = np.shape(inputmask)
    
    # 3-Channel image
    colormask = np.zeros([w,l,3])
    
    # Gray level image
    graymask = np.zeros([w,l]) 
    
    # Color label (black, green, red, blue)
    colorlabel=([0,0,0],[0,255,0],[255,0,0],[0,0,255]) # Colors
    graylabel=[0,1,2,3] # Gray leves
    
    # Replace values in the image 
    for lab in lab:
        colormask[inputmask==lab]=colorlabel[np.int16(lab)]
        graymask[inputmask==lab]=graylabel[np.int16(lab)]
    
    # Mask in color
    colormask=np.int16(colormask)
    
    # Mask in graylevel
    graymask=np.int16(graymask)
    
    
    return colormask,graymask