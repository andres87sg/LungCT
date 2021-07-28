# -*- coding: utf-8 -*-
"""
@author: Andres Sandino

https://stackoverflow.com/questions/53248099/keras-image-segmentation-using-grayscale-masks-and-imagedatagenerator-class

"""

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras import Input,layers, models
from tensorflow.keras.layers import Conv2DTranspose,Dropout,Conv2D,BatchNormalization, Activation,MaxPooling2D
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

import math
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import tqdm

    
#%% Model

scale = 2
filters= 32
nclasses= 2
    
def conv_block(tensor, nfilters, size=3, padding='same', initializer="he_normal"):
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def deconv_block(tensor, residual, nfilters, size=3, padding='same', strides=(2, 2)):
    y = Conv2DTranspose(nfilters, kernel_size=(size, size), strides=strides, padding=padding)(tensor)
    y = tf.concat([y, residual], axis=3)
    y = conv_block(y, nfilters)
    return y


def Unet(img_height, img_width, nclasses=2, filters=64):
# down
    input_layer = Input(shape=(img_height, img_width, 3), name='image_input')
    conv1 = conv_block(input_layer, nfilters=filters)
    conv1_out = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(conv1_out, nfilters=filters*2)
    conv2_out = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(conv2_out, nfilters=filters*4)
    conv3_out = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block(conv3_out, nfilters=filters*8)
    conv4_out = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv4_out = Dropout(0.5)(conv4_out)
    conv5 = conv_block(conv4_out, nfilters=filters*16)
    conv5 = Dropout(0.5,name='BOTTLENECK')(conv5)
# up
    deconv6 = deconv_block(conv5, residual=conv4, nfilters=filters*8)
    deconv6 = Dropout(0.5)(deconv6)
    deconv7 = deconv_block(deconv6, residual=conv3, nfilters=filters*4)
    deconv7 = Dropout(0.5)(deconv7) 
    deconv8 = deconv_block(deconv7, residual=conv2, nfilters=filters*2)
    deconv9 = deconv_block(deconv8, residual=conv1, nfilters=filters)
# output
    output_layer = Conv2D(filters=1, kernel_size=(1, 1))(deconv9)
    output_layer = BatchNormalization()(output_layer)
    output_layer = Activation('sigmoid')(output_layer)

    model = Model(inputs=input_layer, outputs=output_layer, name='Unet')
    return model

model = Unet(512//scale, 512//scale, nclasses, filters)


# Loading model weights

#model.load_weights('C:/Users/Andres/Desktop/CTClassifLungSegmModel.h5')
model.load_weights('C:/Users/Andres/Desktop/LungSegmModel.h5')
#tf.keras.models.load_model('C:/Users/Andres/Desktop/CTClassif/lng_seg_mdl.h5')

#%%

def imoverlay(img,predimg,coloredge):
    
    
    #predictedrgb=np.zeros((512,512,3))
    
    # Upsample image to 512x512
    predmask = cv2.resize(predimg,(512,512), interpolation = cv2.INTER_AREA)
    predmask = predmask*255
    
    # Upsample image to 512x512    
    img = cv2.resize(img,(512,512), interpolation = cv2.INTER_AREA)
    
    overlayimg = img.copy()
    overlayimg[predmask == 255] = coloredge
    
    return overlayimg


def getsmoothlungmask(pred_mask):
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    pred_mask_kk = cv2.morphologyEx(pred_mask, cv2.MORPH_ERODE, kernel)
    
    pred_mask2 = cv2.resize(pred_mask_kk,(512,512), 
                       interpolation = cv2.INTER_AREA)
    
    pred_mask3 = np.uint16(pred_mask2>=0.5)
    
    MASK3 = cv2.GaussianBlur(pred_mask3, (9,9), 5)
    
    pred_mask4 = np.uint16(MASK3>=0.5)
    
    return pred_mask4

def getsmoothmask(mask):
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    MaskClose = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    MaskOut = cv2.GaussianBlur(MaskClose, (5,5), 3)
    
    MaskOut=np.int16(MaskOut>=0.5)
    
    return MaskOut

def getmulticlassmask(im_or,imtrue_mask,num_classes,lng_label,ggo_label,con_label):
    
    im_array=im_or
    im_or=cv2.resize(im_or,(512,512), 
                        interpolation = cv2.INTER_AREA)
    
    # multiplicando por el la cantidad de clases 0: bkg 1:ggo 2:con
    imtrue_mask=np.int16((imtrue_mask/255)*2)
    
    imtrue_mask=cv2.resize(imtrue_mask,(512,512), 
                        interpolation = cv2.INTER_AREA)
    
    # Image resize
    im_array=cv2.resize(im_array,(512//scale,512//scale), 
                        interpolation = cv2.INTER_AREA)
    
    # Image gray level normalization
    im_array=im_array/255
    
    # Adding one dimension to array
    img_array = np.expand_dims(im_array,axis=[0])
    
    # Generate image prediction
    pred_mask = model.predict(img_array)
        
    # Image mask as (NxMx1) array
    pred_mask = pred_mask[0,:,:,0]
    
    pred_mask = getsmoothlungmask(pred_mask)
    lung_mask = pred_mask.copy()
    
    
    try:
        ggo=1
        ggo_mask=np.int16(imtrue_mask[:,:,0]==ggo)
        ggo_smooth_mask=getsmoothmask(ggo_mask)
        pred_mask[ggo_smooth_mask==1]=ggo_label
    except:
        print("Something else went wrong")

    try:
        con=2
        con_mask=np.int16(imtrue_mask[:,:,0]==con)
        con_smooth_mask=getsmoothmask(con_mask)
        pred_mask[con_smooth_mask==1]=con_label # Usar 2 si fusiono cons y ggo
    except:
        print("Something else went wrong")
    
    # Bkg=0, Lung=1, Cons=2,
    # Ojo, si tengo clases Bkg=0, Lung=1, fusion cons&ggo=2 entonces num classes=2
    #num_classes=3
    
    finalmask = pred_mask*lung_mask
    ROI_image = im_or[:,:,0]*(finalmask>0)
    
    norm_mask=np.uint16(((finalmask)/num_classes)*255)
    
    MulticlassMask=np.zeros((512,512,3))
    
    for i in range(3):
        MulticlassMask[:,:,i]=norm_mask
        
    return ROI_image,MulticlassMask

    

#%% Visualizacion de resultados (No es necesario correr esta sección)

#C:/Users/Andres/Desktop/CovidImages2/Training/CT/CT/

path = 'C:/Users/Andres/Desktop/CovidImages2/CT/'
path_mask = 'C:/Users/Andres/Desktop/CovidImages2/Mask/'

destpath = 'C:/Users/Andres/Desktop/CovidImages2/CT2/'
destpath_mask = 'C:/Users/Andres/Desktop/CovidImages2/Mask3/'

# path = 'C:/Users/Andres/Desktop/CovidImages2/CTMedSeg/'
# path_mask = 'C:/Users/Andres/Desktop/CovidImages2/MaskMedSeg/'

# destpath = 'C:/Users/Andres/Desktop/CovidImages2/CTMedSeg2/'
# destpath_mask = 'C:/Users/Andres/Desktop/CovidImages2/MaskMedSeg3/'

scale=2

listfiles = os.listdir(path)
listfiles_mask = os.listdir(path_mask)

for i in tqdm.tqdm(range(len(listfiles))):
#for i in range(10,11):
    
    # List of files
    im_name = listfiles[i]
    
    filename = im_name[:-4]
    
    im_name_mask = listfiles_mask[i]
    
    # Graylevel image (array)
    im_or=cv2.imread(path+im_name)
    
    imtrue_mask=cv2.imread(path_mask+im_name_mask)
    

    ROIimg, Mask = getmulticlassmask(im_or,
                                     imtrue_mask,
                                     num_classes=3,
                                     lng_label=1,
                                     ggo_label=2,
                                     con_label=3)
    

    
    
    cv2.imwrite(destpath+filename+'.png', ROIimg)
    cv2.imwrite(destpath_mask+filename+'_mask'+'.png', Mask)
    