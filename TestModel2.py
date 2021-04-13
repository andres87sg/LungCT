# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 18:35:58 2021

@author: Andres
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 20:30:21 2021

@author: Andres

https://stackoverflow.com/questions/53248099/keras-image-segmentation-using-grayscale-masks-and-imagedatagenerator-class

"""

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import Input,layers, models
from tensorflow.keras.layers import Conv2DTranspose,Dropout,Conv2D,BatchNormalization, Activation,MaxPooling2D
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

import math
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np

#%%

#input_dir = "C:/Users/Andres/Desktop/imexhs/Lung/Prueba/CT/"
#mask_dir = "C:/Users/Andres/Desktop/imexhs/Lung/Prueba/mask/"

input_dir = 'C:/Users/Andres/Desktop/CTClassif/test/'
mask_dir = 'C:/Users/Andres/Desktop/CTClassif/mask_test/'

#%%

image_datagen = ImageDataGenerator(rescale=1./255)
mask_datagen = ImageDataGenerator(rescale=1./255)

target_size=(512//4, 512//4)

image_generator = image_datagen.flow_from_directory(
    input_dir,
    class_mode=None, target_size=target_size, shuffle=False,
    seed=1,batch_size=64)

mask_generator = mask_datagen.flow_from_directory(
    mask_dir,
    class_mode=None, target_size=target_size, shuffle=False,
    seed=1,batch_size=64)

#steps = image_generator.n//image_generator.batch_size


train_generator = zip(image_generator, mask_generator)



# combine generators into one which yields image and masks


#%%

img = image_generator.next()
mask = mask_generator.next()
    
for i in range(20):
    #img = image_generator.next()
    #mask = mask_generator.next()
    
    print(img.shape)
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.imshow(img[i])
    plt.subplot(1,2,2)
    plt.imshow(mask[i])
    plt.axis('off')
    plt.show()
    
    
    
#%%
#model = tf.keras.models.load_model('C:/Users/Andres/Desktop/CTClassif/exp.h5')
    
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
    conv5 = Dropout(0.5)(conv5)
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

#%%

model = Unet(512//4, 512//4, nclasses=2, filters=16)

model.summary()



#%%


model.load_weights('C:/Users/Andres/Desktop/CTClassif/exp.h5')

#img_array = keras.preprocessing.image.img_to_array(img)
#img_array = tf.expand_dims(img_array, 0)
    

#%%

for i in range(0,20):
    #img = image_generator.next()
    #mask = mask_generator.next()
    print(img.shape)
    #plt.figure(1)
    #plt.imshow(img[0])    
    #plt.show()
    img_array = img[i]
    
    img_array = tf.expand_dims(img_array, 0)
    predict = model.predict(img_array)

    predicted_image=predict[0]
    predicted_image=np.int16(np.round(predicted_image>0.5))
    predicted_image=predicted_image[:,:,0]
    
    plt.figure(1)
    plt.subplot(1,3,1)
    plt.axis('off')
    plt.imshow(img[i])
    plt.title('Graylevel')
    plt.subplot(1,3,2)
    
    ground=imoverlay(img[i],mask[i,:,:,1],[0,0,255])
    plt.imshow(ground)
    plt.title('Groundtruth')
    plt.axis('off')    
    plt.subplot(1,3,3)
    
    pred=imoverlay(img[i],predicted_image,[255,0,0])
    plt.imshow(pred,cmap='gray')
    plt.title('Predicted mask')
    plt.axis('off') 
    plt.show()
    
    
    del img_array
    
    # plt.figure(1)
    # plt.imshow(predicted_image[:,:,0],cmap='gray')
    # plt.axis('off')
    # plt.title('predicted_mask')
    
    
#%%

import cv2

def imoverlay(img,predicted_image,coloredge):
    
    
    #predictedrgb=np.zeros((512,512,3))
    
    predictedmask= cv2.resize(predicted_image,(512,512), interpolation = cv2.INTER_AREA)
    predicted=predictedmask*255
    
    
    imX = cv2.resize(img,(512,512), interpolation = cv2.INTER_AREA)
    
    img2=imX.copy()
    img2[predicted == 255] = coloredge
    
    
    #plt.subplot(1,2,1)
    #plt.imshow(imX)
    #plt.axis('off')
    #plt.subplot(122),plt.imshow(img2)
    
    return img2

   
#%%
    
import os
import cv2

path = 'C:/Users/Andres/Desktop/imexhs/Lung/dicomimage/Torax/dcm2png/test_dcm/'
listfiles = os.listdir(path)
#mask = 'C:/Users/Andres/Desktop/imexhs/Lung/dicomimage/Torax/dcm2png/val_dcm/'#

#for i in range(len(listfiles)):
#for i in range(20):
for i in range(len(listfiles)):
    
    imgname = listfiles[i]
    img_array=cv2.imread(path+imgname)
    img_array=cv2.resize(img_array,(512//4,512//4), interpolation = cv2.INTER_AREA)
    img_array=img_array/255 #Normalizing image
    #plt.imshow(img_array,cmap='gray')
    
    #imgray= cv2.merge((img_array,img_array,img_array))
    img_array = tf.expand_dims(img_array, 0)
    
    predict = model.predict(img_array)

    predicted_image=predict[0]
    predicted_image=np.int16(np.round(predicted_image>0.5))
    predicted_image=predicted_image[:,:,0]
    
    img=np.array(img_array)
    
    pred=imoverlay(img[0],predicted_image,[255,0,0])
    
    plt.imshow(pred,cmap='gray')
    plt.title('Predicted mask')
    plt.axis('off') 
    
    plt.show()
    plt.close()
    
    #imn = predict[0]
    #plt.imshow(predicted_image)    
    
   # print(imgname)
    
    
    
    
    