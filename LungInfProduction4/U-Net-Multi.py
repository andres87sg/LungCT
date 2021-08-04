# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 21:04:44 2021

@author: Andres
"""
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
#%%


input_dir = 'C:/Users/Andres/Desktop/MedSegData/CT/'
mask_dir = 'C:/Users/Andres/Desktop/MedSegData/Mask2/'


# val_dir = '/Users/Andres/Desktop/LungInfDataset/Testing/CT2/'
# mask_val_dir = '/Users/Andres/Desktop/LungInfDataset/Testing/Mask2/'


# input_dir = '/home/usuario/Documentos/GBM/Samples/PC_SegmentationSet/train/PC/'
# mask_dir = '/home/usuario/Documentos/GBM/Samples/PC_SegmentationSet/train/Mask/'

# val_dir = '/home/usuario/Documentos/GBM/Samples/PC_SegmentationSet/val/PC/'
# mask_val_dir = '/home/usuario/Documentos/GBM/Samples/PC_SegmentationSet/val/Mask/'

scale_factor = 4
n_filters = 32
image_size = 512
batch_size = 8

img_width =  np.uint16(image_size/scale_factor)
img_height = np.uint16(image_size/scale_factor)
img_channels = 3
color_mode = "rgb"


target_size=(img_width, img_height)

image_datagen = ImageDataGenerator(
                                  rescale=1./255,
                                  validation_split=0.2
                                  #rotation_range=5,                                  
                                  #fill_mode='nearest',
                                  #zoom_range=0.2,
                                 
                                  #width_shift_range=0.01,
                                  #height_shift_range=0.01,
                                  #horizontal_flip=True,
                                  #vertical_flip=True,
                                  #channel_shift_range=0.5,
                                  )

mask_datagen = ImageDataGenerator(
                                  rescale=1./255,
                                  validation_split=0.2
                                  #rotation_range=5,                                  
                                  #fill_mode='nearest',
                                  #zoom_range=0.2,
                                 
                                  #width_shift_range=0.01,
                                  #height_shift_range=0.01,
                                  #horizontal_flip=True,
                                  #vertical_flip=True,
                                  #channel_shift_range=0.5,
                                  )


image_datagen_val = ImageDataGenerator(rescale=1./255,validation_split=0.2)
mask_datagen_val = ImageDataGenerator(rescale=1./255,validation_split=0.2)
# image_datagen_val = ImageDataGenerator(rescale=1./255)
# mask_datagen_val = ImageDataGenerator(rescale=1./255)



image_generator = image_datagen.flow_from_directory(
    input_dir,color_mode=color_mode,
    class_mode=None, target_size=target_size,  batch_size=batch_size,
    shuffle='False',seed=1,subset='training')

mask_generator = mask_datagen.flow_from_directory(
    mask_dir,color_mode=color_mode,
    class_mode=None, target_size=target_size, batch_size=batch_size,
    shuffle='False',seed=1,subset='training')

image_generator_val = image_datagen_val.flow_from_directory(
    input_dir,color_mode=color_mode,
    class_mode=None, target_size=target_size, batch_size=batch_size,
    seed=1,subset='validation')

mask_generator_val = mask_datagen_val.flow_from_directory(
    mask_dir,color_mode=color_mode,
    class_mode=None, target_size=target_size, batch_size=batch_size,
    seed=1,subset='validation')

steps = image_generator.n//image_generator.batch_size
steps_val = image_generator_val.n//image_generator_val.batch_size


train_generator = zip(image_generator, mask_generator)
val_generator = zip(image_generator_val, mask_generator_val)

#%%

import matplotlib.pyplot as plt

import random
n = random.randint(0,batch_size-1)

im1=image_generator[0]
mask1=mask_generator[0]

im1=im1[n,:,:,0]
mask=mask1[n,:,:,:]

plt.subplot(1,2,1)
plt.imshow(im1,cmap='gray')
plt.title('im_gray')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(mask,cmap='gray')
plt.title('mask')

plt.axis('off')



#%%

def conv_block(inputs, filters, pool=True):
    x = Conv2D(filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    if pool == True:
        p = MaxPool2D((2, 2))(x)
        return x, p
    else:
        return x

def build_unet(shape, num_classes):
    inputs = Input(shape)

    """ Encoder """
    x1, p1 = conv_block(inputs, 16, pool=True)
    x2, p2 = conv_block(p1, 32, pool=True)
    x3, p3 = conv_block(p2, 48, pool=True)
    x4, p4 = conv_block(p3, 64, pool=True)

    """ Bridge """
    b1 = conv_block(p4, 128, pool=False)

    """ Decoder """
    u1 = UpSampling2D((2, 2), interpolation="bilinear")(b1)
    c1 = Concatenate()([u1, x4])
    x5 = conv_block(c1, 64, pool=False)

    u2 = UpSampling2D((2, 2), interpolation="bilinear")(x5)
    c2 = Concatenate()([u2, x3])
    x6 = conv_block(c2, 48, pool=False)

    u3 = UpSampling2D((2, 2), interpolation="bilinear")(x6)
    c3 = Concatenate()([u3, x2])
    x7 = conv_block(c3, 32, pool=False)

    u4 = UpSampling2D((2, 2), interpolation="bilinear")(x7)
    c4 = Concatenate()([u4, x1])
    x8 = conv_block(c4, 16, pool=False)

    """ Output layer """
    output = Conv2D(num_classes, 1, padding="same", activation="softmax")(x8)

    return Model(inputs, output)

model = build_unet((128, 128, 3), 3)
model.summary()

#%%

def step_decay(epoch):
	initial_lrate = 1e-4
	drop = 0.1
	epochs_drop = 100
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

lr = LearningRateScheduler(step_decay)
es = EarlyStopping(patience=30,mode='min', verbose=1)
checkpoint_path ='/home/usuario/Documentos/GBM/Experimentos/InfSegmModel.h5'

mc = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1 , save_best_only=True, mode='min')


lr=0.001
model.compile(loss="categorical_crossentropy", optimizer='Adam')


history = model.fit(train_generator,
                    steps_per_epoch=steps,
                    validation_data=train_generator,
                    validation_steps=steps_val,
                    epochs=3,
                    verbose=1,
                    callbacks=[]
                    )

#%%

import matplotlib.pyplot as plt

import random
n = random.randint(0,batch_size-1)


im1=image_generator[0]
mask1=mask_generator[0]

im1=im1[n,:,:,0]
mask=mask1[n,:,:,:]

plt.subplot(1,2,1)
plt.imshow(im1,cmap='gray')
plt.title('im_gray')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(mask,cmap='gray')
plt.title('mask')

plt.axis('off')
#%%

ll=tf.one_hot(mask, 3, dtype=tf.int32)


#%%