# -*- coding: utf-8 -*-
"""
Created on Thu May 27 13:39:27 2021

@author: Andres
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
import PIL
import pandas as pd

import timeit as timeit
from timeit import timeit

import scipy as sp
from scipy.stats import skew,kurtosis

import skimage
from skimage.feature import greycomatrix, greycoprops

import tqdm

#%%

path = 'C:/Users/Andres/Desktop/CovidImages2/Training/CT2/CT/'
pathmask = 'C:/Users/Andres/Desktop/CovidImages2/Training/Mask/Mask/'

listfiles = os.listdir(path)
listfilesmask = os.listdir(pathmask)

statslist=[]

i=10
#for i in range(len(listfiles)):
for i in tqdm.tqdm(range(len(listfiles))):
    
    im_name = listfiles[i] # Gray level
    im_namemask = listfilesmask[i] # Segmentation mask
    
    # Graylevel image (array)
    im_or=cv2.imread(path+im_name)
    im_array=im_or[:,:,0]
    grtr_mask=cv2.imread(pathmask+im_namemask)
    
    mask=np.int16(grtr_mask[:,:,0]>0)
    
    kernel = np.ones((10, 10), np.uint8)
    cropmask = cv2.erode(mask, kernel)
    
    
    im_or=im_or[:,:,0]*cropmask
    grtr_mask = grtr_mask[:,:,0]*cropmask
    
    data_class0=[im_or[grtr_mask==63],0]
    data_class1=[im_or[grtr_mask==127],1]
    data_class2=[im_or[grtr_mask==255],2]
    
    for data_class in ([data_class0,data_class1,data_class2]):
        
        
        # Si hay algo en el vector data_class
        if len(data_class[0]) !=0:
            
            mean_gl = np.mean(data_class[0])
            med_gl  = np.median(data_class[0])
            std_gl  = np.std(data_class[0])
            kurt_gl = sp.stats.kurtosis(data_class[0])
            skew_gl = sp.stats.skew(data_class[0])
            class_gl= data_class[1]
            statist = [class_gl,mean_gl,med_gl,std_gl,kurt_gl,skew_gl]
            statslist.append(statist)
 

classnames=['class','mean','med','std','skew','kurt']

df = pd.DataFrame(statslist, columns = classnames)
df.head()

#%%


is_one=df.loc[:,'class']==0
dfclass_one=df.loc[is_one]

is_two=df.loc[:,'class']==1
dfclass_two=df.loc[is_two]

is_three=df.loc[:,'class']==2
dfclass_three=df.loc[is_three]

#%%
x1=dfclass_one.iloc[:,2]
y1=dfclass_one.iloc[:,3]

x2=dfclass_two.iloc[:,2]
y2=dfclass_two.iloc[:,3]

x3=dfclass_three.iloc[:,2]
y3=dfclass_three.iloc[:,3]


plt.scatter(x1,y1,marker='.')
plt.scatter(x2,y2,marker='.')
plt.scatter(x3,y3,marker='.')
plt.title('mean vs std')
plt.xlabel('mean')
plt.ylabel('median')

#%%

X=df.iloc[1:100,2:4].values

#%%

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]
score
plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()

#%%

kmeans = KMeans(n_clusters=3).fit(X)
centroids = kmeans.cluster_centers_
print(centroids)

#%%
plt.scatter(x1,y1,marker='.')
plt.scatter(x2,y2,marker='.')
plt.scatter(x3,y3,marker='.')
plt.title('mean vs std')
plt.xlabel('mean')
plt.ylabel('median')

plt.plot(centroids[0][0],centroids[0][1], 'r*')
plt.plot(centroids[1][0],centroids[1][1], 'b*')
plt.plot(centroids[2][0],centroids[2][1], 'r*')
#plt.plot(centroids[3][0],centroids[3][1], 'r*')

#%%

from skimage.segmentation import slic

for numSegments in (10, 20, 30):
	segments = slic(im_or, n_segments = numSegments, sigma = 5)
    fig = plt.figure("Superpixels -- %d segments" % (numSegments))
	ax = fig.add_subplot(1, 1, 1)
	ax.imshow(mark_boundaries(im_or, segments))
	plt.axis("off")


#%%

X_new = np.array([[45.92,57.74,15.66]]) #davidguetta
 
new_labels = kmeans.predict(X_new)
print(new_labels)





#%%

Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]
score
plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()
1
2
3
4
5
6
7
8
9
10
Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]
score
plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()





#%%

# mean_gl = np.mean(patch)
# med_gl  = np.median(patch)
# std_gl  = np.std(patch)
# kurt_gl = sp.stats.kurtosis(patch)
# skew_gl = sp.stats.skew(patch)
# class_gl= np.int16(label)

# statist = [class_gl,mean_gl,med_gl,std_gl,kurt_gl,skew_gl,
#            contr,homog,dissi,energ]
# statslist.append(statist)

# classnames=['class','mean','med','std','skew','kurt',
#         'contr','homog','dissi','energ'                
#         ]

# df = pd.DataFrame(statslist, columns = classnames)
# df.head()


# # mean_gl = np.mean(patch)
# # med_gl  = np.median(patch)
# # std_gl  = np.std(patch)
# # kurt_gl = sp.stats.kurtosis(patch)
# # skew_gl = sp.stats.skew(patch)
# # class_gl= np.int16(label)


# # statist = [class_gl,mean_gl,med_gl,std_gl,kurt_gl,skew_gl,
# #                contr,homog,dissi,energ]
# # statslist.append(statist)




# #%%

# # Creating kernel
# kernel = np.ones((10, 10), np.uint8
  
# # Using cv2.erode() method 
# imagemask = cv2.erode(mask, kernel)
# grtr_mask2 = grtr_mask*imagemask



# kk=im_or[im_or>0]

# plt.hist(kk,20)
# plt.figure()
# plt.imshow(imagemask)



