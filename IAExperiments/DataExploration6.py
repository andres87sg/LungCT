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

import scipy as sp
#%%

# path = 'C:/Users/Andres/Desktop/CovidImages2/Training/CT2/CT/'
# pathmask = 'C:/Users/Andres/Desktop/CovidImages2/Training/Mask/Mask/'

path = 'C:/Users/Andres/Desktop/CovidImages2/CTMedSeg2/'
pathmask = 'C:/Users/Andres/Desktop/CovidImages2/MaskMedSeg2/'




listfiles = os.listdir(path)
listfilesmask = os.listdir(pathmask)

statslist=[]

# i=10
#for i in range(len(listfiles)):
for i in tqdm.tqdm(range(len(listfiles))):

#for i in range(1,31):

    
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
    
    data_class0=[im_or[grtr_mask==85],0]
    data_class1=[im_or[grtr_mask==170],1]
    data_class2=[im_or[grtr_mask==255],2]
    
    for data_class in ([data_class0,data_class1,data_class2]):
        
        
        # Si hay algo en el vector data_class
        if len(data_class[0]) !=0:
            
            mean_gl = np.mean(data_class[0])
            med_gl  = np.median(data_class[0])
            std_gl  = np.std(data_class[0])
            kurt_gl = sp.stats.kurtosis(data_class[0])
            skew_gl = sp.stats.skew(data_class[0])
            entr_gl = sp.stats.entropy(data_class[0])
            [mode_gl,b]= sp.stats.mode(data_class[0])
            
            class_gl= data_class[1]
            statist = [class_gl,mean_gl,med_gl,std_gl,kurt_gl,skew_gl,mode_gl[0],
                       entr_gl]
            statslist.append(statist)
 

classnames=['class','mean','med','std','skew','kurt','mode','entr']

df = pd.DataFrame(statslist, columns = classnames)
df.head()

#%%


is_one=df.loc[:,'class']==0
dfclass_one=df.loc[is_one]

is_two=df.loc[:,'class']==1
dfclass_two=df.loc[is_two]

is_three=df.loc[:,'class']==2
dfclass_three=df.loc[is_three]

true_labels=df['class'].values

#%%
x1=dfclass_one.iloc[:,2]
y1=dfclass_one.iloc[:,4]

x2=dfclass_two.iloc[:,2]
y2=dfclass_two.iloc[:,4]

x3=dfclass_three.iloc[:,2]
y3=dfclass_three.iloc[:,4]


plt.scatter(x1,y1,marker='.')
plt.scatter(x2,y2,marker='.')
plt.scatter(x3,y3,marker='.')
plt.title('median vs std')
plt.xlabel('median')
plt.ylabel('skew')

#%%
from sklearn.decomposition import PCA
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2)
nn=pca.fit(X)

#%%

X=df.iloc[:,1:6].values

#%%

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

Nc = range(1, 10)
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
k_labels = kmeans.labels_

print(centroids)

#%%

centroids2=centroids.copy()
centroids2.sort(axis=0)


#%%
plt.scatter(x1,y1,marker='.')
plt.scatter(x2,y2,marker='.')
plt.scatter(x3,y3,marker='.')
plt.title('median vs std')
plt.xlabel('median')
plt.ylabel('std')

plt.plot(centroids2[0][0],centroids2[0][3], 'r*')
plt.plot(centroids2[1][0],centroids2[1][3], 'b*')
plt.plot(centroids2[2][0],centroids2[2][3], 'r*')
#plt.plot(centroids[3][0],centroids[3][1], 'r*')

#%%

i=20

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

plt.imshow(im_or)


#%%
import cv2 as cv

mask1=segmentationmask(im_or,centroids2,0)
mask2=segmentationmask(im_or,centroids2,1)
mask3=segmentationmask(im_or,centroids2,2)

mask_x=np.logical_not(mask1*mask2)

mask_ggo=mask_x*mask2
mask_con=mask_x*mask3

#%%
im1=np.zeros((512,512,3))
im1[:,:,0]=im_or*255
im1[:,:,1]=im_or*255
im1[:,:,2]=im_or*255

mask=mask_con*255



from skimage import io, color

overlapimg=color.label2rgb(mask1,im_or/255,
                          colors=[(0,255,0)],
                          alpha=1, bg_label=0, bg_color=None)  

plt.imshow(overlapimg)

#%%

#kernel = np.ones((2,2),np.uint8)

#closing = cv.morphologyEx(mask2, cv.MORPH_CLOSE, kernel)
#opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)

#plt.imshow(opening)


# for ind in range(3):
    
#     min_a=centroids2[ind,0]-centroids2[ind,2]
#     max_a=centroids2[ind,0]+centroids2[ind,2]
    
#     pp1=np.zeros((512,512))
#     pp2=np.zeros((512,512))
    
#     pp2[im_or>min_a]=1
#     pp1[im_or<max_a]=1
#     kkz=pp1*pp2
#     plt.figure()
#     plt.imshow(kkz,cmap='gray')


#%%

def segmentationmask(im_or,centroids2,ind):
    
    min_a=centroids2[ind,1]-centroids2[ind,2]
    max_a=centroids2[ind,1]+centroids2[ind,2]

    pp1=np.zeros((512,512))
    pp2=np.zeros((512,512))
    
    pp2[im_or>min_a]=1
    pp1[im_or<max_a]=1
    
    kkz=pp1*pp2
    
    return kkz
    # plt.figure()
    # plt.imshow(kkz,cmap='gray')    

#%%
pp1=np.zeros((512,512,3))
pp1[:,:,0]=im_or
pp1[:,:,1]=im_or
pp1[:,:,2]=im_or

pixel_values = np.float32(im_or.reshape((-1,1)))
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

flags = cv.KMEANS_RANDOM_CENTERS

k=3
compactness,labels,centers = cv.kmeans(pixel_values,k,None,criteria,10,flags)

centers = np.uint8(centers)
labels = labels.flatten()

segmented_image = centers[labels.flatten()]

X=pixel_values.copy()

Nc = range(1, 4)
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


segmented_image = segmented_image.reshape(im_or.shape)
# show the image
plt.imshow(segmented_image,cmap='gray')
plt.show()


#%%

masked_image = np.copy(pp1)
masked_image = masked_image.reshape((-1, 3))
cluster = 3
masked_image[labels == cluster] = [0, 0, 0]
masked_image = masked_image.reshape(pp1.shape)
plt.imshow(masked_image)
plt.show()

#k = 3
#_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

#%%


i=1

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

plt.imshow(im_or)

#%%


y=grtr_mask.reshape((-1, 1))
X=im_or.reshape((-1, 1))


#%%
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold




X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.4, 
                                                    random_state=0)

model = KNeighborsClassifier(n_neighbors = 4)
clf = model.fit(X[1:100], y[1:100])

#%%

mm=[]
scores=[]
kf = KFold(n_splits=10)
for train, test in kf.split(X):
    #print('Train: %s | test: %s' % (train, test))
    model = KNeighborsClassifier(n_neighbors = 3)
    clf = model.fit(X[train], y[train])
    mm.append(clf)
    #print(clf)
    sco=clf.score(X[test],y[test])
    scores.append(sco)
    print(sco)
    #scores = cross_val_score(clf, X[test], y[test], cv=5)




'''
https://github.com/AbhinavUtkarsh/Image-Segmentation/blob/master/KNN%20Image%20Segmentation.ipynb

'''
