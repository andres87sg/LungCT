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

path = 'C:/Users/Andres/Desktop/TrainMedSeg/CTMedSeg/'
pathmask = 'C:/Users/Andres/Desktop/TrainMedSeg/MaskMedSeg3/'


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
    
    im_or=cv2.resize(im_or,(512,512),
              interpolation = cv2.INTER_AREA)
    
    
    im_array=im_or[:,:,0]
    im_array=cv2.resize(im_array,(512,512),
              interpolation = cv2.INTER_AREA)
    grtr_mask=cv2.imread(pathmask+im_namemask)
    #print(np.unique(grtr_mask))
    mask=np.int16(grtr_mask[:,:,0]>0)
    
    # kernel = np.ones((10, 10), np.uint8)
    # cropmask = cv2.erode(mask, kernel)
    
    # im_or=im_array.copy()
    
    # im_or=im_or[:,:,0]*cropmask
    # grtr_mask = grtr_mask[:,:,0]*cropmask
    
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
            # statist = [class_gl,mean_gl,med_gl,std_gl,kurt_gl,skew_gl,mode_gl[0],
            #            entr_gl]
            
            statist = [class_gl,mean_gl,med_gl,std_gl]
            statslist.append(statist)
 

#classnames=['class','mean','med','std','skew','kurt','mode','entr']
classnames=['class','mean','med','std']


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
y1=dfclass_one.iloc[:,0]

x2=dfclass_two.iloc[:,2]
y2=dfclass_two.iloc[:,0]

x3=dfclass_three.iloc[:,2]
y3=dfclass_three.iloc[:,0]


plt.scatter(x1,y1,marker='.')
plt.scatter(x2,y2,marker='.')
plt.scatter(x3,y3,marker='.')
plt.title('median vs std')
plt.xlabel('median')
plt.ylabel('std')

#%%

zz=pd.concat([x1,x2,x3])
zz2=pd.concat([y1,y2,y3])

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.boxplot(x=zz2,y=zz)

#%%

from scipy import stats
print(stats.mannwhitneyu(x2,x3,alternative='two-sided'))

#%%

X=df.iloc[:,1:3].values

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

X=df.iloc[:,1:3].values


from sklearn import preprocessing

features_matrix=X.copy()
scaler = preprocessing.StandardScaler().fit(features_matrix)
features_matrix_scal=scaler.transform(features_matrix)
X=features_matrix_scal.copy()
y=true_labels


#%%

from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

import pickle
from joblib import dump, load


models_list=[]
list_scores=[]
kf = KFold(n_splits=10)
for train, test in kf.split(X):
    #print('Train: %s | test: %s' % (train, test))
    model = KNeighborsClassifier(n_neighbors = 3)
    #model = svm.SVC(kernel='linear')
    
    clf = model.fit(X[train], y[train])
    models_list.append(clf)
    #print(clf)
    sco=clf.score(X[test],y[test])
    #print(test)
    list_scores.append(sco)
    print(sco)
    #scores = cross_val_score(clf, X[test], y[test], cv=5)
    

list_scores=np.array(list_scores)

ind_bestmodel=np.where(list_scores==np.max(list_scores))[0][0]


bestmodel=models_list[ind_bestmodel]

joblib_file = "C:/Users/Andres/Desktop/imexhs/Lung/LungCT/IAExperiments/KNNmodel.pkl"
dump(bestmodel, joblib_file)

#%%
import joblib


model_filename =  "C:/Users/Andres/Desktop/imexhs/Lung/LungCT/IAExperiments/KNNmodel.pkl"
clf_model = joblib.load(model_filename)


#%% PARTE DOOOSSS

'''
PARTE DOSSSS
¨******
*****
'''


import cv2 as cv
from tensorflow.keras.models import load_model

path = 'C:/Users/Andres/Desktop/TrainMedSeg/CTMedSeg/'
pathmask = 'C:/Users/Andres/Desktop/TrainMedSeg/MaskMedSeg3/'

listfiles = os.listdir(path)
listfilesmask = os.listdir(pathmask)

#for i in range(len(listfiles)):
for i in range(9,10):
    
    im_name = listfiles[i] # Gray level
    im_namemask = listfilesmask[i] # Segmentation mask
    
    # Graylevel image (array)
    im_or=cv.imread(path+im_name)
    
    

path='C:/Users/Andres/Desktop/CTClassif/'
mdlfilename='lng_seg_mdl.h5'
mdl_lungsegmentation=load_model(path+mdlfilename)


def getprepareimgCNN(inputimg,imgscale):
    imgnormsize=(512,512)
    inputimg=cv.resize(inputimg,(imgnormsize[0]//imgscale,
                            imgnormsize[1]//imgscale),
                    interpolation = cv.INTER_NEAREST)
      
    # Input image normalization imnorm = im/max(im)
    norminputimg=inputimg/np.max(inputimg)
    
    # Adding one dimension to array
    nn_inputimg = np.expand_dims(norminputimg,axis=[0])
    
    return nn_inputimg

inputCNNimg=getprepareimgCNN(im_or,imgscale=4)
predictedmask = mdl_lungsegmentation.predict(inputCNNimg)

#%%


predictedmask2 = predictedmask[0,:,:,0]

im_or_res=cv.resize(im_or,(512,
                            512),
                    interpolation = cv.INTER_AREA)

pp1=cv.resize(predictedmask2,(512,
                            512),
                    interpolation = cv.INTER_NEAREST)

lngsegm=np.round(pp1)*im_or_res[:,:,0]

#%%

path='C:/Users/Andres/Desktop/'
mdlfilename = 'InfSegmModel-Multi-Python.h5'
mdl_infsegmentation = load_model(path+mdlfilename)

lngsegm2=np.zeros((512,512,3))

#%%

lngsegm2[:,:,0]=lngsegm/255
lngsegm2[:,:,1]=lngsegm/255
lngsegm2[:,:,2]=lngsegm/255    
#%%

inputCNNimg2=getprepareimgCNN(lngsegm2,imgscale=4)

pred_maskmulti = mdl_infsegmentation.predict(inputCNNimg2)

lnginfmask=np.uint16(pred_maskmulti[0,:,:,0]>0.1)

lnginfmask2=cv.resize(lnginfmask,(512,
                            512),
                    interpolation = cv.INTER_AREA)


roi = np.where(lnginfmask2 == 1)

#%%

def feature_extraction(im_or,roi,subsample):
    dist=7
    statslist=[]
    
    #subsample=1
    
    xcoord=roi[0][::subsample]
    ycoord=roi[1][::subsample]
    
    # Slicing window
    for k in range(np.shape(xcoord)[0]):
        #print(k)
        c=ycoord[k],xcoord[k]
        data=(im_or[c[1]-dist:c[1]+dist,c[0]-dist:c[0]+dist]).flatten()
        mean_gl = np.mean(data)
        med_gl  = np.median(data)
        std_gl  = np.std(data)
        kurt_gl = sp.stats.kurtosis(data)
        skew_gl = sp.stats.skew(data)        
        [mode_gl,b]= sp.stats.mode(data)
        entr_gl = sp.stats.entropy(data)       
        
        #statslist.append([mean_gl,med_gl,std_gl,kurt_gl,skew_gl])
        #statslist.append([mean_gl,med_gl,std_gl])
        statslist.append([mean_gl,med_gl])

    featurematrix = np.array(statslist)    

    return featurematrix

#%%

subsample=3

featurematrix=feature_extraction(im_or_res,roi,subsample)
scaler = preprocessing.StandardScaler().fit(featurematrix)
featurematrix_norm=scaler.transform(featurematrix)

#%%
import joblib

model_filename =  "C:/Users/Andres/Desktop/imexhs/Lung/LungCT/IAExperiments/KNNmodel.pkl"
clf_model = joblib.load(model_filename)


#%%

def predmask(im_or,roi,subsample,predicted_label,label):
    
    #subsample=1   
    predcoordy=roi[0][::subsample][predicted_label==label]
    predcoordx=roi[1][::subsample][predicted_label==label]
    predictedmask=np.zeros((np.shape(im_or)[0],np.shape(im_or)[1]))
    predictedmask[predcoordy,predcoordx]=label+1
    
    return predictedmask

#%%
subsample=3
predicted_label = clf_model.predict(featurematrix_norm)

lngmask=predmask(im_or_res,roi,subsample,predicted_label,0)
ggomask=predmask(im_or_res,roi,subsample,predicted_label,1)
conmask=predmask(im_or_res,roi,subsample,predicted_label,2)


#%%


# zz1=predictedmaskinf[0,:,:,:]
# kmn=np.argmax(zz1,axis=-1)


    
    # scale=1  
    # im_or2=cv.resize(im_or,(512//scale,512//scale), 
    #                     interpolation = cv.INTER_AREA)
    
    # final_mask=regionsegmentation(im_or2,scale)




