# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 15:02:57 2021

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

import scipy as sp
from scipy.stats import skew,kurtosis

import skimage
from skimage.feature import greycomatrix, greycoprops

import random

#%% Load and save
#df.to_csv ('export_dataframe.csv', index = False, header=True)
df = pd.read_csv('export_dataframe.csv')

#%%

is_one=df.loc[:,'class']==63
dfclass_one=df.loc[is_one]

is_two=df.loc[:,'class']==127
dfclass_two=df.loc[is_two]

is_three=df.loc[:,'class']==255
dfclass_three=df.loc[is_three]

class_one=dfclass_one.iloc[:,1:6].values
class_two=dfclass_two.iloc[:,1:6].values
class_three=dfclass_three.iloc[:,1:6].values

label_one = dfclass_one.iloc[:,0].values
label_two = dfclass_two.iloc[:,0].values
label_three = dfclass_three.iloc[:,0].values


random.seed(1)
ind_class_one=random.sample(range(len(class_one)),5000)

random.seed(1)
ind_class_two=random.sample(range(len(class_two)),5000)

random.seed(1)
ind_class_three=random.sample(range(len(class_three)),5000)



features_matrix=np.concatenate((class_one[ind_class_one],
                                class_two[ind_class_two],
                                class_three[ind_class_three]
                                ),axis=0)
labels = np.concatenate((label_one[ind_class_one],
                         label_two[ind_class_two],
                         label_three[ind_class_three]),axis=0)

#%%

from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

scaler = preprocessing.StandardScaler().fit(features_matrix)
features_matrix_scal=scaler.transform(features_matrix)
X=features_matrix_scal.copy()
y=np.zeros(np.shape(labels))

y[labels==63]=0
y[labels==127]=1
y[labels==255]=2

#%%

mm=[]
scores=[]
k=0
listaclf=[]


kf = KFold(n_splits=10,shuffle=True)
for train, test in kf.split(X):
    k=k+1
    print(k)
    #print('Train: %s | test: %s' % (train, test))
    clf = svm.SVC(kernel='rbf', C=3).fit(X[train], y[train])
    mm.append(clf)
    listaclf.append(k)
    #print(clf)
    sco=clf.score(X[test],y[test])
    scores.append(sco)
    #print(sco)
    #scores = cross_val_score(clf, X[test], y[test], cv=5)
    
scores==np.max(scores)

print(scores)

zmn=listaclf*(scores==np.max(scores))
bestindclf=np.max(zmn)-1
model=mm[bestindclf]

#%%
import pickle
  
# Save the trained model as a pickle string.
pkl_filename = "pickle_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)

    
#%%

import pickle

pkl_filename = "pickle_model.pkl"

with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)


#%% Read Features from testin set

df = pd.read_csv('test_dataframe.csv')

#%%

is_one=df.loc[:,'class']==63
dfclass_one=df.loc[is_one]

is_two=df.loc[:,'class']==127
dfclass_two=df.loc[is_two]

is_three=df.loc[:,'class']==255
dfclass_three=df.loc[is_three]

class_one=dfclass_one.iloc[:,1:6].values
class_two=dfclass_two.iloc[:,1:6].values
class_three=dfclass_three.iloc[:,1:6].values

label_one = dfclass_one.iloc[:,0].values
label_two = dfclass_two.iloc[:,0].values
label_three = dfclass_three.iloc[:,0].values


random.seed(1)
ind_class_one=random.sample(range(len(class_one)),500)

random.seed(1)
ind_class_two=random.sample(range(len(class_two)),500)

random.seed(1)
ind_class_three=random.sample(range(len(class_three)),500)



features_matrix=np.concatenate((class_one[ind_class_one],
                                class_two[ind_class_two],
                                class_three[ind_class_three]
                                ),axis=0)
labels = np.concatenate((label_one[ind_class_one],
                         label_two[ind_class_two],
                         label_three[ind_class_three]),axis=0)

#%%

scaler = preprocessing.StandardScaler().fit(features_matrix)
features_matrix_scal=scaler.transform(features_matrix)
X=features_matrix_scal.copy()
y=np.zeros(np.shape(labels))


y[labels==63]=0
y[labels==127]=1
y[labels==255]=2

#%%

predicted_label = pickle_model.predict(X)
true_label=y.copy()

#%%
from sklearn.metrics import confusion_matrix, classification_report

print("Classification report")
print(confusion_matrix(true_label, predicted_label))
print(classification_report(true_label, predicted_label, digits=3))

