# -*- coding: utf-8 -*-
"""
Created on Thu May 27 13:39:27 2021
Este script está encargado de entrenar el modelo de knn
@author: Andres
"""
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import PIL
import pandas as pd

import scipy as sp
from scipy.stats import skew,kurtosis

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

true_labels=df['class'].values

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
plt.title('mean vs std')
plt.xlabel('mean')
plt.ylabel('median')

plt.plot(centroids2[0][0],centroids2[0][2], 'r*')
plt.plot(centroids2[1][0],centroids2[1][2], 'b*')
plt.plot(centroids2[2][0],centroids2[2][2], 'r*')
#plt.plot(centroids[3][0],centroids[3][1], 'r*')

#%%

i=30

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

#%%
import cv2 as cv

mask1=segmentationmask(im_or,centroids2,0)
mask2=segmentationmask(im_or,centroids2,1)
mask3=segmentationmask(im_or,centroids2,2)

kernel = np.ones((3,3),np.uint8)

closing = cv.morphologyEx(mask2, cv.MORPH_CLOSE, kernel)
opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)

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
from sklearn import preprocessing

features_matrix=X.copy()
scaler = preprocessing.StandardScaler().fit(features_matrix)
features_matrix_scal=scaler.transform(features_matrix)
X=features_matrix_scal.copy()
y=true_labels


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

#%%
bestmodel=mm[6]

joblib_file = "KNNmodel.pkl"
joblib.dump(bestmodel, joblib_file)



#%%


# import the necessary packages
from skimage.exposure import rescale_intensity
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io
import numpy as np
import argparse
import cv2

i=10

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

kk=np.zeros((512,512,3))

kk[:,:,0]=im_or
kk[:,:,1]=im_or
kk[:,:,2]=im_or

kk = kk[100:400,100:400]
cropmask = cropmask[100:400,100:400]

segments = slic(img_as_float(kk), n_segments=1000,sigma=2,start_label=1)
plt.imshow(segments,cmap='gray')
plt.imshow(im_or,cmap='gray')

#%%

zz1=segments*cropmask
plt.imshow(zz1,cmap='gray')

labels = np.unique(zz1)

#zz2=im_or(segments==labels[1])

#%%
statslist=[]
for ind in range(1,24):
    mean_gl = np.mean(im_or[zz1==labels[ind]])
    med_gl  = np.median(im_or[zz1==labels[ind]])
    std_gl  = np.std(im_or[zz1==labels[ind]])
    kurt_gl = sp.stats.kurtosis(im_or[zz1==labels[ind]])
    skew_gl = sp.stats.skew(im_or[zz1==labels[ind]])
    
    
    statist = [mean_gl,med_gl,std_gl,kurt_gl,skew_gl]
    statslist.append(statist)
    
    
X=np.array(statslist)
    
features_matrix=X.copy()
scaler = preprocessing.StandardScaler().fit(features_matrix)
features_matrix_scal=scaler.transform(features_matrix)
X=features_matrix_scal.copy()
y=true_labels




#X1=[mean_gl,med_gl,std_gl,kurt_gl,skew_gl]
#X1=[med_gl,std_gl,kurt_gl,skew_gl]
#op=np.array(X1).reshape(1,-1)
esto=model.predict(X)
print(esto)
#print(esto)


#%%

# alpha = 0.6
# overlay = np.dstack([vis] * 3)
# output = orig.copy()
# cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

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



