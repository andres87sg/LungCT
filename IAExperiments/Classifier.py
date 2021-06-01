# -*- coding: utf-8 -*-
"""
Created on Mon May 31 10:22:31 2021

@author: Andres
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

X, y = datasets.load_iris(return_X_y=True)
X.shape, y.shape
((150, 4), (150,))

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.4, 
                                                    random_state=0)

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
#clf = svm.SVC(kernel='linear', C=1, random_state=42)
#scores = cross_val_score(clf, X, y, cv=5)

mm=[]
scores=[]
kf = KFold(n_splits=10)
for train, test in kf.split(X):
    print('Train: %s | test: %s' % (train, test))
    clf = svm.SVC(kernel='linear', C=1).fit(X[train], y[train])
    mm.append(clf)
    #print(clf)
    sco=clf.score(X[test],y[test])
    scores.append(sco)
    print(sco)
    #scores = cross_val_score(clf, X[test], y[test], cv=5)