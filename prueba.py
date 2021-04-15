# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 17:22:21 2021

@author: Andres
"""


import numpy as np

import skimage as ski
from skimage.measure import label, regionprops
#from skimage import mesh_surface_area


A=np.array(([1,1,0],[1,1,0],[0,0,0],[0,0,1],[1,1,1]))

label_image = label(A,connectivity=2)

