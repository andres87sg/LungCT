# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 14:20:58 2021

@author: Andres

install xlrd
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


archivo = 'C:/Users/Andres/Desktop/DistanceMeasures.xlsx'

df = pd.read_excel(archivo, sheet_name='diameter')
#%%
#df.describe()

a=df.iloc[:,0]
c=df.iloc[:,1]

plt.figure(frameon=True,figsize=(3, 4), dpi=200)
sns.boxplot(y=df['ow'],x=df['scr'],width= 0.6,whis=1)
plt.xlabel("")
plt.ylabel("Orbital width")
plt.ylim(27,40)
#fig.set_size_inches(4, 4)