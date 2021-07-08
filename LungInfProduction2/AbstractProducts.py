import pickle
import joblib
from tensorflow.keras.models import load_model

def load_mdl_lungsegmentation():
    path='C:/Users/Andres/Desktop/CTClassif/'
    mdlfilename='lng_seg_mdl.h5'
    mdl_lungsegmentation=load_model(path+mdlfilename)
    return mdl_lungsegmentation

def load_mdl_infsegmentation():
    path='C:/Users/Andres/Desktop/CTClassif/'
    mdlfilename = 'LungInf_SF1_Filt32_03072021_Python.h5'
    mdl_infsegmentation = load_model(path+mdlfilename)
    return mdl_infsegmentation