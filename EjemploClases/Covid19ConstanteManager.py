import os
from os.path import join

COVID19_PATH_WEIGHTS = join(os.path.dirname(os.path.realpath(__file__)), 'pretrained/model.meta_eval')
COVID19_PATH_MODEL_METADATA = join(os.path.dirname(os.path.realpath(__file__)), 'pretrained/model-2069')
COVID19_PATH_SAVE_VISUAL_RESPONSE = '/tmp/'