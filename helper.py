# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 20:06:02 2022

@author: Baseet
"""


import numpy as np
from skimage import transform


def Load_Images(img):
  pred_img = np.array(img).astype('float32')/255
  pred_img = transform.resize(pred_img,(200,200,3))
  pred_img = np.expand_dims(pred_img,axis=0)
  return pred_img
