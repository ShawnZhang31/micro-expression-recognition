"""
Facial Micro Expression Spatial and Temproal Features
"""
import os
import cv2
import dlib
import imageio
import sklearn
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential
from keras import backend as K
import sys

# show the versions of the important packages 
CV_VERSION = cv2.__version__
DLIB_VERSION = dlib.__version__
Scikit_Learn_VERSION = sklearn.__version__
KERAS_VERSION = keras.__version__

print("Versions of the key packages:\n      opencv:{}\n     dlib:{}     \n      scikit-learn:{}\n       keras:{}"
        .format(CV_VERSION, DLIB_VERSION, Scikit_Learn_VERSION, KERAS_VERSION))


K.set_image_dim_ordering('th')

image_rows, image_columns, image_depth = 64, 64, 96







