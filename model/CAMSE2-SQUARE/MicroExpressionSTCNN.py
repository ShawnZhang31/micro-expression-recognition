"""
Facial Micro Expression Spatial and Temproal Features
"""
import os
import numpy as np
import cv2
import dlib
import imageio
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils, generic_utils
from keras import backend as K
import sys
import argparse

# show the versions of the important packages 
CV_VERSION = cv2.__version__
DLIB_VERSION = dlib.__version__
Scikit_Learn_VERSION = sklearn.__version__
KERAS_VERSION = keras.__version__

print("Versions of the key packages:\n\
        opencv:{}\n\
        dlib:{}\n\
        scikit-learn:{}\n\
        keras:{}"
        .format(CV_VERSION, 
                DLIB_VERSION, 
                Scikit_Learn_VERSION, 
                KERAS_VERSION))

# initiate the parser
parser = argparse.ArgumentParser()

# add long and short argument
parser.add_argument("--angrypath", '-angry', help="set angry micro-expression data path")
parser.add_argument("--happypath", '-happy', help="set happy micro-expression data path")
parser.add_argument("--disgustpath", '-disgust', help="set disgust micro-expression data path")
# check for all data path
args = parser.parse_args()

K.set_image_dim_ordering('th')

image_rows, image_columns, image_depth = 64, 64, 96

# # training and test datasets
# training_list = []
# if args.angrypath:
#         angrypath = args.angrypath
# else:
#         raise Exception("angrypath must be specified")
#         sys.exit(1)

# if args.happypath:
#         happypath = args.happypath
# else:
#         raise Exception("happypath must be specified")
#         sys.exit(1)

# if args.disgustpath:
#         disgustpath = args.disgustpath
# else:
#         raise Exception("disgustpath must be specified")
#         sys.exit(1)

# # check all videos
# for path in (angrypath, happypath, disgustpath):
#         for videos in os.listdir(path):
#                 frames = []
#                 videopath = path + videos
#                 loadedvideo = imageio.get_reader(videopath, 'ffmpeg')
#                 framerange = [x + 72 for x in range(96)]
#                 for frame in framerange:
#                         image = loadedvideo.get_data(frame)
#                         imageresize = cv2.resize(image, (image_rows, image_columns), interpolation=cv2.INTER_AREA)
#                         grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
#                         frames.append(grayimage)
#                 frames = np.asarray(frames)
#                 videoarray = np.rollaxis(np.rollaxis(frames, 2, 0), 2, 0)
#                 training_list.append(videoarray)

# training_list = np.asarray(training_list)
# trainingsamples = len(training_list)

# traininglabels = np.zeros((trainingsamples, ), dtype=np.int)

# traininglabels[0:76] = 0
# traininglabels[76:170] = 1
# traininglabels[170:206] = 2

# traininglabels = np_utils.to_categorical(traininglabels, 3)

# training_data = [training_list, traininglabels]
# (trainingframes, traininglabels) = (training_data[0], training_data[1])
# training_set = np.zeros((trainingsamples, 1, image_rows, image_columns, image_depth))

# for h in range(trainingsamples):
#         training_set[h][0][:][:][:] = trainingframes[h, :, :, :]

# # normalize the training data
# traing_set = traing_set.astype('float32')
# traing_set -= np.mean(training_set)
# traing_set /= numpy.max(training_set)

# # save traing images and labels in numpy array
# np.save("./../../datasets/CAS(ME)^2/CASME-SQUARE/microexpstcnn_images.npy", training_set)
# np.save("./../../datasets/CAS(ME)^2/CASME-SQUARE/microexpstcnn_labels.npy", traininglabels)

# #load traing images and labels that are stored in numpy array
# training_set = np.load("./datasets/CAS(ME)^2/CASME-SQUARE/microexpstcnn_images.npy")
# traininglabels = np.load("./datasets/CAS(ME)^2/CASME-SQUARE/microexpstcnn_labels.npy")

# MicroExpressionSTCNN Model
model = Sequential()
model.add(Convolution3D(32, (3, 3, 15), input_shape=(1, image_rows, image_columns, image_depth), activation='relu'))
model.add(MaxPooling3D(pool_size=(3, 3, 3)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, init='normal', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, init='normal'))
model.add(Activation('softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'SGD', metrics = ['accuracy'])

model.summary()

# Load pre-trained weights
"""
model.load_weights('./datasets/CAS(ME)^2/CASME-SQUARE/weights-improvement-53-0.88.hdf5')
"""

filepath="weights_microexpstcnn/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Spliting the dataset into training and validation sets
train_images, validation_images, train_labels, validation_labels =  train_test_split(training_set, traininglabels, test_size=0.2, random_state=42)

# Save validation set in a numpy array
"""
numpy.save('numpy_validation_dataset/microexpstcnn_val_images.npy', validation_images)
numpy.save('numpy_validation_dataset/microexpstcnn_val_labels.npy', validation_labels)
"""

# Load validation set from numpy array
"""
validation_images = numpy.load('numpy_validation_datasets/microexpstcnn_val_images.npy')
validation_labels = numpy.load('numpy_validation_datasets/microexpstcnn_val_labels.npy')
"""

# Training the model
hist = model.fit(train_images, train_labels, validation_data = (validation_images, validation_labels), callbacks=callbacks_list, batch_size = 16, nb_epoch = 100, shuffle=True)

# Finding Confusion Matrix using pretrained weights
"""
predictions = model.predict(validation_images)
predictions_labels = numpy.argmax(predictions, axis=1)
validation_labels = numpy.argmax(validation_labels, axis=1)
cfm = confusion_matrix(validation_labels, predictions_labels)
print (cfm)
"""

