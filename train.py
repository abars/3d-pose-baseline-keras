# ----------------------------------------------
# Train 3dpose baseline
# ----------------------------------------------

import os.path,sys
import numpy as np
import h5py

os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers import BatchNormalization
from keras.layers import InputLayer
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,AveragePooling2D,Input,Add
from keras.layers import SeparableConv2D
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet import MobileNet
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import layers
from keras.optimizers import SGD

import keras.callbacks

# ----------------------------------------------
# Setting
# ----------------------------------------------

DATASET_PATH='../3d-pose-baseline-master'
MODEL_HDF5 = "3d-pose-baseline.hdf5"
PLOT_FILE = "3d-pose-baseline.png"

# ----------------------------------------------
# Import Dataset
# ----------------------------------------------

with h5py.File(DATASET_PATH+"/train.h5", 'r') as f:
  x_train = np.array(f['encoder_inputs'])
  y_train = np.array(f['decoder_outputs'])
  data_mean_2d = np.array(f['data_mean_2d'])
  data_std_2d = np.array(f['data_std_2d'])
  data_mean_3d = np.array(f['data_mean_3d'])
  data_std_3d = np.array(f['data_std_3d'])

with h5py.File(DATASET_PATH+"/test.h5", 'r') as f:
  x_test = np.array(f['encoder_inputs'])
  y_test = np.array(f['decoder_outputs'])

# ----------------------------------------------
# Model
# ----------------------------------------------

model = Sequential()

input_size=32
output_size=48
linear_size=1024
dropout_rate=0.5

SIMPLE_MODEL=False

if(SIMPLE_MODEL):
  #pre-processing
  inputs = Input(shape=(input_size,))

  x1=Dense(linear_size)(inputs)
  y1=BatchNormalization()(x1)
  y1=Activation("relu")(y1)
  x2=Dropout(dropout_rate)(y1)

  y2=Dense(linear_size)(x2)
  y2=BatchNormalization()(y2)
  y2=Activation("relu")(y2)
  y2=Dropout(dropout_rate)(y2)

  #output
  predictions=Dense(output_size)(y2)

  model = Model(input=inputs, output=predictions)
else:
  #pre-processing
  inputs = Input(shape=(input_size,))

  x1=Dense(linear_size)(inputs)
  y1=BatchNormalization()(x1)
  y1=Activation("relu")(y1)
  x2=Dropout(dropout_rate)(y1)

  #stage1
  y2=Dense(linear_size)(x2)
  y2=BatchNormalization()(y2)
  y2=Activation("relu")(y2)
  y2=Dropout(dropout_rate)(y2)

  y2=Dense(linear_size)(y2)
  y2=BatchNormalization()(y2)
  y2=Activation("relu")(y2)
  y2=Dropout(dropout_rate)(y2)

  x3=Add()([x2,y2])

  #stage2
  y3=Dense(linear_size)(x3)
  y3=BatchNormalization()(y3)
  y3=Activation("relu")(y3)
  y3=Dropout(dropout_rate)(y3)

  y3=Dense(linear_size)(y3)
  y3=BatchNormalization()(y3)
  y3=Activation("relu")(y3)
  y3=Dropout(dropout_rate)(y3)

  x4=Add()([x3,y3])

  #output
  predictions=Dense(output_size)(x4)

  model = Model(input=inputs, output=predictions)

model.summary()

# ----------------------------------------------
# Input Data Preprocessing
# ----------------------------------------------

#data_mean = np.mean(complete_data, axis=0)
#data_std  =  np.std(complete_data, axis=0)

# ----------------------------------------------
# Data
# ----------------------------------------------

preprocessing_function=None

#x_train=np.zeros(input_size)
#y_train=np.zeros(output_size)

#x_test=np.zeros(input_size)
#y_test=np.zeros(output_size)

#x_train = np.reshape(x_train, (1, input_size))
#y_train = np.reshape(y_train, (1, output_size))

#x_test = np.reshape(x_test, (1, input_size))
#y_test = np.reshape(y_test, (1, output_size))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error',
              optimizer=sgd,
              metrics=['accuracy'])

fit = model.fit(x_train, y_train,
          epochs=20,
          batch_size=128,
          validation_data=(x_test, y_test))

model.save(MODEL_HDF5)

# ----------------------------------------------
# Plot
# ----------------------------------------------

import matplotlib.pyplot as plt

fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))

# loss
def plot_history_loss(fit):
    # Plot the loss in the history
    axL.plot(fit.history['loss'],label="loss for training")
    axL.plot(fit.history['val_loss'],label="loss for validation")
    axL.set_title('model loss')
    axL.set_xlabel('epoch')
    axL.set_ylabel('loss')
    axL.legend(loc='upper right')

# acc
def plot_history_acc(fit):
    # Plot the loss in the history
    axR.plot(fit.history['acc'],label="accuracy for training")
    axR.plot(fit.history['val_acc'],label="accuracy for validation")
    axR.set_title('model accuracy')
    axR.set_xlabel('epoch')
    axR.set_ylabel('accuracy')
    axR.legend(loc='upper right')

plot_history_loss(fit)
plot_history_acc(fit)
fig.savefig(PLOT_FILE)
plt.close()
