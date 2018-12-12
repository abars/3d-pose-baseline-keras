# ----------------------------------------------
# Convert 3dpose baseline to caffemodel
# ----------------------------------------------

import os
os.environ['GLOG_minloglevel'] = '2' 

import caffe
import cv2
import numpy as np
import h5py

from keras.preprocessing import image
from keras.models import load_model

import keras2caffe

import tensorflow as tf

#mean value
with h5py.File('3d-pose-baseline-mean.h5', 'r') as f:
  data_mean_2d = np.array(f['data_mean_2d'])
  data_std_2d = np.array(f['data_std_2d'])
  data_mean_3d = np.array(f['data_mean_3d'])
  data_std_3d = np.array(f['data_std_3d'])

mean_and_std="";
mean_and_std=mean_and_std+"float data_mean_2d[32]={"
for i in range(32):
	mean_and_std=mean_and_std+str(data_mean_2d[i])+","
mean_and_std=mean_and_std+"};\n"

mean_and_std=mean_and_std+"float data_std_2d[32]={"
for i in range(32):
	mean_and_std=mean_and_std+str(data_std_2d[i])+","
mean_and_std=mean_and_std+"};\n"

mean_and_std=mean_and_std+"float data_mean_3d[32]={"
for i in range(48):
	mean_and_std=mean_and_std+str(data_mean_3d[i])+","
mean_and_std=mean_and_std+"};\n"

mean_and_std=mean_and_std+"float data_std_3d[32]={"
for i in range(48):
	mean_and_std=mean_and_std+str(data_std_3d[i])+","
mean_and_std=mean_and_std+"};\n"

print(mean_and_std)

#converting
keras_model = load_model('3d-pose-baseline.hdf5')
keras_model.summary()

keras2caffe.convert(keras_model, '3d-pose-baseline.prototxt', '3d-pose-baseline.caffemodel')

net  = caffe.Net('3d-pose-baseline.prototxt', '3d-pose-baseline.caffemodel', caffe.TEST)

data = np.random.rand(32)
data = np.reshape(np.array(data),(1,32))

#verify
pred = keras_model.predict(data)[0]
print(pred)

out = net.forward_all(data = data)
pred = out['dense_6']
print(pred)
