# ----------------------------------------------
# OpenPose Position to 3D Position
# ----------------------------------------------

import cv2
import sys
import numpy as np
import pandas as pd
import os
import caffe
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from mpl_toolkits.mplot3d import Axes3D
from keras.models import load_model

# ----------------------------------------------
# Setting
# ----------------------------------------------

MODEL_HDF5 = "3d-pose-baseline.hdf5"
IMAGE_PATH = "images/running.jpg"
CAFFE_MODEL = 'pose_iter_440000.caffemodel'
PROTOTXT = 'pose_deploy.prototxt'

# ----------------------------------------------
# Input Data
# ----------------------------------------------

IMAGE_WIDTH=368
IMAGE_HEIGHT=368

if len(sys.argv) >= 5:
  CAFFE_MODEL = sys.argv[1]
  PROTOTXT = sys.argv[2]
  IMAGE_WIDTH = int(sys.argv[3])
  IMAGE_HEIGHT = int(sys.argv[4])

if not os.path.exists(IMAGE_PATH):
	print(IMAGE_PATH+" not found")
	sys.exit(1)

print IMAGE_PATH
input_img = cv2.imread(IMAGE_PATH)

img = cv2.resize(input_img, (IMAGE_WIDTH, IMAGE_HEIGHT))

img = img[...,::-1]  #BGR 2 RGB

data = np.array(img, dtype=np.float32)
data.shape = (1,) + data.shape

data = data / 255.0

net  = caffe.Net(PROTOTXT, CAFFE_MODEL, caffe.TEST)
data = data.transpose((0, 3, 1, 2))
out = net.forward_all(data = data)

paf = out[net.outputs[0]]
confidence = out[net.outputs[1]]

# ----------------------------------------------
# Display output
# ----------------------------------------------

points = []
threshold = 0.1

for i in range(confidence.shape[1]):
	probMap = confidence[0, i, :, :]
	minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

	x = (input_img.shape[1] * point[0]) / confidence.shape[3]
	y = (input_img.shape[0] * point[1]) / confidence.shape[2] 
 
	if prob > threshold : 
		points.append(x)
		points.append(y)
	else :
		points.append(0)
		points.append(0)

# ----------------------------------------------
# Convert format
# ----------------------------------------------

with h5py.File('3d-pose-baseline-mean.h5', 'r') as f:
  data_mean_2d = np.array(f['data_mean_2d'])
  data_std_2d = np.array(f['data_std_2d'])
  data_mean_3d = np.array(f['data_mean_3d'])
  data_std_3d = np.array(f['data_std_3d'])

H36M_NAMES = ['']*32
H36M_NAMES[0]  = 'Hip' #ignore when 3d
H36M_NAMES[1]  = 'RHip'
H36M_NAMES[2]  = 'RKnee'
H36M_NAMES[3]  = 'RFoot'
H36M_NAMES[6]  = 'LHip'
H36M_NAMES[7]  = 'LKnee'
H36M_NAMES[8]  = 'LFoot'
H36M_NAMES[12] = 'Spine'
H36M_NAMES[13] = 'Thorax'
H36M_NAMES[14] = 'Neck/Nose' #ignore when 2d
H36M_NAMES[15] = 'Head'
H36M_NAMES[17] = 'LShoulder'
H36M_NAMES[18] = 'LElbow'
H36M_NAMES[19] = 'LWrist'
H36M_NAMES[25] = 'RShoulder'
H36M_NAMES[26] = 'RElbow'
H36M_NAMES[27] = 'RWrist'

#mean list
h36m_2d_mean = [0,1,2,3,6,7,8,12,13,15,17,18,19,25,26,27]
h36m_3d_mean = [1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]

OPENPOSE_Nose = 0
OPENPOSE_Neck = 1
OPENPOSE_RightShoulder = 2
OPENPOSE_RightElbow = 3
OPENPOSE_RightWrist = 4
OPENPOSE_LeftShoulder = 5
OPENPOSE_LeftElbow = 6
OPENPOSE_LeftWrist = 7
OPENPOSE_RightHip = 8
OPENPOSE_RightKnee = 9
OPENPOSE_RightAnkle = 10
OPENPOSE_LeftHip = 11
OPENPOSE_LeftKnee = 12
OPENPOSE_LAnkle = 13
OPENPOSE_RightEye = 14
OPENPOSE_LeftEye = 15
OPENPOSE_RightEar = 16
OPENPOSE_LeftEar = 17
OPENPOSE_Background = 18

#openpose -> 3dpose baseline 2d
openpose_to_3dposebaseline=[-1,8,9,10,11,12,13,-1,1,0,5,6,7,2,3,4]

inputs = np.zeros(32)
for i in range(16):
	if openpose_to_3dposebaseline[i]==-1:
		continue
	inputs[i*2+0]=points[openpose_to_3dposebaseline[i]*2+0]
	inputs[i*2+1]=points[openpose_to_3dposebaseline[i]*2+1]

inputs[0*2+0] = (points[11*2+0]+points[8*2+0])/2
inputs[0*2+1] = (points[11*2+1]+points[8*2+1])/2
inputs[7*2+0] = (points[5*2+0]+points[2*2+0])/2
inputs[7*2+1] = (points[5*2+1]+points[2*2+1])/2

for i in range(16):
	j=h36m_2d_mean[i]
	inputs[i*2+0]=(inputs[i*2+0]-data_mean_2d[j*2+0])/data_std_2d[j*2+0]
	inputs[i*2+1]=(inputs[i*2+1]-data_mean_2d[j*2+1])/data_std_2d[j*2+1]

# ----------------------------------------------
# Predict
# ----------------------------------------------

keras_model = load_model(MODEL_HDF5)
keras_model.summary()
reshape_input = np.reshape(np.array(inputs),(1,32))
outputs = keras_model.predict(reshape_input,batch_size=1)[0]

# ----------------------------------------------
# Display result
# ----------------------------------------------

fig = plt.figure()
ax = Axes3D(fig)

IS_3D=False
cnt=0

X=[]
Y=[]
Z=[]

def search_name(name):
	j=0
	for i in range(32):
		if(IS_3D):
			if(H36M_NAMES[i]=="Hip"):
				continue
		else:
			if(H36M_NAMES[i]=="Neck/Nose"):
				continue
		if(H36M_NAMES[i]==""):
			continue
		if(H36M_NAMES[i]==name):
			return j
		j=j+1
	return -1

def draw_connect(from_id,to_id,color="#00aa00"):
	from_id=search_name(from_id)
	to_id=search_name(to_id)
	if(from_id==-1 or to_id==-1):
		return
	x = [X[from_id], X[to_id]]
	y = [Y[from_id], Y[to_id]]
	z = [Z[from_id], Z[to_id]]

	ax.plot(x, z, y, "o-", color=color, ms=4, mew=0.5)

def plot(data):
	plt.cla()

	ax.set_xlabel('X axis')
	ax.set_ylabel('Z axis')
	ax.set_zlabel('Y axis')
	ax.set_zlim([600, -600])

	global cnt,X,Y,Z,IS_3D
	k=cnt

	for mode in range(2):
		X=[]
		Y=[]
		Z=[]

		if(mode==0):
			IS_3D=True
		else:
			IS_3D=False

		for i in range(16):
			if IS_3D:
				j=h36m_3d_mean[i]
				X.append(outputs[i*3+0]*data_std_3d[j*3+0]+data_mean_3d[j*3+0])
				Y.append(outputs[i*3+1]*data_std_3d[j*3+1]+data_mean_3d[j*3+1])
				Z.append(outputs[i*3+2]*data_std_3d[j*3+2]+data_mean_3d[j*3+2])
			else:
				j=h36m_2d_mean[i]
				X.append(inputs[i*2+0]*data_std_2d[j*2+0]+data_mean_2d[j*2+0])
				Y.append(inputs[i*2+1]*data_std_2d[j*2+1]+data_mean_2d[j*2+1])
				Z.append(0)

		if(IS_3D):
			draw_connect("Head","Thorax","#0000aa")
			draw_connect("Thorax",'RShoulder')
			draw_connect('RShoulder','RElbow')
			draw_connect('RElbow','RWrist')
			draw_connect("Thorax",'LShoulder')
			draw_connect('LShoulder','LElbow')
			draw_connect('LElbow','LWrist')
			draw_connect('Thorax','Spine')
			draw_connect('Spine','LHip')
			draw_connect('Spine','RHip')
			draw_connect('RHip','RKnee')
			draw_connect('RKnee','RFoot')
			draw_connect('LHip','LKnee')
			draw_connect('LKnee','LFoot')
		else:
			draw_connect("Head","Thorax","#0000ff")
			draw_connect("Thorax",'RShoulder',"#00ff00")
			draw_connect('RShoulder','RElbow',"#00ff00")
			draw_connect('RElbow','RWrist',"#00ff00")
			draw_connect("Thorax",'LShoulder',"#00ff00")
			draw_connect('LShoulder','LElbow',"#00ff00")
			draw_connect('LElbow','LWrist',"#00ff00")
			draw_connect('Thorax','Spine',"#00ff00")
			draw_connect('Spine','Hip',"#00ff00")
			draw_connect('Hip','LHip',"#ff0000")
			draw_connect('Hip','RHip',"#ff0000")
			draw_connect('RHip','RKnee',"#ff0000")
			draw_connect('RKnee','RFoot',"#ff0000")
			draw_connect('LHip','LKnee',"#ff0000")
			draw_connect('LKnee','LFoot',"#ff0000")

	#cnt=cnt+1

ani = animation.FuncAnimation(fig, plot, interval=1000)
plt.show()
