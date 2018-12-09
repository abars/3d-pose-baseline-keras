# ----------------------------------------------
#Display 3dpose baseline dataset
# ----------------------------------------------

import os
import numpy as np
import h5py

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

DATASET_PATH='../3d-pose-baseline-master'

with h5py.File(DATASET_PATH+'/train.h5', 'r') as f:
  inputs = np.array(f['encoder_inputs'])
  outputs = np.array(f['decoder_outputs'])
  data_mean_2d = np.array(f['data_mean_2d'])
  data_std_2d = np.array(f['data_std_2d'])
  data_mean_3d = np.array(f['data_mean_3d'])
  data_std_3d = np.array(f['data_std_3d'])

print(inputs.shape)
print(outputs.shape)

print(data_mean_2d.shape)
print(data_std_2d.shape)

H36M_NAMES = ['']*32
H36M_NAMES[0]  = 'Hip'
H36M_NAMES[1]  = 'RHip'
H36M_NAMES[2]  = 'RKnee'
H36M_NAMES[3]  = 'RFoot'
H36M_NAMES[6]  = 'LHip'
H36M_NAMES[7]  = 'LKnee'
H36M_NAMES[8]  = 'LFoot'
H36M_NAMES[12] = 'Spine'
H36M_NAMES[13] = 'Thorax'
H36M_NAMES[14] = 'Neck/Nose'
H36M_NAMES[15] = 'Head'
H36M_NAMES[17] = 'LShoulder'
H36M_NAMES[18] = 'LElbow'
H36M_NAMES[19] = 'LWrist'
H36M_NAMES[25] = 'RShoulder'
H36M_NAMES[26] = 'RElbow'
H36M_NAMES[27] = 'RWrist'

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

		IS_NORMALIZE=True
		if(mode==0):
			IS_3D=True
		else:
			IS_3D=False

		for i in range(16):
			for j in range(32):
				if(search_name(H36M_NAMES[j])==i):
					break

			if IS_NORMALIZE:
				if IS_3D:
					X.append(outputs[k,i*3+0]*data_std_3d[j*3+0]+data_mean_3d[j*3+0])
					Y.append(outputs[k,i*3+1]*data_std_3d[j*3+1]+data_mean_3d[j*3+1])
					Z.append(outputs[k,i*3+2]*data_std_3d[j*3+2]+data_mean_3d[j*3+2])
				else:
					X.append(inputs[k,i*2+0]*data_std_2d[j*2+0]+data_mean_2d[j*2+0])
					Y.append(inputs[k,i*2+1]*data_std_2d[j*2+1]+data_mean_2d[j*2+1])
					Z.append(0)
			else:
				if IS_3D:
					X.append(outputs[k,i*3+0])
					Y.append(outputs[k,i*3+1])
					Z.append(outputs[k,i*3+2])
				else:
					X.append(inputs[k,i*2+0])
					Y.append(inputs[k,i*2+1])
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
			draw_connect('Spine','LHip',"#aa0000")
			draw_connect('Spine','RHip',"#aa0000")
			draw_connect('RHip','RKnee',"#aa0000")
			draw_connect('RKnee','RFoot',"#aa0000")
			draw_connect('LHip','LKnee',"#aa0000")
			draw_connect('LKnee','LFoot',"#aa0000")
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

	cnt=cnt+1

ani = animation.FuncAnimation(fig, plot, interval=10)
plt.show()
