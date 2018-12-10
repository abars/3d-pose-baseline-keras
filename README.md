## 3d-pose-baseline-keras

Keras implementation of 3d-pose-baseline.

<img src="https://github.com/abars/3d-pose-baseline-keras/blob/master/images/running.jpg" width="50%" height="50%">

from WiderFaceDataset

# Demo

## Download Model

`python download_model.py`

## Predict

Predict using Caffe OpenPose and Keras

`python predict.py`

<img src="https://github.com/abars/3d-pose-baseline-keras/blob/master/images/running.png" width="75%" height="75%">

# How to Train

## Create Dataset

Dump training data from 3d-pose-baseline using export_dataset.py

https://github.com/ArashHosseini/3d-pose-baseline

## Check Exported Data

`python plot.py`

<img src="https://github.com/abars/3d-pose-baseline-keras/blob/master/images/plot.png" width="50%" height="50%">

from WiderFaceDataset

## Train

Training using Keras

`python train.py`

This is a pretrained output

http://www.abars.biz/keras/3d-pose-baseline.hdf5

# About 3d-pose-baseline

## Architecture

3d-pose-baseline predict 3d pose from 2d pose.

Input is 16 keypoint. Each keypoint has 2 axis.

Output is 16 keypoint. Eash keypoint has 3 axis.

Output should be denormalize using  mean value.

Mean value has 32 keypoint, So you should remove unused dimension. Mean value is sparse.

## Original work

https://github.com/ArashHosseini/3d-pose-baseline

# OpenPose to 3dpose

## Related work

https://github.com/miu200521358/3d-pose-baseline-vmd/blob/master/src/openpose_3dpose_sandbox_vmd.py
