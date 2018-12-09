## 3d-pose-baseline-keras

Underconstructing

Keras implementation of 3d-pose-baseline.

# Demo

## Download Model

`python download_model.py`

## Predict

Predict using Caffe OpenPose and Keras

`python predict.py` (implementing)

# How to Train

## Create Dataset

Dump training data from 3d-pose-baseline using export_dataset.py

https://github.com/ArashHosseini/3d-pose-baseline

## Check Exported Data

`python plot.py`

<img src="https://github.com/abars/3d-pose-baseline-keras/blob/master/plot.png" width="50%" height="50%">

## Train

Training using Keras

`pyrhon train.py` (implementing)

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
