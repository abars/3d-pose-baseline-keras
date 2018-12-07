## 3d-pose-baseline-keras

Underconstructing

Keras implementation of 3d-pose-baseline.

# Architecture

Dump training data from 3d-pose-baseline

https://github.com/ArashHosseini/3d-pose-baseline

Training using Keras

`pyrhon train.py` (implementing)

Prediction using Keras

`python plot.py`

<img src="https://github.com/abars/3d-pose-baseline-keras/blob/master/plot.png" width="50%" height="50%">

# About 3d-pose-baseline

3d-pose-baseline predict 3d pose from 2d pose.

Input is 16 keypoint. Each keypoint has 2 axis.

Output is 16 keypoint. Eash keypoint has 3 axis.

Output should be denormalize using  mean value.

Mean value has 32 keypoint, So you should remove unused dimension. Mean value is sparse.

# Original work

https://github.com/ArashHosseini/3d-pose-baseline

