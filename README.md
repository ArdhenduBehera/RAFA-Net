# RAFA-NET
Code used in the research of estimating head pose orientation from RGB images. Code to build and train RAFA-Net are provided. The model takes an input image with a face bounding box and outputs the yaw, pitch and roll of the persons head in radians.

<kbd>![RAFA-NET Model Overview](https://github.com/ZWharton15/RAFA-NET-1/blob/master/doc/RAFA_model_overview.png?raw=true)</kbd>

## Published Results
<kbd>![AFLW validation results](https://github.com/ZWharton15/RAFA-NET-1/blob/master/doc/Table1.JPG?raw=true)</kbd>
<kbd>![BIWI validation results](https://github.com/ZWharton15/RAFA-NET-1/blob/master/doc/Table2.JPG?raw=true)</kbd>

## Grad-CAM outputs of the 3 angles
<kbd>![Grad-CAM Output](https://github.com/ZWharton15/RAFA-NET-1/blob/master/doc/Grad-CAM.png?raw=true)</kbd>

## Paper
[Final ACCV Version](https://openaccess.thecvf.com/content/ACCV2020/papers/Behera_Rotation_Axis_Focused_Attention_Network_RAFA-Net_for_Estimating_Head_Pose_ACCV_2020_paper.pdf)

## Abstract
Head pose is a vital indicator of human attention and behavior. Therefore, automatic estimation of head pose from images is key to many applications. In this paper, we propose a novel approach for head pose estimation from a single RGB image. Many existing approaches often predict head poses by localizing facial landmarks and then solve 2D to 3D correspondence problem with a mean head model. Such approaches rely entirely on the landmark detection accuracy, an ad-hoc alignment step, and the extraneous head model. To address this drawback, we present an end-to-end deep network, which explores rotation axis (yaw, pitch and roll) focused innovative attention mechanism to capture the subtle changes in images. The mechanism uses attentional spatial pooling from a self-attention layer and learns the importance over fine-grained to coarse spatial structures and combine them to capture rich semantic information concerning a given rotation axis. The evaluation of our approach using three benchmark datasets is very competitive to state-of-the-arts, including with and without landmark-based methods

## Dependencies
### Python Modules
The code was written in python 3.6.5 and run on Ubuntu 18.04.4. All requirements can be installed by running the following command:
```
pip install -r requirements.txt
```
* Keras 2.2.4
* TensorFlow 1.13.1
* OpenCV 4.2.0
* SelfAttention 0.46.0
* scikit-learn

### Datasets
3 datasets were used:
* 300W-LP & AFLW2000 - http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm
* BIWI - https://www.kaggle.com/kmader/biwi-kinect-head-pose-database

Bounding box information for all datasets can be found at: https://github.com/MingzhenShao/HeadPose

## Running the Code
The model can be created by running:
```
python train_rafa-net.py
```
By default the model will train on 300W-LP and test on AFLW2000 (Line number 350-354 in train_rafa-net.py).

## Citation
```
Behera, A., Wharton, Z., Hewage, P., Kumar, S., 2020. Rotation Axis Focused Attention Network (RAFA-Net) for Estimating Head Pose. In: Asian Confernce on Computer Vision 2020, 30 Nov-4 Dec 2020.
```
