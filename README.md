# RAFA-NET
Code used in the research of estimating head pose orientation from RGB images.
![RAFA-NET Model Overview](https://github.com/ZWharton15/RAFA-NET-1/blob/master/doc/RAFA_model_overview.png?raw=true)


## Requirements
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

### Data
3 datasets were used:
* 300W-LP & AFLW2000 - http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm
* BIWI - https://www.kaggle.com/kmader/biwi-kinect-head-pose-database

Bounding box information for each dataset were taken from:
?

## Running the Code
The model can be created by running:
```
python train_rafa-net.py
```

## Citation
