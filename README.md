
# Computer Vision Internship Assessment

## Project Overview

This project is implemented an end-to-end computer vision pipeline for segmantion, camera callibration, undistortion, and real-word
measurment
The project gets an input notebook image and perform 
1. Camera callibration
2. Image undistortion
3.Notebook instance segmentation
4. mask extraction
5 real-world dimension estimation
6 annotated output generation

##model (Mask R-cnn with ResNet-50-FPN)
this model is selected because the task requires pixel-level segmentation rather than only bounding-boc detection. The model
provides multi-scale feature extraction and good for instance segmentaion

the model was initialized with COCO-pretrained weights and fine-tuned for two classes:
-background
-Notebook
##Dataset
dataset contains:
-100 notebook images
--100 polygon annotations
COCO annotation format
One class: notebook

train| 70 |70%
validation| 20| 20%
Test| 10|10%

##Camera Calibration
A printed checkerboard was used for camera calibration
-10 x 8 squares
square size 20 mm x 20 mm
20 images
the callibration process estimates

Camera intrinsic matrix
lens distortion coefficient
the parameters are store in calibration/camera_params.npz

##Training Configuration
Architecture   --> Mask R-CNN ResNEt-50-FPN
Pretrained weights --> COCO
Epoches --> 10
Batch size -- > 2
LEarning Rate --> 0.005
Optimizer --> SGD
Momentum --> 0.9
Weight decay -- > StepLR
Scheduler step --> 3 epoches
Augmentation --> Random horizontal flip
classes -- > 2
Device cuda gpu

##Training result
epoch1   --> trainloss(0.9411) -->validation loss(0.3752)
epoch10 --> trainloss(0.478) -->validation loss(0.2640)

#Test Evaluation
Evaluation was performed on the held-out 10-image test set
mAP@0.5	 -->  1.0000
mAP@0.5:0.95 -->	0.8566
Mean Mask IoU -->	0.9266
Precision  -->	0.7143
Recall -->	1.0000
F1 Score  -->	0.8333

#Inference
standalone inference module is located at inference/inference.py
perform 
Input image--> camera unditortion --> Mask R-CNN --> Nitebook Detection --> Segmentation Mask --> bounding Box --> Annotated Ouput

# END-TO-END pipeline
is available in inference/run_pipeline.py  combine inference and measurement
#Real-World Measurement
is located in measurement/measure.py. The notebook dimension used as physical reference are width: 18 cm, Height: 30 cm

The segmentation mask is converted into a contour and a rotated minimum-area rectangle. Pixel dimensions are then converted to centimetres using the known physical notebook dimensions.
#measurment limitaion:
uses known notebook dimension therefore the 18 x 30 dimensions demonstrate the measurement piplene but should not be interpreted as an independent of an unknown objects's pyshical dimensions
For independent measurement, a known reference object or calibrated planar homography setup would be required.

# requirements
Install dependences with pip install -r requirements.txt

Limitaions
The data set contain only 100 images
the held-out test set contains only 10 images
only one object class is currently supported
measurement pipeline uses known notebook dimensions as the pyysical scale
the trained model checkpoint is larger than github normal 100 mb file limit

ConclusionA

The project provides a complete computer vision workflow covering camera calibration, image preprocessing, instance segmentation, evaluation, inference, and real-world measurement.

The implementation uses Mask R-CNN rather than YOLO or Roboflow models and includes a
 held-out test evaluation, standalone inference module, and end-to-end measurement pipeline
