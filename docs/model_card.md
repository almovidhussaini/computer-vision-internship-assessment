# Notebook Instance segmentation
1. Model overview:
this project uses mask R-CNN with a Resnet-50-fpn for notebook instance segmentation
The model was seleacted because the assessment required an instance segmentationarchitecture and does not allo roboflow models 
or ultralytics yolo
##Model: 
==>architecture: mask r-cnn
==>backbone: resnet-50-fpn
==>Pretrained weights COCO
==> Task: instance segmentaion
==> Number of classes: 2(background and notebook)
==> framework: pytorch/torchVsion
==> input: RGB notebook images
==> output: Notebook bounding box, segmentation mask, and confidence score
2) Intended use
==> Detect a notebook in an image
==> denerate a lixel-level segmentation mask
==> provide a bounding box arounf the notebook
==> provide confidence scor
==> pychical measurement of notebook width and height
3) dataset contains 100 notebook images with manually created polygon annotations
 #dataset information
total images=100, total annotation=100, annotation format = coco, dataset souce= custom images, train split = 70,
valodation split = 20,test = 10
4) camera calibration and processing
images are undistorted using intrinsic camera calibration parameters
calibration was performed using 20 checkerboard images.
5) training configuration
model was initialized with coco-pretrained mask r-cnn wieghts and adapted for notebook segmentaion task
training parameters
achittecture ==> Mask R-CNN Resnet-50-FPN
pretrained weights ==> COCO
Epochs == >10
Batch Size = 2
learning rate == 0.005
Optimizer = 0.9
Schedular step size = 3 epochs
Number of classes = 2

6) TRaining and validation loss
the model was trainned for 10 epochs
epoch1 ) training loss == 0.9411 , validation loss = 0.3752
epoch10) taining loss = -.1478 , validation loss = 0.2640
7) evaluation
was perfomred on the held-out 10-image test set
Segmentation and detection matrics

==> mAP@0.5 = 1.00, mAP@0.5:0.95 = 0.8566, Mean mask liO == 0.9266, recall = 1, precision - 0.7143, f1-score = 0.9333
8) inference behavios 
==> loads the camera callibration parameters, undistorts the input image, loads the trained mask model, perform notebook
instance segmentation, select the highest-confidence notebook detection above the confidenece threshold, produce the
segmentation mask and bounding box, produces an annotated output image.
the main inference pipeline is located at inference/run_pipeline.py
9) physical measurement integration
the segmentation model is also used as part of the pyhsical measurement pipeline
a red object with known dimensions 100mm x 60 mm is places in the measurement image. the system detect the image , get the
pixels and converts to milimeter and applied to notebook mask.
the measure model is located at measurement/measure.py and the measurement evaluation is documented in docs/measurement_accuracy_report.md
10) measurement evaluation:
the experiment used 10 images of the same physical notebook. grouth-thruth dimensions are 300mm x 180mm
the final measuremnt results were 
length mae == 52.01mm, width mae == 20.58mm, length mpe == 17.34%, width mpe == 11.44%
11) confidene vs measeuremnt accuracy:
The segmentation confidence scores on the 10 measurement images were approximately between:

0.979 and 0.993

High segmentation confidence does not necessarily imply high physical measurement accuracy.

Measurement accuracy is also affected by:

perspective distortion
camera-to-object geometry
reference-object placement
differences in orientation
pixel-to-millimeter scale estimation
segmentation boundary accuracy

Therefore, confidence scores should be interpreted separately from physical measurement error.
12) limitaions.
small dataset, limited augmentaion, perspective effects,reference-object assumption,model checkpint size

13) Summary

This project implements a complete notebook segmentation and measurement pipeline using Mask R-CNN with a ResNet-50-FPN backbone.

The model was trained on 100 manually annotated notebook images using a 70/20/10 train-validation-test split. On the held-out test set, the model achieved an mAP@0.5 of 1.0000, mAP@0.5:0.95 of 0.8566, and mean mask IoU of 0.9266.

Camera calibration achieved a mean reprojection error of approximately 0.2362 pixels.

The trained segmentation model is integrated with camera undistortion and a physical reference-object-based measurement system. The 10-image measurement experiment produced a length MAE of 52.01 mm and width MAE of 20.58 mm.

The measurement results demonstrate that segmentation performance and physical measurement accuracy are separate aspects of the overall system, with perspective and scale-estimation effects remaining important limitations.
 
