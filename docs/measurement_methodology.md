
#Measurement methodolgy
## overview
the measurement pipeline estimates the pyhsical dimensions of a segment notebook from an image
the complete process is
--> camera calibration
--> image undistortion
--> notebook segmentation
--> independed reference object detection
--> pixel to milimeter scal estimation
--> notebook dimension calcultaion
--> accurasy evalutaion against pyhsical ground truth

## camera calibration was performed using printed cheakerboard(10x8 squares,20mm x 20mm,20 images)
opencv clibration was used to estimated the camera intrinsic matrix and distortion coefficient
the result is put in calibration/camera_params.npz
## image undistortion
before measuremnt each input imae is undistored

The measurement pipeline is real image--> camera calibrated parameter --> undistored image --> segmentation and measurement
##Notebook segmentation
the notebook is detected using mask r-cnn model and trained with manually annoteted notebook dataset
the model produces bounding box --> segmentation mask --> confidence score
the segmentation mask used to determine notebook visible boundry
a rotated minimun-area rectangle is then fitted to the contour using cv2.minAreaRect()
this provides the nootbook pixel width and pixel height

##Independent reference object
the pythici dimension of the refrence object is 100 mm x 60 mm and colored red. The largest red contour is selected as reference object

## pixel to millimeter conversion
if the detected long side is L_px and known pyshical long side is 100 mm
 pixels_per_mm_long = L_px / 100, pixels_per_mm_long = s_px/60
pixels_per_mm - (pixel_per_mm_long + pixel_per_mm_short)/2

 the notebook dimensions are then calculated 
width_mm = width_pixels/pixels_per_mm
height_mm = height_pixels/pixels_per_mm

##Ground thruth
the pyshical notebook has originial dimensions l = 300mm,w = 1180mm
## accuracy evaulation
each image contains notebook and reference red object
the following metrices were calculated
1) mean absolute error(MAE) ==> mean(predicted-groug_truth)
2) mean pecentage error(MPE) ==> (predicted-groudth_truth)/groudth_thruth * 100
3) Root Mean squeare error (RMSE) ==> sqrt(mean(predicted-groudth_thruth)^2))
 Final MEasurement results
Length MAE == 52.01 mm
width mae == 20.58 mm
length mpe == 17.34%
width mpe == 11.44%
Length RMSE == 63.117 mm
width rmse == 28.82 mm

## Perspective limitaion
current implementation uses a single scalar pixels - per millimeter value for each image(this is approximation)
when the notebook and refence objeact are viewd from different positions or depth perspective projection can cause different
parts of the image to have effective scales
the limitaionc an cause measruement errors even when the notebook is segmented correctly. reference object is detected
correctly, camera has been calibrated
##limitation the notrebook and reference object may be at different depths or positions relative to the camers
the evaluation contains 10 images of the same pysical notebook rather than 10 different pysical noteboomk instances
#Possible future improvements
==>using a larger planar calibration target
==>placing the ferenece object in the smae plane as the notebook
==>using multiple reference points
