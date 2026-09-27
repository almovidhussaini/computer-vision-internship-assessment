
Measurement accuracy report
1) objective
evaluates the pyhsical measuremnt performance of the computer vision piples
it detects a notebook using Mask R-CNN and estimates pyshical dimensions using independent red reference object with known
dimensions( used 10 measurment images of the same physical notebook. size:300mm x180mm). it is ground thruth and used to 
calculate the pixels to mm scale
2) measurement pipeline: input image -> camera undistortion-> notebook segmentation --> notebook mask --> red reference detection
--> pix to mm scale estimation --> notebook pix dimension--> pysical dimension in mm --> accuracy evaluation
3) reference object@ a red rectangular object used for scale estimation(100mm x 60mm). pixel per mm is calcuated using the ratio
and notebook width and height in mm are calucated using width pixel and pixel per mm
4) evaluation dataset
the evaulation contains 10 measurement images of same width and height (300mm x 180mm)
5) overall accuracy
mae ==> legth (52.01 mm), width (20.58mm), mpe (17.34%,11.44%), RMSE (63.17mm,26.82mm)
7) segmentaion confidence
 Approx confidence range. min conf = 0.070, max conf = 0.993
8. MEasurement error analysis
the measruement errors vary substantially between images
image4 has relatively small length error of approximately 4.18%
image9 has a width error of approximately 0.83%
9) main sourxce of error
is uses one global pixels-per-millimeter scale for each image.
the following factors can affect the result
 ==> camera viewpoint, distance between the camera and object,relative positin if the notebook and ref object,perspective 
projection, segmentation boundary accuracy.
10) limitaion
same physical object, single reference object, global scal approximation, perspective effects,

11) the measuremtn implementaion is located in measurement/measure.py. The end - to -end pipeline is located in 
inference/run_pipeline.py, the trained segmentation model is stored locally as model/maskrcnn_notebook.pth
