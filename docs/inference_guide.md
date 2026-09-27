
inference guide -- Notebook segmentation and measurement
1) overview
this guide explains how to run the trained notebook segmentation and physical measurement pipeline
input image--> camera undistortion--> mask r-cnn inference--> notebook segmentation mask--> red reference object detection
--> pixel-to-mm scale--> notebook width /height -->annotated output image. the system uses mask r-cnn and a resnet 50 fpn

2) caemra calbiration: the project reuires intrinsic calibration parameters which is stored in calibration/camera_params.npz
3) input image:
for physical measurement the input image should contain 
==> notebook, red reference object, both objects visible clearly.
example measurement images are stored in dataset/measurement/
4) loading the trained model. the model is present in nodels/maskrcc_notebook.pth. inference class is implemented in 
inference/inference.py
6)running basic segmentation inference. the NotebookInference class can be used to detect a notebook in an image
7)image undistortion. The inference pipeline applies camera undistortion before performing pysical measurement
8) end-to-end measurement pipeline. The integrated pipeline is implemented in inference/run_pipeline.py
the piplenine performs loading the image, undistort the image, run mask r-cnn, detect the reference object, calculate pix-to-
mm scale, measure the notebook,generate the annotated output
9) running the complete pipeline
Example:

from inference.run_pipeline import run_pipeline

result = run_pipeline(
    image_path="dataset/measurement/image1.jpeg",
    output_path="inference/output.jpg"
)

print("Confidence:", result["confidence"])
print("Pixels/mm:", result["pixels_per_mm"])
print("Width:", result["width_mm"], "mm")
print("Height:", result["height_mm"], "mm")
print("Output:", result["output_path"])

The exact function arguments should match the current implementation in:

inference/run_pipeline.py
11. Expected Output

A successful pipeline execution returns information similar to:

Confidence: 0.985
Reference long side: 296.48 px
Reference short side: 191.10 px
Pixels/mm: 3.0750
Notebook pixel width: 602.30 px
Notebook pixel height: 1203.69 px
Notebook width: 195.87 mm
Notebook height: 391.45 mm
Rotation: -89.44 degrees

The output image is saved to the specified output path.

For example:

inference/output.jpg

