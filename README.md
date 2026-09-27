Computer Vision Internship Assessment
Project Overview

This project implements an end-to-end computer vision pipeline for:

Camera calibration
Image undistortion
Notebook instance segmentation
Red reference object detection
Real-world measurement
Annotated output generation

The project uses Mask R-CNN with ResNet-50-FPN for notebook segmentation.

Installation

Clone the repository:

git clone https://github.com/almovidhussaini/computer-vision-internship-assessment.git
cd computer-vision-internship-assessment

Install dependencies:

pip install -r requirements.txt

For Google Colab:

from google.colab import drive
drive.mount("/content/drive")

%cd /content/drive/MyDrive/project-root

A CUDA GPU is recommended for training and inference.

The trained model is not included in GitHub because of its large file size.

Place the model at:

models/maskrcnn_notebook.pth
How to Run

The main workflow is:

Input Image
    ↓
Camera Undistortion
    ↓
Mask R-CNN
    ↓
Notebook Segmentation
    ↓
Red Reference Detection
    ↓
Pixel-to-mm Conversion
    ↓
Dimension Measurement
    ↓
Annotated Output

Standalone inference:

inference/inference.py

End-to-end inference and measurement:

inference/run_pipeline.py
Project Structure
project-root/
├── calibration/
├── dataset/
├── models/
├── inference/
├── measurement/
├── docs/
├── requirements.txt
├── README.md
└── Internship_task.ipynb
Documentation

Detailed information is available in the docs/ folder:

model_card.md — model, dataset, training and evaluation
inference_guide.md — inference and running the pipeline
measurement_methodology.md — calibration and measurement method
measurement_accuracy_report.md — measurement results and error analysis
Limitations
Dataset contains 100 images.
Only one foreground class, notebook, is supported.
Measurement accuracy is affected by perspective and reference-object placement.
The trained model checkpoint is excluded from GitHub because of its large file size.
Conclusion

This project provides a complete computer vision workflow from camera calibration and notebook segmentation to real-world measurement.
