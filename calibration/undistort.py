
import cv2
import numpy as np
import glob
import os

# ==============================
# PATHS
# ==============================

PROJECT_ROOT = "/content/drive/MyDrive/project-root"

PARAMS_PATH = os.path.join(
    PROJECT_ROOT,
    "calibration",
    "camera_params.npz"
)

INPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "raw"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "undistorted"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================
# LOAD CAMERA PARAMETERS
# ==============================

data = np.load(PARAMS_PATH)

camera_matrix = data["camera_matrix"]
distortion_coefficients = data["distortion_coefficients"]

print("Camera parameters loaded.")
print("Camera matrix:")
print(camera_matrix)

print("\nDistortion coefficients:")
print(distortion_coefficients)


# ==============================
# FIND IMAGES
# ==============================

image_paths = []

for extension in [
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.JPG",
    "*.JPEG",
    "*.PNG"
]:
    image_paths.extend(
        glob.glob(
            os.path.join(INPUT_DIR, extension)
        )
    )

print("\nImages found:", len(image_paths))


# ==============================
# UNDISTORT IMAGES
# ==============================

successful = 0

for image_path in image_paths:

    image = cv2.imread(image_path)

    if image is None:
        print("[FAILED TO READ]", image_path)
        continue

    undistorted = cv2.undistort(
        image,
        camera_matrix,
        distortion_coefficients
    )

    filename = os.path.basename(image_path)

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    cv2.imwrite(
        output_path,
        undistorted
    )

    successful += 1

    print("[OK]", filename)


# ==============================
# SUMMARY
# ==============================

print("\n==============================")
print("UNDISTORTION COMPLETE")
print("==============================")

print("Input images:", len(image_paths))
print("Successful:", successful)
print("Output directory:")
print(OUTPUT_DIR)
