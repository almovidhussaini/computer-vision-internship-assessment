import cv2
import numpy as np
import glob
import os


# ============================================================
# CONFIGURATION
# ============================================================
print("hello3")

IMAGE_DIR = "/content/drive/MyDrive/project-root/calibration/images"

# Number of INNER corners
# Example: checkerboard has 9 x 7 inner corners
CHECKERBOARD = (9, 7)

# Physical size of one checkerboard square
# Change this according to your printed checkerboard.
# Example: 20 mm
SQUARE_SIZE = 20.0


# ============================================================
# PREPARE 3D OBJECT POINTS
# ============================================================

objp = np.zeros(
    (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
    np.float32
)

objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)

# Convert from checkerboard units to millimeters
objp *= SQUARE_SIZE


# ============================================================
# STORAGE
# ============================================================

object_points = []
image_paths = []
image_points = []

for extension in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
    image_paths.extend(
        glob.glob(os.path.join(IMAGE_DIR, extension))
    )

print("Images found:", len(image_paths))


# ============================================================
# FIND CHECKERBOARD CORNERS
# ============================================================

image_size = None

successful_images = 0

for image_path in image_paths:

    image = cv2.imread(image_path)

    if image is None:
        print("Could not read:", image_path)
        continue

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    image_size = gray.shape[::-1]

    # Detect checkerboard corners
    ret, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        None
    )

    if ret:

        # Improve corner accuracy
        criteria = (
            cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )

        corners_refined = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        object_points.append(objp)
        image_points.append(corners_refined)

        successful_images += 1

        print(
            f"[OK] {os.path.basename(image_path)}"
        )

    else:

        print(
            f"[FAILED] {os.path.basename(image_path)}"
        )


# ============================================================
# CHECK WHETHER ENOUGH IMAGES WERE FOUND
# ============================================================

print()
print("Successful detections:", successful_images)

if successful_images < 10:
    raise RuntimeError(
        "Too few successful checkerboard detections."
    )


# ============================================================
# CAMERA CALIBRATION
# ============================================================

ret, camera_matrix, distortion_coefficients, rvecs, tvecs = \
    cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None
    )


# ============================================================
# REPROJECTION ERROR
# ============================================================

# REPROJECTION ERROR
total_error = 0

for i in range(len(object_points)):

    projected_points, _ = cv2.projectPoints(
        object_points[i],
        rvecs[i],
        tvecs[i],
        camera_matrix,
        distortion_coefficients
    )

    img_points = image_points[i].reshape(-1, 2)
    proj_points = projected_points.reshape(-1, 2)

    error = np.linalg.norm(
        img_points - proj_points
    ) / len(img_points)

    total_error += error

mean_error = total_error / len(object_points)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n==============================")
print("CAMERA CALIBRATION RESULTS")
print("==============================")

print("\nCamera Matrix:")
print(camera_matrix)

print("\nDistortion Coefficients:")
print(distortion_coefficients)

print("\nReprojection Error:")
print(mean_error)


# ============================================================
# SAVE PARAMETERS
# ============================================================

np.savez( "/content/drive/MyDrive/project-root/calibration/camera_params.npz", camera_matrix=camera_matrix, distortion_coefficients=distortion_coefficients, reprojection_error=mean_error )

print("\nCamera parameters saved to:")
print("camera_params.npz")