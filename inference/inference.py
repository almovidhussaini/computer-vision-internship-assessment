
import cv2
import torch
import numpy as np
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class NotebookInference:

    def __init__(
        self,
        model_path="models/maskrcnn_notebook.pth",
        calibration_path="calibration/camera_params.npz"
    ):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # -----------------------------
        # Load camera calibration
        # -----------------------------

        calib = np.load(calibration_path)

        self.camera_matrix = calib["camera_matrix"]
        self.dist_coeffs = calib["distortion_coefficients"]

        # -----------------------------
        # Create Mask R-CNN
        # -----------------------------

        self.model = maskrcnn_resnet50_fpn(
            weights=None
        )

        num_classes = 2

        # Box predictor
        in_features = (
            self.model
            .roi_heads
            .box_predictor
            .cls_score
            .in_features
        )

        self.model.roi_heads.box_predictor = (
            FastRCNNPredictor(
                in_features,
                num_classes
            )
        )

        # Mask predictor
        in_features_mask = (
            self.model
            .roi_heads
            .mask_predictor
            .conv5_mask
            .in_channels
        )

        hidden_layer = 256

        self.model.roi_heads.mask_predictor = (
            MaskRCNNPredictor(
                in_features_mask,
                hidden_layer,
                num_classes
            )
        )

        # -----------------------------
        # Load trained weights
        # -----------------------------

        checkpoint = torch.load(
            model_path,
            map_location=self.device
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)
        self.model.eval()

    # ==================================================
    # Undistort image
    # ==================================================

    def undistort(self, image):

        h, w = image.shape[:2]

        new_camera_matrix, roi = (
            cv2.getOptimalNewCameraMatrix(
                self.camera_matrix,
                self.dist_coeffs,
                (w, h),
                0,
                (w, h)
            )
        )

        undistorted = cv2.undistort(
            image,
            self.camera_matrix,
            self.dist_coeffs,
            None,
            new_camera_matrix
        )

        x, y, crop_w, crop_h = roi

        if crop_w > 0 and crop_h > 0:

            undistorted = undistorted[
                y:y + crop_h,
                x:x + crop_w
            ]

        return undistorted

    # ==================================================
    # Run inference
    # ==================================================

    def predict(
        self,
        image_path,
        score_threshold=0.5
    ):

        # -----------------------------
        # Load image
        # -----------------------------

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        # -----------------------------
        # Undistort
        # -----------------------------

        undistorted = self.undistort(image)

        # -----------------------------
        # Convert BGR → RGB
        # -----------------------------

        image_rgb = cv2.cvtColor(
            undistorted,
            cv2.COLOR_BGR2RGB
        )

        # -----------------------------
        # Convert to tensor
        # -----------------------------

        image_tensor = to_tensor(
            image_rgb
        ).to(self.device)

        # -----------------------------
        # Model inference
        # -----------------------------

        with torch.no_grad():

            prediction = self.model(
                [image_tensor]
            )[0]

        # -----------------------------
        # Select best detection
        # -----------------------------

        scores = prediction["scores"]

        valid_indices = torch.where(
            scores >= score_threshold
        )[0]

        if len(valid_indices) == 0:

            return {
                "image": image_rgb,
                "annotated_image": image_rgb,
                "mask": None,
                "box": None,
                "score": None
            }

        # Highest confidence detection
        best_idx = valid_indices[
            torch.argmax(
                scores[valid_indices]
            )
        ].item()

        score = (
            prediction["scores"]
            [best_idx]
            .item()
        )

        mask = (
            prediction["masks"]
            [best_idx, 0]
            .cpu()
            .numpy()
        )

        mask = mask > 0.5

        box = (
            prediction["boxes"]
            [best_idx]
            .cpu()
            .numpy()
            .astype(int)
        )

        # -----------------------------
        # Create annotation
        # -----------------------------

        annotated = image_rgb.copy()

        # Mask overlay
        overlay = np.zeros_like(
            annotated
        )

        overlay[:, :, 1] = (
            mask.astype(np.uint8) * 255
        )

        annotated = cv2.addWeighted(
            annotated,
            1.0,
            overlay,
            0.35,
            0
        )

        # Bounding box
        x1, y1, x2, y2 = box

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            3
        )

        # Confidence label
        label = (
            f"Notebook: {score:.2f}"
        )

        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        return {
            "image": image_rgb,
            "annotated_image": annotated,
            "mask": mask,
            "box": box,
            "score": score
        }
