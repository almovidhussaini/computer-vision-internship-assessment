import cv2
import numpy as np


class NotebookMeasurement:
    """
    Measurement utilities for the notebook.

    Supports:
    1. Known-dimension scaling (legacy/preliminary method)
    2. Independent red reference object scaling
    """

    def __init__(self, real_width_cm=18.0, real_height_cm=30.0):
        self.real_width_cm = real_width_cm
        self.real_height_cm = real_height_cm

    # ---------------------------------------------------------
    # Existing / legacy measurement method
    # ---------------------------------------------------------
    def measure_from_mask(self, mask):
        """
        Measure notebook dimensions from a segmentation mask
        using the known notebook dimensions.

        Returns:
            dict containing pixel dimensions, physical dimensions,
            angle and rectangle.
        """

        mask_uint8 = (mask > 0).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            raise ValueError("No contour found in segmentation mask.")

        contour = max(contours, key=cv2.contourArea)

        rect = cv2.minAreaRect(contour)

        (_, _), (w_px, h_px), angle = rect

        pixel_width = min(w_px, h_px)
        pixel_height = max(w_px, h_px)

        cm_per_pixel_width = self.real_width_cm / pixel_width
        cm_per_pixel_height = self.real_height_cm / pixel_height

        width_cm = pixel_width * cm_per_pixel_width
        height_cm = pixel_height * cm_per_pixel_height

        return {
            "pixel_width": float(pixel_width),
            "pixel_height": float(pixel_height),
            "width_cm": float(width_cm),
            "height_cm": float(height_cm),
            "angle": float(angle),
            "rect": rect
        }

    # ---------------------------------------------------------
    # Red reference object detection
    # ---------------------------------------------------------
    def detect_red_reference(
        self,
        image,
        min_area=1000
    ):
        """
        Detect the red rectangular reference object.

        Reference object physical dimensions:
            Long side  = 100 mm
            Short side = 60 mm

        Returns:
            dict containing reference pixel dimensions,
            angle, contour and rectangle.
        """

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Red range 1
        lower_red1 = np.array([0, 80, 50])
        upper_red1 = np.array([15, 255, 255])

        # Red range 2
        lower_red2 = np.array([165, 80, 50])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(
            hsv,
            lower_red1,
            upper_red1
        )

        mask2 = cv2.inRange(
            hsv,
            lower_red2,
            upper_red2
        )

        red_mask = cv2.bitwise_or(mask1, mask2)

        # Clean small noise
        kernel = np.ones((7, 7), np.uint8)

        red_mask = cv2.morphologyEx(
            red_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        red_mask = cv2.morphologyEx(
            red_mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        contours, _ = cv2.findContours(
            red_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        contours = [
            contour
            for contour in contours
            if cv2.contourArea(contour) > min_area
        ]

        if not contours:
            raise ValueError(
                "Red reference object was not detected."
            )

        # Largest red object
        red_contour = max(
            contours,
            key=cv2.contourArea
        )

        rect = cv2.minAreaRect(red_contour)

        (_, _), (w_px, h_px), angle = rect

        reference_long_px = max(w_px, h_px)
        reference_short_px = min(w_px, h_px)

        return {
            "mask": red_mask,
            "contour": red_contour,
            "rect": rect,
            "long_px": float(reference_long_px),
            "short_px": float(reference_short_px),
            "angle": float(angle)
        }

    # ---------------------------------------------------------
    # Pixel-to-mm conversion
    # ---------------------------------------------------------
    def calculate_reference_scale(
        self,
        reference_long_px,
        reference_short_px,
        reference_long_mm=100.0,
        reference_short_mm=60.0
    ):
        """
        Calculate pixels-per-mm using both dimensions
        of the physical reference object.

        The two directional scales are averaged.

        Returns:
            dict containing directional scales and
            averaged pixels/mm.
        """

        if reference_long_px <= 0 or reference_short_px <= 0:
            raise ValueError(
                "Reference pixel dimensions must be positive."
            )

        pixels_per_mm_long = (
            reference_long_px /
            reference_long_mm
        )

        pixels_per_mm_short = (
            reference_short_px /
            reference_short_mm
        )

        pixels_per_mm = (
            pixels_per_mm_long +
            pixels_per_mm_short
        ) / 2.0

        return {
            "pixels_per_mm_long": float(pixels_per_mm_long),
            "pixels_per_mm_short": float(pixels_per_mm_short),
            "pixels_per_mm": float(pixels_per_mm)
        }

    # ---------------------------------------------------------
    # Advanced independent measurement
    # ---------------------------------------------------------
    def measure_from_mask_with_reference(
        self,
        mask,
        image,
        reference_long_mm=100.0,
        reference_short_mm=60.0
    ):
        """
        Measure notebook dimensions in millimeters using
        the independently measured red reference object.

        IMPORTANT:
        The notebook's known 300 x 180 mm dimensions are NOT
        used for calculating the measurement.

        Args:
            mask:
                Notebook segmentation mask.

            image:
                Undistorted image containing the red reference.

            reference_long_mm:
                Physical long dimension of reference object.

            reference_short_mm:
                Physical short dimension of reference object.

        Returns:
            Dictionary containing reference measurements,
            notebook measurements and scale information.
        """

        # -----------------------------------------------------
        # 1. Detect red reference
        # -----------------------------------------------------
        reference = self.detect_red_reference(image)

        # -----------------------------------------------------
        # 2. Calculate pixel-to-mm scale
        # -----------------------------------------------------
        scale = self.calculate_reference_scale(
            reference_long_px=reference["long_px"],
            reference_short_px=reference["short_px"],
            reference_long_mm=reference_long_mm,
            reference_short_mm=reference_short_mm
        )

        pixels_per_mm = scale["pixels_per_mm"]

        # -----------------------------------------------------
        # 3. Find notebook contour
        # -----------------------------------------------------
        mask_uint8 = (
            (mask > 0).astype(np.uint8) * 255
        )

        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            raise ValueError(
                "No notebook contour found in segmentation mask."
            )

        notebook_contour = max(
            contours,
            key=cv2.contourArea
        )

        # -----------------------------------------------------
        # 4. Get rotated notebook rectangle
        # -----------------------------------------------------
        notebook_rect = cv2.minAreaRect(
            notebook_contour
        )

        (_, _), (w_px, h_px), notebook_angle = (
            notebook_rect
        )

        notebook_width_px = min(
            w_px,
            h_px
        )

        notebook_height_px = max(
            w_px,
            h_px
        )

        # -----------------------------------------------------
        # 5. Convert pixels → millimeters
        # -----------------------------------------------------
        width_mm = (
            notebook_width_px /
            pixels_per_mm
        )

        height_mm = (
            notebook_height_px /
            pixels_per_mm
        )

        return {
            # Reference object
            "reference_long_px": reference["long_px"],
            "reference_short_px": reference["short_px"],
            "reference_angle": reference["angle"],

            # Scale
            "pixels_per_mm_long": scale[
                "pixels_per_mm_long"
            ],
            "pixels_per_mm_short": scale[
                "pixels_per_mm_short"
            ],
            "pixels_per_mm": pixels_per_mm,

            # Notebook pixels
            "pixel_width": float(
                notebook_width_px
            ),
            "pixel_height": float(
                notebook_height_px
            ),

            # Notebook physical dimensions
            "width_mm": float(width_mm),
            "height_mm": float(height_mm),

            # Useful additional information
            "angle": float(notebook_angle),
            "notebook_contour": notebook_contour,
            "reference_contour": reference["contour"],
            "reference_rect": reference["rect"],
            "notebook_rect": notebook_rect,
            "reference_mask": reference["mask"]
        }

    # ---------------------------------------------------------
    # Draw measurement result
    # ---------------------------------------------------------
    def draw_measurement(
        self,
        image,
        result,
        confidence=None
    ):
        """
        Draw notebook measurement and reference object
        on the image.
        """

        output = image.copy()

        # Notebook rectangle
        notebook_box = cv2.boxPoints(
            result["notebook_rect"]
        )

        notebook_box = np.int32(
            notebook_box
        )

        cv2.drawContours(
            output,
            [notebook_box],
            -1,
            (0, 255, 0),
            3
        )

        # Red reference rectangle
        reference_box = cv2.boxPoints(
            result["reference_rect"]
        )

        reference_box = np.int32(
            reference_box
        )

        cv2.drawContours(
            output,
            [reference_box],
            -1,
            (255, 0, 255),
            3
        )

        # Measurement text
        width_text = (
            f"Width: "
            f"{result['width_mm']:.1f} mm"
        )

        height_text = (
            f"Height: "
            f"{result['height_mm']:.1f} mm"
        )

        scale_text = (
            f"Scale: "
            f"{result['pixels_per_mm']:.3f} px/mm"
        )

        cv2.putText(
            output,
            width_text,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            height_text,
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            scale_text,
            (30, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 255),
            2
        )

        if confidence is not None:

            confidence_text = (
                f"Confidence: "
                f"{confidence:.3f}"
            )

            cv2.putText(
                output,
                confidence_text,
                (30, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

        return output