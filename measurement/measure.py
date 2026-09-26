
import cv2
import numpy as np


class NotebookMeasurement:

    def __init__(
        self,
        real_width_cm=18.0,
        real_height_cm=30.0
    ):
        """
        Known physical dimensions of the notebook.
        """

        self.real_width_cm = real_width_cm
        self.real_height_cm = real_height_cm

    # ==================================================
    # Find notebook contour
    # ==================================================

    def get_contour(self, mask):

        mask_uint8 = (
            mask.astype(np.uint8) * 255
        )

        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            raise ValueError(
                "No notebook contour found."
            )

        # Select largest contour
        contour = max(
            contours,
            key=cv2.contourArea
        )

        return contour

    # ==================================================
    # Calculate pixel dimensions
    # ==================================================

    def calculate_pixel_dimensions(self, mask):

        contour = self.get_contour(mask)

        # Rotated bounding rectangle
        rect = cv2.minAreaRect(contour)

        (center_x, center_y), \
        (rect_width, rect_height), \
        angle = rect

        pixel_width = min(
            rect_width,
            rect_height
        )

        pixel_height = max(
            rect_width,
            rect_height
        )

        return {
            "pixel_width": pixel_width,
            "pixel_height": pixel_height,
            "angle": angle,
            "center": (center_x, center_y),
            "rect": rect
        }

    # ==================================================
    # Estimate dimensions using known scale
    # ==================================================

    def calculate_measurement(self, mask):

        dimensions = (
            self.calculate_pixel_dimensions(mask)
        )

        pixel_width = dimensions[
            "pixel_width"
        ]

        pixel_height = dimensions[
            "pixel_height"
        ]

        # Pixel-to-cm scale
        cm_per_pixel_width = (
            self.real_width_cm /
            pixel_width
        )

        cm_per_pixel_height = (
            self.real_height_cm /
            pixel_height
        )

        return {
            **dimensions,

            "width_cm": self.real_width_cm,
            "height_cm": self.real_height_cm,

            "cm_per_pixel_width":
                cm_per_pixel_width,

            "cm_per_pixel_height":
                cm_per_pixel_height
        }

    # ==================================================
    # Draw measurement result
    # ==================================================

    def draw_measurement(
        self,
        image,
        measurement
    ):

        output = image.copy()

        rect = measurement["rect"]

        # Four corners
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        # Draw notebook boundary
        cv2.drawContours(
            output,
            [box],
            0,
            (255, 0, 0),
            4
        )

        # Measurement text
        width_text = (
            f"Width: "
            f"{measurement['width_cm']:.1f} cm"
        )

        height_text = (
            f"Height: "
            f"{measurement['height_cm']:.1f} cm"
        )

        cv2.putText(
            output,
            width_text,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            3
        )

        cv2.putText(
            output,
            height_text,
            (30, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            3
        )

        return output
