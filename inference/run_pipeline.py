import cv2

from inference.inference import NotebookInference
from measurement.measure import NotebookMeasurement


def run_pipeline(
    image_path,
    output_path="inference/output.jpg",
    score_threshold=0.5,
    reference_long_mm=100.0,
    reference_short_mm=60.0
):
    """
    End-to-end notebook measurement pipeline.

    Steps:
        1. Load image
        2. Undistort image
        3. Detect notebook
        4. Detect independent red reference
        5. Calculate pixels/mm
        6. Convert notebook dimensions to mm
        7. Save annotated result
    """

    # ---------------------------------------------------------
    # 1. Load detector
    # ---------------------------------------------------------
    detector = NotebookInference(
        model_path="models/maskrcnn_notebook.pth"
    )

    # ---------------------------------------------------------
    # 2. Measurement module
    # ---------------------------------------------------------
    measurement = NotebookMeasurement()

    # ---------------------------------------------------------
    # 3. Read image
    # ---------------------------------------------------------
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    # ---------------------------------------------------------
    # 4. Undistort image
    # ---------------------------------------------------------
    undistorted = detector.undistort(image)

    # Save temporary undistorted image
    temp_undistorted = "inference/temp_undistorted.jpg"

    cv2.imwrite(
        temp_undistorted,
        undistorted
    )

    # ---------------------------------------------------------
    # 5. Run existing inference method
    # ---------------------------------------------------------
    prediction = detector.predict(
        temp_undistorted,
        score_threshold=score_threshold
    )

    if prediction is None:
        raise RuntimeError(
            "No notebook detected."
        )

    # ---------------------------------------------------------
    # 6. Get prediction values
    # ---------------------------------------------------------
    mask = prediction["mask"]
    confidence = prediction["score"]

    # ---------------------------------------------------------
    # 7. Independent reference measurement
    # ---------------------------------------------------------
    result = measurement.measure_from_mask_with_reference(
        mask=mask,
        image=undistorted,
        reference_long_mm=reference_long_mm,
        reference_short_mm=reference_short_mm
    )

    # ---------------------------------------------------------
    # 8. Draw measurement
    # ---------------------------------------------------------
    annotated = measurement.draw_measurement(
        image=undistorted,
        result=result,
        confidence=confidence
    )

    # ---------------------------------------------------------
    # 9. Draw notebook segmentation contour
    # ---------------------------------------------------------
    mask_uint8 = (
        (mask > 0).astype("uint8") * 255
    )

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        annotated,
        contours,
        -1,
        (0, 255, 0),
        2
    )

    # ---------------------------------------------------------
    # 10. Save final output
    # ---------------------------------------------------------
    cv2.imwrite(
        output_path,
        annotated
    )

    # ---------------------------------------------------------
    # 11. Print results
    # ---------------------------------------------------------
    print("=" * 55)
    print("END-TO-END MEASUREMENT RESULT")
    print("=" * 55)

    print(f"Confidence: {confidence:.3f}")

    print(
        f"Reference size: "
        f"{reference_long_mm:.0f} × "
        f"{reference_short_mm:.0f} mm"
    )

    print(
        f"Reference long side: "
        f"{result['reference_long_px']:.2f} px"
    )

    print(
        f"Reference short side: "
        f"{result['reference_short_px']:.2f} px"
    )

    print(
        f"Pixels/mm: "
        f"{result['pixels_per_mm']:.4f}"
    )

    print(
        f"Notebook pixel width: "
        f"{result['pixel_width']:.2f} px"
    )

    print(
        f"Notebook pixel height: "
        f"{result['pixel_height']:.2f} px"
    )

    print(
        f"Notebook width: "
        f"{result['width_mm']:.2f} mm"
    )

    print(
        f"Notebook height: "
        f"{result['height_mm']:.2f} mm"
    )

    print(
        f"Rotation: "
        f"{result['angle']:.2f} degrees"
    )

    print(
        f"Output saved to: "
        f"{output_path}"
    )

    print("=" * 55)

    # ---------------------------------------------------------
    # 12. Return results
    # ---------------------------------------------------------
    return {
        "confidence": float(confidence),

        "reference_long_px":
            result["reference_long_px"],

        "reference_short_px":
            result["reference_short_px"],

        "pixels_per_mm":
            result["pixels_per_mm"],

        "pixel_width":
            result["pixel_width"],

        "pixel_height":
            result["pixel_height"],

        "width_mm":
            result["width_mm"],

        "height_mm":
            result["height_mm"],

        "angle":
            result["angle"],

        "output_path":
            output_path,

        "mask":
            mask,

        "undistorted_image":
            undistorted,

        "annotated_image":
            annotated
    }