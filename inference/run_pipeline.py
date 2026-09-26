
import os
import cv2

from inference.inference import NotebookInference
from measurement.measure import NotebookMeasurement


def run_pipeline(
    image_path,
    output_path="inference/output.jpg",
    score_threshold=0.5
):

    # -----------------------------------------
    # Load inference model
    # -----------------------------------------

    detector = NotebookInference(
        model_path="models/maskrcnn_notebook.pth",
        calibration_path="calibration/camera_params.npz"
    )

    # -----------------------------------------
    # Run detection
    # -----------------------------------------

    result = detector.predict(
        image_path,
        score_threshold=score_threshold
    )

    if result["mask"] is None:
        print("No notebook detected.")

        return None

    # -----------------------------------------
    # Measurement
    # -----------------------------------------

    measurement = NotebookMeasurement(
        real_width_cm=18.0,
        real_height_cm=30.0
    )

    measurement_result = (
        measurement.calculate_measurement(
            result["mask"]
        )
    )

    # -----------------------------------------
    # Draw measurement
    # -----------------------------------------

    final_image = measurement.draw_measurement(
        result["annotated_image"],
        measurement_result
    )

    # -----------------------------------------
    # Save output
    # -----------------------------------------

    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True
        )

    # RGB → BGR for OpenCV
    final_bgr = cv2.cvtColor(
        final_image,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        output_path,
        final_bgr
    )

    # -----------------------------------------
    # Print results
    # -----------------------------------------

    print("\n================================")
    print("NOTEBOOK INFERENCE RESULT")
    print("================================")

    print(
        f"Confidence: "
        f"{result['score']:.3f}"
    )

    print(
        f"Pixel width: "
        f"{measurement_result['pixel_width']:.2f} px"
    )

    print(
        f"Pixel height: "
        f"{measurement_result['pixel_height']:.2f} px"
    )

    print(
        f"Width: "
        f"{measurement_result['width_cm']:.2f} cm"
    )

    print(
        f"Height: "
        f"{measurement_result['height_cm']:.2f} cm"
    )

    print(
        f"Rotation: "
        f"{measurement_result['angle']:.2f} degrees"
    )

    print(
        f"\nOutput saved to: "
        f"{output_path}"
    )

    return {
        "result": result,
        "measurement": measurement_result,
        "output_path": output_path
    }
