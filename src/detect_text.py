import cv2
import numpy as np

def detect_text_regions(input_path, output_path):
    # Load the preprocessed image
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Use Canny edge detection
    edges = cv2.Canny(image, 100, 200)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on a copy of the image
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        # Get bounding boxes for each contour
        x, y, w, h = cv2.boundingRect(contour)
        if w > 30 and h > 15:  # Filter small noise
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save the image with detected regions
    cv2.imwrite(output_path, output_image)
    print(f"Detected text regions saved to {output_path}")

# Example usage
input_image_path = r"C:\Users\vyshn\OneDrive\Documents\Catenate Corp- AI Engineer Assignment\HandWrittenRecognition\Data\0003_137003.tif"
output_image_path = r"C:\Users\vyshn\OneDrive\Documents\Catenate Corp- AI Engineer Assignment\HandWrittenRecognition\Results\preprocessed_image.jpg"
detect_text_regions(input_image_path, output_image_path)
