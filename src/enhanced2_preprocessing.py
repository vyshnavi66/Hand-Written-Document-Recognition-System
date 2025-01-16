import cv2
import numpy as np
from PIL import Image

def enhanced2_preprocessing(input_path, output_path):
    """
    Advanced preprocessing for handwriting recognition.
    """
    # Load the grayscale image
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Enhance contrast using histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)

    # Deskew the image
    coords = np.column_stack(np.where(enhanced > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = enhanced.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(enhanced, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Resize to the model's expected size
    resized = cv2.resize(deskewed, (320, 320), interpolation=cv2.INTER_AREA)

    # Apply adaptive thresholding
    thresholded = cv2.adaptiveThreshold(
        resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Save the preprocessed image
    cv2.imwrite(output_path, thresholded)
    print(f"Preprocessed image saved to {output_path}")

# Example usage
input_image_path = r"C:\Users\vyshn\OneDrive\Documents\Catenate Corp- AI Engineer Assignment\HandWrittenRecognition\Data\0003_137003.tif"
output_image_path = r"C:\Users\vyshn\OneDrive\Documents\Catenate Corp- AI Engineer Assignment\HandWrittenRecognition\Results\enhanced2_preprocessed_image.jpg"
enhanced2_preprocessing(input_image_path, output_image_path)
