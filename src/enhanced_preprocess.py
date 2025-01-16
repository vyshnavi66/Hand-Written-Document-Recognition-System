import cv2
import numpy as np

def enhanced_preprocess(input_path, output_path):
    """
    Enhanced preprocessing pipeline for handwritten OCR.
    
    Args:
    input_path (str): Path to the input image.
    output_path (str): Path to save the preprocessed image.
    """
    # Load the image in grayscale
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Apply adaptive thresholding
    thresholded = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Perform morphological operations to clean up
    kernel = np.ones((2, 2), np.uint8)
    morphed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

    # Save the preprocessed image
    cv2.imwrite(output_path, morphed)
    print(f"Enhanced preprocessed image saved to {output_path}")

# Example usage
input_image_path = r"C:\Users\vyshn\OneDrive\Documents\Catenate Corp- AI Engineer Assignment\HandWrittenRecognition\Data\0003_137003.tif"
output_image_path = r"C:\Users\vyshn\OneDrive\Documents\Catenate Corp- AI Engineer Assignment\HandWrittenRecognition\Results\enhanced_preprocessed_image.jpg"
enhanced_preprocess(input_image_path, output_image_path)
