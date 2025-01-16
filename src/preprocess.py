import cv2
import numpy as np
import os

def preprocess_image(input_path, output_path):


    """
    Preprocesses an image for text recognition.
    - Converts to grayscale.
    - Denoises the image using Gaussian blur.
    - Applies adaptive thresholding. 
    
    Args:
    input_path (str): Path to the input image.
    output_path (str): Path to save the preprocessed image.

    """


    # Load the image
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image was loaded properly
    if image is None:
        raise ValueError(f"Image not found at {input_path}")

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply adaptive thresholding for contrast enhancement
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Save the preprocessed image
    cv2.imwrite(output_path, binary)
    print(f"Preprocessed image saved to {output_path}")

# Example usage
if __name__ == "__main__":
    #input_image_path = "../data/handwritten_sample.jpg"
    #output_image_path = "../results/preprocessed_sample.jpg"
    input_image_path = r"C:\Users\vyshn\OneDrive\Documents\Catenate Corp- AI Engineer Assignment\HandWrittenRecognition\Data\0003_137003.tif"
    output_image_path = r"C:\Users\vyshn\OneDrive\Documents\Catenate Corp- AI Engineer Assignment\HandWrittenRecognition\Results\preprocessed_image.jpg"
    preprocess_image(input_image_path, output_image_path)
