import cv2
import easyocr

def recognize_text(input_path, output_path):
    """
    Recognizes handwritten text from an image using EasyOCR.

    Args:
    input_path (str): Path to the input image.
    output_path (str): Path to save the recognized text.
    """
    # Load the image
    image = cv2.imread(input_path)

    # Initialize the EasyOCR reader
    reader = easyocr.Reader(['en'])

    # Perform text recognition
    results = reader.readtext(image, detail=1)  # detail=1 provides bounding boxes, text, and confidence

    # Process and print the results
    with open(output_path, "w") as f:
        for (bbox, text, prob) in results:
            print(f"Detected Text: {text} (Confidence: {prob:.2f})")
            f.write(f"{text}\n")
    print(f"Recognized text saved to {output_path}")

# Example usage
input_image_path = r"C:\Users\vyshn\OneDrive\Documents\Catenate Corp- AI Engineer Assignment\HandWrittenRecognition\Results\preprocessed_image.jpg"      #output img from detect_text.py
output_text_path = r"C:\Users\vyshn\OneDrive\Documents\Catenate Corp- AI Engineer Assignment\HandWrittenRecognition\results\recognized_text.txt"


recognize_text(input_image_path, output_text_path)
