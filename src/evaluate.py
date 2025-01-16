from jiwer import wer, cer

def evaluate_ocr(ground_truth_path, recognized_text_path):
    """
    Evaluates OCR performance using CER and WER.

    Args:
    ground_truth_path (str): Path to the file containing ground truth text.
    recognized_text_path (str): Path to the file containing recognized text.
    
    Returns:
    None

    Character Error Rate (CER):

Measures the ratio of errors (insertions, deletions, substitutions) to the total characters.
​
 
Word Error Rate (WER):

Similar to CER but considers words instead of characters.
Precision, Recall, and F1-score:

Useful for binary metrics, e.g., detecting whether text regions were correctly identified.
    """
    # Read the ground truth and recognized text
    with open(ground_truth_path, "r", encoding="utf-8") as gt_file:
        ground_truth = gt_file.read().strip()
    
    with open(recognized_text_path, "r", encoding="utf-8") as rec_file:
        recognized_text = rec_file.read().strip()

    # Calculate CER and WER
    cer_value = cer(ground_truth, recognized_text)
    wer_value = wer(ground_truth, recognized_text)

    # Print results
    print(f"Character Error Rate (CER): {cer_value:.2%}")
    print(f"Word Error Rate (WER): {wer_value:.2%}")

# Example usage
ground_truth_file = r"C:\Users\vyshn\OneDrive\Documents\Catenate Corp- AI Engineer Assignment\HandWrittenRecognition\Data\ground_truth.txt"
recognized_text_file = r"C:\Users\vyshn\OneDrive\Documents\Catenate Corp- AI Engineer Assignment\HandWrittenRecognition\results\recognized_text.txt"

evaluate_ocr(ground_truth_file, recognized_text_file)
