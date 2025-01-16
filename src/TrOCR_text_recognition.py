from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# Load model and processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Load and preprocess image
#image = Image.open("C:\Users\vyshn\OneDrive\Documents\Catenate Corp- AI Engineer Assignment\HandWrittenRecognition\results\enhanced_preprocessed_image.jpg").convert("RGB")
image = Image.open("C:/Users/vyshn/OneDrive/Documents/Catenate Corp- AI Engineer Assignment/HandWrittenRecognition/results/enhanced_preprocessed_image.jpg").convert("RGB")

# Perform OCR
pixel_values = processor(images=image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(f"Enhanced Recognized text: {generated_text}")
