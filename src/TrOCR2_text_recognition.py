from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Trainer, TrainingArguments
#from datasets import load_dataset
from PIL import Image
# Load your dataset
#dataset = load_dataset('your_dataset_script.py')
image = Image.open("C:/Users/vyshn/OneDrive/Documents/Catenate Corp- AI Engineer Assignment/HandWrittenRecognition/results/enhanced2_preprocessed_image.jpg").convert("RGB")

# Load the model and processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    evaluation_strategy="epoch",
    save_steps=500,
    save_total_limit=2,
    num_train_epochs=3,
    fp16=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=processor,
)

# Train the model
trainer.train()
model.save_pretrained("./finetuned_trocr")
