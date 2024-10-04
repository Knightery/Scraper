from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

# Load the tokenizer and model
model_path = "C:\\Users\\nyter\\Desktop\\Scraper\\fine-tuned-distilbert"  # Update with the actual path
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Set the model to evaluation mode
model.eval()

texts = ["best bollywood dance 10/10 comiplation"]
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
predicted_classes = torch.argmax(logits, dim=1)

print(f"Predicted classes: {predicted_classes}")
