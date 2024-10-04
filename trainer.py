import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load the Excel file
file_path = "C:\\Users\\nyter\\Desktop\\Scraper\\trainingdata.xlsx"  # Ensure correct path
data = pd.read_excel(file_path)

# Drop rows where either the description or tag is missing
data = data.dropna(subset=['Description', 'Tag'])

# Extract descriptions and tags
descriptions = data['Description'].tolist()
tags = data['Tag'].tolist()

# Convert tags to numerical labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(tags)

# Save label mapping
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Mapping: ", label_mapping)

# Create a pandas DataFrame with descriptions and labels
df = pd.DataFrame({'text': descriptions, 'label': labels})

# Convert the DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Split into train and test sets (80% train, 20% test)
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Preview the first example in the training set
print(train_dataset[0])

# Load the DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize the dataset
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

train_dataset = train_dataset.map(tokenize, batched=True, batch_size=64)
test_dataset = test_dataset.map(tokenize, batched=True, batch_size=64)

# Load the pre-trained DistilBERT model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(label_mapping))

# Move the model to the appropriate device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define compute metrics function for evaluation
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=True  # Enable mixed precision training
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics  # Include custom metrics
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print("Evaluation Results:", results)

# Save the model and tokenizer
model.save_pretrained("./fine-tuned-distilbert")
tokenizer.save_pretrained("./fine-tuned-distilbert")

# Tokenize new descriptions for prediction
new_descriptions = [
    "Bros NEVER trying to setup a trap again💀Use code:KQDEE in the item shop❤️#fortnite #fortnitefunny",
    "#martialarts Don't Blink or you will miss it. #artesmarcias #selfdefense",
    "Yoru 200IQ ulti 😳 #valorant #valorantclips",
    "he's a great painter from austria #shrots #darkhumor #germany #ww2",
    "pranking my teacher", "bollywood movie", "avengeres episode 2"
]

inputs = tokenizer(new_descriptions, return_tensors="pt", padding=True, truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to the same device as the model

# Get predictions
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=-1)

# Convert predictions back to tags
predicted_tags = label_encoder.inverse_transform(predictions.cpu().numpy())
print("Predicted Tags: ", predicted_tags)
