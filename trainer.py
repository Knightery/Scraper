import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import os

# Set up the Google Sheets API client
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("googleauth.json", scope)
client = gspread.authorize(creds)

# Function to load data from Google Sheets
def load_data(sheet_name, tab_name):
    sheet = client.open(sheet_name).worksheet(tab_name)
    data = sheet.get_all_records()
    return pd.DataFrame(data), sheet

# Function to preprocess data for training
def preprocess_training_data(df):
    df = df.dropna(subset=['Description', 'Category'])
    descriptions = df['Description'].tolist()
    tags = df['Category'].tolist()
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(tags)
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    df = pd.DataFrame({'text': descriptions, 'label': labels})
    return df, label_encoder, label_mapping

# Load the DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_dataset(dataset, tokenizer):
    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=512)
    return dataset.map(tokenize, batched=True, batch_size=64)

# Function to compute metrics
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Main function
def main():
    df, _ = load_data("MAIN", "TrainingData")
    df, label_encoder, label_mapping = preprocess_training_data(df)
    dataset = Dataset.from_pandas(df)
    train_test_split = dataset.train_test_split(test_size=0.2)
    train_dataset = train_test_split['train']
    test_dataset = train_test_split['test']
    
    tokenizer_path = './fine-tuned-distilbert/tokenizer'
    model_path = './fine-tuned-distilbert/model'

    # Check if tokenizer exists
    if os.path.exists(tokenizer_path):
        tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        tokenizer.save_pretrained(tokenizer_path)  # Save the tokenizer

    # Tokenize the datasets
    train_dataset = tokenize_dataset(train_dataset, tokenizer)
    test_dataset = tokenize_dataset(test_dataset, tokenizer)

    # Check if model directory exists
    if os.path.exists(model_path):
        model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=len(label_mapping))
    else:
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(label_mapping))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    training_args = TrainingArguments(
        output_dir='./results',
        eval_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        weight_decay=0.01,
        save_total_limit=2,
        fp16=True
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()
    results = trainer.evaluate()
    print("Evaluation Results:", results)
    model.save_pretrained("./fine-tuned-distilbert")
    tokenizer.save_pretrained("./fine-tuned-distilbert")

if __name__ == "__main__":
    main()
