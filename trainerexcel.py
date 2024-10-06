import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Function to load data from Excel file
def load_data(file_name, sheet_name):
    df = pd.read_excel(file_name, sheet_name=sheet_name)
    return df

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

# Function to preprocess data for prediction
def preprocess_prediction_data(df):
    df = df.dropna(subset=['Description'])
    descriptions = df['Description'].astype(str).tolist()  # Convert to strings
    return descriptions

# Function to tokenize dataset
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
    
    mode = input("Enter mode (train/predict): ").strip().lower()

    if mode == "train":
        df = load_data("MAIN.xlsx", "TrainingData")
        df, label_encoder, label_mapping = preprocess_training_data(df)
        dataset = Dataset.from_pandas(df)
        train_test_split = dataset.train_test_split(test_size=0.2)
        train_dataset = train_test_split['train']
        test_dataset = train_test_split['test']
        tokenizer = DistilBertTokenizer.from_pretrained('./fine-tuned-distilbert')
        train_dataset = tokenize_dataset(train_dataset, tokenizer)
        test_dataset = tokenize_dataset(test_dataset, tokenizer)
        model = DistilBertForSequenceClassification.from_pretrained('./fine-tuned-distilbert', num_labels=len(label_mapping))
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
        print(f"Using device: {device}")
        trainer.train()
        results = trainer.evaluate()
        print("Evaluation Results:", results)
        model.save_pretrained("./fine-tuned-distilbert")
        tokenizer.save_pretrained("./fine-tuned-distilbert")

    elif mode == "predict":
        df = load_data("MAIN.xlsx", "Predict")
        descriptions = preprocess_prediction_data(df)
        tokenizer = DistilBertTokenizer.from_pretrained('./fine-tuned-distilbert')
        model = DistilBertForSequenceClassification.from_pretrained('./fine-tuned-distilbert')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        inputs = tokenizer(descriptions, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        predicted_classes = torch.argmax(logits, dim=1)

        print("Predictions shape:", predicted_classes.shape)
        print("Predictions values:", predicted_classes[:5])  # print the first 5 predicted labels

        # Move the predictions tensor to the host memory

        predictions = predicted_classes.cpu().numpy()

        # Update the predictions array to match the length of the DataFrame
        predictions = predictions[:len(df)]

        # Write predictions back to the Google Sheet
        df = df.iloc[:-1]
        df['predictions'] = predictions
        # Write the updated DataFrame back to the Excel file
        with pd.ExcelWriter('MAIN.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name='Predict', index=False)
    else:
        print("Invalid mode. Use 'train' or 'predict'.")

if __name__ == "__main__":
    main()