import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
from langdetect import detect, LangDetectException

# Set up the Google Sheets API client
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("googleauth.json", scope)
client = gspread.authorize(creds)

# Function to load data from Google Sheets
def load_data(sheet_name, tab_name):
    sheet = client.open(sheet_name).worksheet(tab_name)
    data = sheet.get_all_records()
    return pd.DataFrame(data), sheet

# Function to preprocess data for prediction
def preprocess_prediction_data(df):
    df = df.dropna(subset=['Description'])
    descriptions = df['Description'].astype(str).tolist()  # Convert to strings
    
    def is_english(text):
        try:
            return detect(text) == 'en'
        except LangDetectException:
            return False
    
    processed_descriptions = []
    for desc in descriptions:
        if not is_english(desc) or len(desc.split()) <= 2:
            processed_descriptions.append("N/A")
        else:
            processed_descriptions.append(desc)
    
    return processed_descriptions

# Function to tokenize dataset
def tokenize_dataset(dataset, tokenizer):
    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=512)
    return dataset.map(tokenize, batched=True, batch_size=64)

def main():
    # Load data and preprocess descriptions
    df, sheet = load_data("MAIN", "Predict")
    descriptions = preprocess_prediction_data(df)
    
    # Load tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('./fine-tuned-distilbert')
    model = DistilBertForSequenceClassification.from_pretrained('./fine-tuned-distilbert')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Filter out "N/A" descriptions before tokenization and prediction
    valid_descriptions = [desc for desc in descriptions if desc != "N/A"]
    valid_indices = [i for i, desc in enumerate(descriptions) if desc != "N/A"]
    
    if valid_descriptions:
        # Tokenize descriptions
        inputs = tokenizer(valid_descriptions, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict categories
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        predicted_classes = torch.argmax(logits, dim=1)
        
        print("Predictions shape:", predicted_classes.shape)
        print("Predictions values:", predicted_classes[:5])  # print the first 5 predicted labels
        
        # Move the predictions tensor to the host memory
        predictions = predicted_classes.cpu().numpy()
        
        # Define the category mapping
        category_mapping = {
            0: "animal",
            1: "arts",
            2: "beauty",
            3: "car",
            4: "content",
            5: "cooking",
            6: "gaming",
            7: "gym",
            8: "home",
            9: "meme",
            10: "motivational",
            11: "movie",
            12: "news",
            13: "outdoors",
            14: "professional",
            15: "sports",
            16: "tech",
            17: "travel"
        }
        
        # Map predictions to category names
        predicted_categories = [category_mapping[pred] for pred in predictions]
        
        # Insert "N/A" for invalid descriptions
        final_predictions = ["N/A"] * len(descriptions)
        for idx, pred in zip(valid_indices, predicted_categories):
            final_predictions[idx] = pred
        
        # Write predictions back to the DataFrame
        df['predictions'] = final_predictions
        
        # Prepare data for batch update
        cell_range = f'B2:B{len(final_predictions) + 1}'  # Adjust the range as needed
        cell_values = [[pred] for pred in final_predictions]
        
        # Batch update the Google Sheet
        sheet.update(cell_range, cell_values)
    else:
        print("No valid descriptions to process.")

if __name__ == "__main__":
    main()
