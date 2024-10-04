import pandas as pd
import re
from collections import Counter

# Load your Excel file
file_path = "C:\\Users\\nyter\\Desktop\\Scraper\\shorts_descriptions.xlsx"  # Update with your file path
df = pd.read_excel(file_path)

# Assuming your descriptions are in the first column (Column A)
descriptions = df.iloc[:, 0].astype(str).tolist()  # Convert to list of strings

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation, emojis, and hashtags
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\#\w+', '', text)     # Remove hashtags
    return text

# Preprocess all descriptions and tokenize
cleaned_descriptions = [preprocess_text(desc) for desc in descriptions]
words = [word for desc in cleaned_descriptions for word in desc.split() if word]  # Exclude empty strings

# Count word frequencies across all descriptions
word_counts = Counter(words)

# Get the top 1000 most common words and their counts
top_common_words = word_counts.most_common(1000)

# Create a DataFrame for the top common words
top_words_df = pd.DataFrame(top_common_words, columns=['Word', 'Count'])

# Save the top common words DataFrame to a new Excel file
output_file_path = 'top_common_words.xlsx'  # Change this if needed
top_words_df.to_excel(output_file_path, index=False)

print("Top 1000 common words have been saved successfully.")