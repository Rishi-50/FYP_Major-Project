import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
from tqdm import tqdm

# Load dataset
file_path = r"nifty_50_google_news_headlines.csv"
df = pd.read_csv(file_path)

# Load FinBERT tokenizer & model
model_name = "ProsusAI/finbert"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Define sentiment analysis pipeline
nlp_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Function to analyze sentiment
def get_finbert_sentiment(text):
    result = nlp_pipeline(text[:512])  # Truncate text if too long
    label = result[0]["label"]
    
    # Convert FinBERT labels to numerical scores
    if label == "positive":
        return 1  # Positive
    elif label == "negative":
        return 0  # Negative
    else:
        return 0.5  # Neutral

# Apply sentiment analysis to headlines
tqdm.pandas()  # Enable progress bar
df["Sentiment_Score"] = df["title"].progress_apply(get_finbert_sentiment)

# Save results
output_file = r"finbert_sentiment_scores.csv"
df.to_csv(output_file, index=False)

print(f"FinBERT sentiment scores saved to {output_file}")
