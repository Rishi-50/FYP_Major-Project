import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load dataset (Ensure correct path)
file_path = r"nifty_50_google_news_headlines.csv"
df = pd.read_csv(file_path)

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to assign sentiment score
def get_sentiment_score(text):
    sentiment = analyzer.polarity_scores(str(text))  # Ensure text is string
    compound = sentiment['compound']
    if compound >= 0.05:
        return 1  # Positive
    elif compound <= -0.05:
        return 0  # Negative
    else:
        return 0.5  # Neutral

# Apply sentiment scoring to headlines
df["Sentiment_Score"] = df["title"].apply(get_sentiment_score)

# Select required columns
output_df = df[["company", "title", "published", "source", "Sentiment_Score"]]

# Save to a new CSV file
output_file = r"nifty50_sentiment_scores.csv"
output_df.to_csv(output_file, index=False)

print(f"Sentiment scores saved successfully to {output_file}")
