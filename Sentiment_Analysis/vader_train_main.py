import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download("stopwords")
nltk.download("punkt")

# Load dataset
file_path = r"nifty_50_google_news_headlines.csv"
df = pd.read_csv(file_path)

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to clean text
def clean_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    tokens = word_tokenize(text)  # Tokenization
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return " ".join(tokens)

# Function to optimize sentiment scoring
def optimized_sentiment_score(text):
    text = clean_text(text)  # Preprocess text
    sentiment = analyzer.polarity_scores(text)
    compound = sentiment['compound']
    
    # Fine-tuned sentiment classification
    if compound >= 0.1:  # More precise threshold for positive
        return 1  # Positive
    elif compound <= -0.1:  # More precise threshold for negative
        return 0  # Negative
    else:
        return 0.5  # Neutral

# Apply text cleaning & sentiment scoring
df["Cleaned_Title"] = df["title"].apply(clean_text)
df["Sentiment_Score"] = df["Cleaned_Title"].apply(optimized_sentiment_score)

# Select required columns
output_df = df[["company", "title", "published", "source", "Sentiment_Score"]]

# Save to new CSV file
output_file = r"nifty50_sentiment_scores.csv"
output_df.to_csv(output_file, index=False)
print(f"Optimized sentiment scores saved to {output_file}")

# Feature Engineering: TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["Cleaned_Title"])
y = df["Sentiment_Score"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for RandomForestClassifier
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10]
}

clf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Save optimized model & vectorizer
joblib.dump(grid_search.best_estimator_, r"optimized_sentiment_model.pkl")
joblib.dump(vectorizer, r"tfidf_vectorizer.pkl")

print("Optimized sentiment model trained and saved successfully!")

# Function to predict sentiment on new data
def predict_sentiment(new_text):
    vectorizer = joblib.load(r"tfidf_vectorizer.pkl")
    model = joblib.load(r"optimized_sentiment_model.pkl")
    
    cleaned_text = clean_text(new_text)  # Preprocess input
    new_text_tfidf = vectorizer.transform([cleaned_text])  # Transform input
    prediction = model.predict(new_text_tfidf)[0]  # Predict sentiment
    return prediction

