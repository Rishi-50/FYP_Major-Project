import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import numpy as np
from tqdm import tqdm
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

class EnhancedSentimentAnalyzer:
    def __init__(self):
        # Initialize FinBERT
        self.model_name = "ProsusAI/finbert"
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(self.model_name)
        self.nlp_pipeline = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)
        
        # Initialize VADER
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Financial terms for domain-specific preprocessing
        self.positive_financial_terms = {
            'bullish', 'outperform', 'buy', 'upgrade', 'growth', 'profit', 'surge',
            'rally', 'gain', 'positive', 'strong', 'upbeat', 'recovery'
        }
        self.negative_financial_terms = {
            'bearish', 'underperform', 'sell', 'downgrade', 'loss', 'decline',
            'fall', 'negative', 'weak', 'downturn', 'crash', 'risk'
        }

    def preprocess_text(self, text):
        """Enhanced preprocessing for financial news"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Tokenization
        words = word_tokenize(text)
        
        # Remove stopwords while keeping important financial terms
        stop_words = set(stopwords.words('english')) - self.positive_financial_terms - self.negative_financial_terms
        words = [word for word in words if word not in stop_words]
        
        return ' '.join(words)

    def get_finbert_sentiment(self, text):
        """Get FinBERT sentiment with error handling"""
        try:
            result = self.nlp_pipeline(text[:512])
            label = result[0]["label"]
            score = result[0]["score"]
            
            if label == "positive":
                return score
            elif label == "negative":
                return -score
            else:
                return 0
        except Exception as e:
            print(f"FinBERT Error: {e}")
            return 0

    def get_vader_sentiment(self, text):
        """Get VADER sentiment with error handling"""
        try:
            sentiment = self.vader_analyzer.polarity_scores(str(text))
            return sentiment['compound']
        except Exception as e:
            print(f"VADER Error: {e}")
            return 0

    def calculate_temporal_weight(self, date_str):
        """Calculate weight based on news recency"""
        try:
            news_date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            current_date = datetime.now()
            days_diff = (current_date - news_date).days
            
            # Exponential decay weight
            weight = np.exp(-0.1 * days_diff)
            return min(max(weight, 0.1), 1.0)
        except Exception as e:
            print(f"Date processing error: {e}")
            return 0.5

    def analyze_sentiment(self, df):
        """Main sentiment analysis function"""
        results = []
        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            try:
                # Preprocess text
                processed_text = self.preprocess_text(row['title'])
                
                # Get sentiment scores
                finbert_score = self.get_finbert_sentiment(processed_text)
                vader_score = self.get_vader_sentiment(processed_text)
                
                # Calculate temporal weight
                temporal_weight = self.calculate_temporal_weight(row['published'])
                
                # Combine scores with weights
                # FinBERT gets higher weight (0.6) as it's specifically trained on financial text
                combined_score = temporal_weight * (0.6 * finbert_score + 0.4 * vader_score)
                
                # Add domain-specific adjustments
                pos_terms = sum(1 for term in self.positive_financial_terms if term in processed_text)
                neg_terms = sum(1 for term in self.negative_financial_terms if term in processed_text)
                domain_adjustment = (pos_terms - neg_terms) * 0.1
                
                final_score = np.clip(combined_score + domain_adjustment, -1, 1)
                
                results.append({
                    'company': row['company'],
                    'title': row['title'],
                    'published': row['published'],
                    'source': row['source'],
                    'finbert_score': finbert_score,
                    'vader_score': vader_score,
                    'temporal_weight': temporal_weight,
                    'final_sentiment': final_score,
                    'confidence': abs(final_score)  # Higher absolute score = higher confidence
                })
                
            except Exception as e:
                print(f"Error processing row: {e}")
                continue
        
        return pd.DataFrame(results)

    def analyze_stock_sentiment(self, stock_symbol):
        """Analyze sentiment for a specific stock"""
        try:
            # Load the news data
            df = pd.read_csv("nifty_50_google_news_headlines.csv")
            
            # Filter for the specific stock
            stock_df = df[df['company'] == stock_symbol].copy()
            
            if len(stock_df) == 0:
                return None, "No news data found for this stock"
            
            # Analyze sentiment
            results_df = self.analyze_sentiment(stock_df)
            
            # Calculate aggregate metrics
            avg_sentiment = results_df['final_sentiment'].mean()
            sentiment_std = results_df['final_sentiment'].std()
            recent_sentiment = results_df.sort_values('published', ascending=False)['final_sentiment'].iloc[0]
            
            sentiment_summary = {
                'average_sentiment': avg_sentiment,
                'sentiment_volatility': sentiment_std,
                'recent_sentiment': recent_sentiment,
                'confidence': np.mean(results_df['confidence']),
                'news_count': len(results_df)
            }
            
            return results_df, sentiment_summary
            
        except Exception as e:
            print(f"Error in stock analysis: {e}")
            return None, str(e)

if __name__ == "__main__":
    # Initialize analyzer
    analyzer = EnhancedSentimentAnalyzer()
    
    # Load and process all news data
    input_file = "nifty_50_google_news_headlines.csv"
    df = pd.read_csv(input_file)
    
    # Analyze sentiment for all news
    results_df = analyzer.analyze_sentiment(df)
    
    # Save detailed results
    output_file = "enhanced_sentiment_scores.csv"
    results_df.to_csv(output_file, index=False)
    
    print(f"Enhanced sentiment analysis completed. Results saved to {output_file}") 