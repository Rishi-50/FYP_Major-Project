import feedparser
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from dateutil import parser
import time
import requests
import json
from bs4 import BeautifulSoup
import yfinance as yf
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('news_collection.log'),
        logging.StreamHandler()
    ]
)

# List of user agents to rotate
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59'
]

class NewsCollector:
    def __init__(self):
        self.trusted_sources = {
            # Financial News
            "Bloomberg", "Reuters", "Economic Times", "Moneycontrol", "CNBC", "The Hindu Business Line",
            "Financial Times", "Business Standard", "Livemint", "Forbes", "NDTV Profit", "Wall Street Journal",
            "The Economic Times", "Financial Express", "Times of India", "Business Today", "The Hindu",
            "Business Insider", "MarketWatch", "Benzinga", "Seeking Alpha", "The Motley Fool",
            
            # Market Data Sources
            "Yahoo Finance", "Investing.com", "StockTwits", "TradingView",
            
            # Indian Sources
            "Zee Business", "ET Now", "BSE India", "NSE India", "Money Control",
            
            # Additional Sources
            "Financial Post", "TheStreet", "Barron's", "Market Realist", "InvestorPlace",
            "Economic Times Markets", "Business Wire", "PR Newswire"
        }
        
        self.session = requests.Session()

    def get_random_headers(self):
        """Generate random headers for requests"""
        return {
            'User-Agent': random.choice(USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }

    def fetch_google_news(self, query):
        """Enhanced Google News RSS feed fetcher with retry mechanism"""
        max_retries = 3
        delay = 1
        
        for attempt in range(max_retries):
            try:
                query = query.replace(" ", "+")
                url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
                feed = feedparser.parse(url)
                
                if feed.entries:
                    return feed.entries
                
                logging.warning(f"No entries found for query: {query}")
                time.sleep(delay)
                delay *= 2
                
            except Exception as e:
                logging.error(f"Error fetching Google News (attempt {attempt + 1}): {str(e)}")
                time.sleep(delay)
                delay *= 2
        
        return []

    def fetch_yahoo_finance_news(self, ticker):
        """Fetch news from Yahoo Finance"""
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            
            if not news:
                return []
            
            formatted_news = []
            for item in news:
                formatted_news.append({
                    'title': item.get('title', ''),
                    'published': datetime.fromtimestamp(item.get('providerPublishTime', 0)).strftime('%Y-%m-%d'),
                    'source': item.get('publisher', 'Yahoo Finance'),
                    'link': item.get('link', ''),
                    'summary': item.get('summary', '')
                })
            
            return formatted_news
            
        except Exception as e:
            logging.error(f"Error fetching Yahoo Finance news: {str(e)}")
            return []

    def fetch_moneycontrol_news(self, query):
        """Fetch news from Moneycontrol"""
        try:
            url = f"https://www.moneycontrol.com/news/business/stocks/{query.lower().replace(' ', '-')}"
            response = self.session.get(url, headers=self.get_random_headers(), timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            articles = soup.find_all('li', class_='clearfix')
            
            news_items = []
            for article in articles[:15]:  # Get first 15 articles
                try:
                    title_elem = article.find('h2')
                    if not title_elem:
                        continue
                        
                    title = title_elem.text.strip()
                    link = title_elem.find('a')['href'] if title_elem.find('a') else ''
                    date_elem = article.find('span', class_='date')
                    date = date_elem.text.strip() if date_elem else ''
                    
                    if title and date:
                        news_items.append({
                            'title': title,
                            'published': parser.parse(date).strftime('%Y-%m-%d'),
                            'source': 'Moneycontrol',
                            'link': link
                        })
                except Exception as e:
                    logging.warning(f"Error parsing Moneycontrol article: {str(e)}")
                    continue
                    
            return news_items
            
        except Exception as e:
            logging.error(f"Error fetching from Moneycontrol: {str(e)}")
            return []

    def fetch_economic_times_news(self, query):
        """Fetch news from Economic Times"""
        try:
            url = f"https://economictimes.indiatimes.com/markets/stocks/news"
            params = {'searchtext': query}
            response = self.session.get(url, params=params, headers=self.get_random_headers(), timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            articles = soup.find_all('div', class_='eachStory')
            
            news_items = []
            for article in articles[:15]:  # Get first 15 articles
                try:
                    title_elem = article.find('h3')
                    if not title_elem:
                        continue
                        
                    title = title_elem.text.strip()
                    link = 'https://economictimes.indiatimes.com' + title_elem.find('a')['href'] if title_elem.find('a') else ''
                    date_elem = article.find('time')
                    date = date_elem.text.strip() if date_elem else ''
                    
                    if title and date:
                        news_items.append({
                            'title': title,
                            'published': parser.parse(date).strftime('%Y-%m-%d'),
                            'source': 'Economic Times',
                            'link': link
                        })
                except Exception as e:
                    logging.warning(f"Error parsing Economic Times article: {str(e)}")
                    continue
                    
            return news_items
            
        except Exception as e:
            logging.error(f"Error fetching from Economic Times: {str(e)}")
            return []

    def get_articles_for_stock(self, stock_name, date_threshold):
        """Get articles from all sources for a given stock"""
        queries = [
            stock_name,
            f"{stock_name} stock",
            f"{stock_name} share price",
            f"{stock_name} company news",
            f"{stock_name} market analysis",
            f"{stock_name} financial results"
        ]
        
        all_articles = {}
        
        # Collect from multiple sources in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Google News
            google_futures = [executor.submit(self.fetch_google_news, query) for query in queries]
            
            # Yahoo Finance
            yahoo_future = executor.submit(self.fetch_yahoo_finance_news, f"{stock_name}.NS")
            
            # Moneycontrol
            moneycontrol_future = executor.submit(self.fetch_moneycontrol_news, stock_name)
            
            # Economic Times
            et_future = executor.submit(self.fetch_economic_times_news, stock_name)
            
            # Collect Google News results
            for future in google_futures:
                try:
                    entries = future.result()
                    filtered_entries = self.filter_headlines(entries, date_threshold)
                    for entry in filtered_entries:
                        all_articles[entry["title"]] = entry
                except Exception as e:
                    logging.error(f"Error collecting Google News: {str(e)}")
            
            # Collect Yahoo Finance results
            try:
                yahoo_articles = yahoo_future.result()
                for article in yahoo_articles:
                    all_articles[article["title"]] = article
            except Exception as e:
                logging.error(f"Error collecting Yahoo Finance news: {str(e)}")
            
            # Collect Moneycontrol results
            try:
                mc_articles = moneycontrol_future.result()
                for article in mc_articles:
                    all_articles[article["title"]] = article
            except Exception as e:
                logging.error(f"Error collecting Moneycontrol news: {str(e)}")
            
            # Collect Economic Times results
            try:
                et_articles = et_future.result()
                for article in et_articles:
                    all_articles[article["title"]] = article
            except Exception as e:
                logging.error(f"Error collecting Economic Times news: {str(e)}")
        
        return list(all_articles.values())

    def filter_headlines(self, entries, date_threshold):
        """Filter headlines based on date and source"""
        filtered = []
        for entry in entries:
            try:
                published_date = parser.parse(entry.published)
                
                if published_date.tzinfo is None:
                    published_date = published_date.replace(tzinfo=pytz.UTC)
                else:
                    published_date = published_date.astimezone(pytz.UTC)
                
                source = entry.get("source", {}).get("title", "Unknown")
                
                if published_date >= date_threshold and (source in self.trusted_sources or source == "Unknown"):
                    filtered.append({
                        "title": entry.title,
                        "published": published_date.strftime("%Y-%m-%d"),
                        "source": source,
                        "link": entry.link
                    })
            except Exception as e:
                logging.warning(f"Error filtering headline: {str(e)}")
        
        return filtered

def main():
    # List of Nifty 50 companies
    nifty_50_companies = [
        "ADANIPORTS", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE", 
        "BAJAJFINSV", "BPCL", "BHARTIARTL", "BRITANNIA", "CIPLA", 
        "COALINDIA", "DIVISLAB", "DRREDDY", "EICHERMOT", "GRASIM", 
        "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO", 
        "HINDUNILVR", "ICICIBANK", "ITC", "INDUSINDBK", "INFY", 
        "JSWSTEEL", "KOTAKBANK", "LT", "M&M", "MARUTI", 
        "NTPC", "NESTLEIND", "ONGC", "POWERGRID", "RELIANCE", 
        "SBILIFE", "SHREECEM", "SBIN", "SUNPHARMA", "TCS", 
        "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TECHM", "TITAN", 
        "ULTRACEMCO", "UPL", "WIPRO", "ZEEL"
    ]

    # Set date threshold (last 2 years)
    date_threshold = datetime.now(pytz.UTC) - timedelta(days=2 * 365)

    # Initialize NewsCollector
    collector = NewsCollector()
    all_headlines = []

    # Process each company with progress bar
    for company in tqdm(nifty_50_companies, desc="Processing companies"):
        logging.info(f"Collecting news for {company}...")
        articles = collector.get_articles_for_stock(company, date_threshold)
        
        for entry in articles:
            entry["company"] = company
            all_headlines.append(entry)
        
        logging.info(f"Collected {len(articles)} articles for {company}")
        time.sleep(2)  # Prevent rate limiting

    # Convert to DataFrame
    df = pd.DataFrame(all_headlines)

    # Save to CSV with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"enhanced_nifty_50_news_{timestamp}.csv"
    df.to_csv(filename, index=False)
    logging.info(f"Saved {len(df)} headlines to {filename}")

    # Save summary statistics
    summary = {
        'total_articles': len(df),
        'articles_per_company': df.groupby('company').size().to_dict(),
        'articles_per_source': df.groupby('source').size().to_dict(),
        'date_range': {
            'start': df['published'].min(),
            'end': df['published'].max()
        }
    }

    with open(f'collection_summary_{timestamp}.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    logging.info("Data collection completed successfully!")

if __name__ == "__main__":
    main() 