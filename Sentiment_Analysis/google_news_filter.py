import feedparser
import pandas as pd
from dateutil import parser
from datetime import datetime, timedelta
import pytz
import time

# Function to fetch Google News RSS feed
def fetch_google_news(query):
    query = query.replace(" ", "+")  # Ensure query formatting
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    
    if not feed.entries:
        print(f"No news found or request blocked for query: {query}")
    
    return feed.entries

# Function to filter headlines
def filter_headlines(entries, trusted_sources, date_threshold):
    filtered = []
    for entry in entries:
        try:
            published_date = parser.parse(entry.published)  # Convert to datetime

            # Convert offset-naive datetime to UTC
            if published_date.tzinfo is None:
                published_date = published_date.replace(tzinfo=pytz.UTC)
            else:
                published_date = published_date.astimezone(pytz.UTC)

            source = entry.get("source", {}).get("title", "Unknown")

            # Apply filtering
            if published_date >= date_threshold and source in trusted_sources:
                filtered.append({
                    "title": entry.title,
                    "published": published_date.strftime("%Y-%m-%d"),  # Format date
                    "source": source,
                    "link": entry.link
                })
        except Exception as e:
            print(f"Skipping an entry due to error: {e}")
    
    return filtered

def get_articles_for_stock(stock_name, trusted_sources, date_threshold):
    queries = [stock_name, f"{stock_name} stock", f"{stock_name} earnings", f"{stock_name} performance"]
    unique_articles = {}
    
    for query in queries:
        print(f"Fetching news for: {query}...")
        entries = fetch_google_news(query)

        filtered_entries = filter_headlines(entries, trusted_sources, date_threshold)

        for entry in filtered_entries:
            unique_articles[entry["title"]] = entry  # Avoid duplicates

        if len(unique_articles) >= 100:  # Stop when 100 articles are collected
            break

        time.sleep(1)  # Avoid sending too many requests too quickly
    
    return list(unique_articles.values())

# Function to process and save headlines for all Nifty 50 companies
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

    # List of trusted sources
    trusted_sources = {
        "Bloomberg", "Reuters", "Economic Times", "Moneycontrol", "CNBC", "The Hindu Business Line",
        "Financial Times", "Business Standard", "Livemint", "Forbes", "NDTV Profit", "The Wall Street Journal",
        "The Economic Times", "The Financial Express", "The Times of India", "Business Today", "The Hindu", "Business Insider", "InvestorsHub", "Yahoo Finance", "StockTwits", "MarketWatch", "New York Times", "The Guardian", "Zee Business", "ET Now", "BSE India", "NSE India"
    }

    # Set date threshold (last 2 years)
    date_threshold = (datetime.now(pytz.UTC) - timedelta(days=2 * 365))

    all_headlines = []

    # Loop through each company and fetch headlines
    for company in nifty_50_companies:
        print(f"\nFetching at least 100 news articles for {company}...")
        articles = get_articles_for_stock(company, trusted_sources, date_threshold)

        for entry in articles:
            entry["company"] = company  # Add company name
            all_headlines.append(entry)

        print(f"✅ {len(articles)} articles fetched for {company}")

    # Convert to DataFrame
    df = pd.DataFrame(all_headlines)

    # Save to CSV
    filename = "filtered_nifty_50_news.csv"
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} filtered headlines to {filename}")

if __name__ == "__main__":
    main()
