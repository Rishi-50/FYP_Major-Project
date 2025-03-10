import feedparser
import pandas as pd

# Function to fetch Google News RSS feed
def fetch_google_news(query):
    query = query.replace(" ", "+")  # Ensure query formatting
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    
    if not feed.entries:
        print(f"No news found or blocked request for query: {query}")
    
    return feed.entries

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
    
    all_headlines = []
    
    # Loop through each company and fetch headlines
    for company in nifty_50_companies:
        print(f"Fetching headlines for {company}...")
        entries = fetch_google_news(company)
        
        if not entries:
            print(f"No headlines found for {company}.")
            continue
        
        for entry in entries:
            headline_data = {
                'company': company,
                'title': entry.title,
                'published': entry.published,
                'source': entry.get("source", {}).get("title", "Unknown"),
                'link': entry.link
            }
            all_headlines.append(headline_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_headlines)
    
    # Save to CSV
    filename = "nifty_50_google_news_headlines.csv"
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} headlines to {filename}")

if __name__ == "__main__":
    main()