import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import os

# Nifty 50 stocks with their Yahoo Finance symbols
NIFTY_50_STOCKS = {
    'ADANIENT': 'ADANIENT.NS',
    'ADANIPORTS': 'ADANIPORTS.NS',
    'APOLLOHOSP': 'APOLLOHOSP.NS',
    'ASIANPAINT': 'ASIANPAINT.NS',
    'AXISBANK': 'AXISBANK.NS',
    'BAJAJ-AUTO': 'BAJAJ-AUTO.NS',
    'BAJFINANCE': 'BAJFINANCE.NS',
    'BAJAJFINSV': 'BAJAJFINSV.NS',
    'BHARTIARTL': 'BHARTIARTL.NS',
    'BPCL': 'BPCL.NS',
    'BRITANNIA': 'BRITANNIA.NS',
    'CIPLA': 'CIPLA.NS',
    'COALINDIA': 'COALINDIA.NS',
    'DIVISLAB': 'DIVISLAB.NS',
    'DRREDDY': 'DRREDDY.NS',
    'EICHERMOT': 'EICHERMOT.NS',
    'GRASIM': 'GRASIM.NS',
    'HCLTECH': 'HCLTECH.NS',
    'HDFCBANK': 'HDFCBANK.NS',
    'HDFCLIFE': 'HDFCLIFE.NS',
    'HEROMOTOCO': 'HEROMOTOCO.NS',
    'HINDALCO': 'HINDALCO.NS',
    'HINDUNILVR': 'HINDUNILVR.NS',
    'ICICIBANK': 'ICICIBANK.NS',
    'INDUSINDBK': 'INDUSINDBK.NS',
    'INFY': 'INFY.NS',
    'ITC': 'ITC.NS',
    'JSWSTEEL': 'JSWSTEEL.NS',
    'KOTAKBANK': 'KOTAKBANK.NS',
    'LT': 'LT.NS',
    'M&M': 'M&M.NS',
    'MARUTI': 'MARUTI.NS',
    'NESTLEIND': 'NESTLEIND.NS',
    'NTPC': 'NTPC.NS',
    'ONGC': 'ONGC.NS',
    'POWERGRID': 'POWERGRID.NS',
    'RELIANCE': 'RELIANCE.NS',
    'SBILIFE': 'SBILIFE.NS',
    'SBIN': 'SBIN.NS',
    'SUNPHARMA': 'SUNPHARMA.NS',
    'TATACONSUM': 'TATACONSUM.NS',
    'TATAMOTORS': 'TATAMOTORS.NS',
    'TATASTEEL': 'TATASTEEL.NS',
    'TCS': 'TCS.NS',
    'TECHM': 'TECHM.NS',
    'TITAN': 'TITAN.NS',
    'ULTRACEMCO': 'ULTRACEMCO.NS',
    'UPL': 'UPL.NS',
    'WIPRO': 'WIPRO.NS'
}

def fetch_stock_news(stock_name, symbol):
    """
    Fetch news for a specific stock from Yahoo Finance
    """
    try:
        print(f"Fetching news for {stock_name}...")
        
        # Create Ticker object
        stock = yf.Ticker(symbol)
        
        # Get news
        news = stock.news
        
        if news:
            # Convert news to DataFrame
            news_data = []
            for item in news:
                try:
                    news_item = {
                        'title': item.get('title', ''),
                        'published': pd.to_datetime(item.get('providerPublishTime', int(time.time())), unit='s'),
                        'link': item.get('link', ''),
                        'source': item.get('publisher', ''),
                        'company': stock_name,
                        'summary': item.get('summary', '')
                    }
                    # Only add news items that have a title
                    if news_item['title']:
                        news_data.append(news_item)
                except Exception as e:
                    print(f"Error processing news item for {stock_name}: {str(e)}")
                    continue
            
            if news_data:
                news_df = pd.DataFrame(news_data)
                print(f"Found {len(news_df)} headlines for {stock_name}")
                return news_df
        
        print(f"No news found for {stock_name}")
        return pd.DataFrame()
        
    except Exception as e:
        print(f"Error fetching news for {stock_name}: {str(e)}")
        return pd.DataFrame()

def main():
    print("Nifty 50 Yahoo Finance News Scraper")
    print("===================================")
    
    # Create directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Initialize empty DataFrame for all news
    all_news = pd.DataFrame()
    
    # Fetch news for each stock
    for stock_name, symbol in NIFTY_50_STOCKS.items():
        news_df = fetch_stock_news(stock_name, symbol)
        
        if not news_df.empty:
            # Add to main DataFrame
            all_news = pd.concat([all_news, news_df], ignore_index=True)
        
        # Add delay to avoid rate limiting
        time.sleep(1)
    
    # Clean and prepare final DataFrame
    if not all_news.empty:
        # Remove duplicates
        all_news = all_news.drop_duplicates(subset=['title', 'company'])
        
        # Sort by date
        all_news = all_news.sort_values('published')
        
        # Save to CSV
        output_file = os.path.join('data', 'nifty50_yahoo_news.csv')
        all_news.to_csv(output_file, index=False, encoding='utf-8')
        
        print("\nData Collection Summary:")
        print(f"Total headlines collected: {len(all_news)}")
        print(f"Date range: {all_news['published'].min()} to {all_news['published'].max()}")
        print(f"Companies covered: {len(all_news['company'].unique())}")
        print(f"\nData saved to: {output_file}")
        
        # Print sample of collected headlines
        print("\nSample Headlines:")
        print(all_news[['published', 'company', 'title']].head())
    
    else:
        print("No news data was collected.")

if __name__ == "__main__":
    main() 