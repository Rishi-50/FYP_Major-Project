import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time

# Set page configuration
st.set_page_config(page_title="InsightFlow - A smart portfolio optimizizer for stocks", layout="wide")

def get_market_data():
    try:
        # Use the correct Nifty 50 symbol with .NS suffix
        nifty = yf.Ticker("^NSEI")
        
        # Add retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Fetch data with longer period to ensure we get some data
                nifty_data = nifty.history(period="5d")
                if not nifty_data.empty:
                    break
                time.sleep(2)  # Wait before retry
            except Exception as e:
                if attempt == max_retries - 1:
                    st.error(f"Failed to fetch market data after {max_retries} attempts.")
                    return None
                time.sleep(2)  # Wait before retry
        
        if nifty_data.empty:
            st.warning("No market data available. Using default values.")
            return {
                'nifty_price': 22000,  # Default value
                'nifty_change': 0.0,
                'volume': 0,
                'volume_change': 0,
                'volatility': 0
            }
        
        # Get the most recent data points
        current_price = round(nifty_data['Close'].iloc[-1], 2)
        prev_price = nifty_data['Close'].iloc[-2] if len(nifty_data) > 1 else current_price
        
        # Calculate metrics
        change_pct = round(((current_price - prev_price) / prev_price) * 100, 2)
        
        # Handle volume data safely
        try:
            volume = round(nifty_data['Volume'].iloc[-1] / 1e6, 2)  # Convert to millions
            prev_volume = nifty_data['Volume'].iloc[-2] if len(nifty_data) > 1 else volume
            volume_change = round(((volume - prev_volume) / prev_volume) * 100, 2) if prev_volume != 0 else 0
        except:
            volume = 0
            volume_change = 0
        
        # Calculate volatility with proper handling of NaN values
        returns = nifty_data['Close'].pct_change(fill_method=None)
        volatility = round(returns.std(skipna=True) * np.sqrt(252) * 100, 2)  # Annualized volatility
        
        return {
            'nifty_price': current_price,
            'nifty_change': change_pct,
            'volume': volume,
            'volume_change': volume_change,
            'volatility': volatility
        }
    except Exception as e:
        st.error(f"Error fetching market data: {str(e)}")
        return None

def main():
    # Header section
    st.title("📊 InsightFlow - A smart portfolio optimizizer for stocks")
    
    # Welcome message
    st.markdown("""
    ## Welcome to Your Intelligent Investment Platform
    
    Optimize your investment portfolio using advanced analytics, market sentiment,
    and machine learning techniques.
    """)
    
    # Features overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📈 Traditional Portfolio Optimization
        
        Optimize your portfolio using modern portfolio theory:
        - Mean-variance optimization
        - Risk-adjusted returns
        - Efficient frontier analysis
        """)
    
    with col2:
        st.markdown("""
        ### 🗞️ Sentiment-Based Analysis
        
        Enhance your portfolio with market sentiment:
        - News sentiment analysis
        - Social media trends
        - Market momentum indicators
        """)
    
    with col3:
        st.markdown("""
        ### 🚫 Feature Temporarily Disabled
        
        This feature is currently under maintenance:
        - Check back later for updates
        - Feature will be re-enabled soon
        - Contact support for more information
        """)
    
    # Market Overview Section
    st.markdown("## 📊 Market Overview")
    
    # Add cache to prevent too frequent updates
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_cached_market_data():
        return get_market_data()
    
    # Fetch real-time market data
    market_data = get_cached_market_data()
    
    if market_data:
        # Market metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric(
                "Nifty 50",
                f"{market_data['nifty_price']:,.2f}",
                f"{market_data['nifty_change']}%",
                help="Current Nifty 50 index value and daily change"
            )
        
        with metric_col2:
            sentiment = "Positive" if market_data['nifty_change'] > 0 else "Negative"
            st.metric(
                "Market Sentiment",
                sentiment,
                "↑" if market_data['nifty_change'] > 0 else "↓",
                help="Simple sentiment based on price movement"
            )
        
        with metric_col3:
            st.metric(
                "Volatility Index",
                f"{market_data['volatility']}%",
                "",
                help="Annualized volatility based on daily returns"
            )
        
        with metric_col4:
            st.metric(
                "Trading Volume (M)",
                f"{market_data['volume']}M",
                f"{market_data['volume_change']}%",
                help="Daily trading volume in millions and change"
            )
    else:
        st.error("Unable to fetch market data. Please check your internet connection.")
    
    # Getting Started Guide
    st.markdown("""
    ## 🚀 Getting Started
    
    Follow these steps to optimize your portfolio:
    1. Choose your optimization approach (Traditional, Sentiment-based, or AI-powered)
    2. Select your investment universe and constraints
    3. Review the optimization results and insights
    4. Implement the recommended portfolio allocation
    """)
    
    # Resources Section
    st.markdown("""
    ## 📚 Resources & Documentation
    
    - 📖 User Guide: Learn how to use the platform effectively
    - 📊 Market Research: Access latest market analysis and reports
    - 🔍 API Documentation: Technical documentation for developers
    - ❓ FAQ: Frequently asked questions and answers
    """)

if __name__ == "__main__":
    main()
