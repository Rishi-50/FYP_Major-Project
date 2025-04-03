import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# Stock symbol mapping for Yahoo Finance
STOCK_SYMBOL_MAPPING = {
    'INDUSINDBK': 'INDUSINDBK.NS',
    'DIVISLAB': 'DIVISLAB.NS',
    'SBIN': 'SBIN.NS',
    'BAJAJFINSV': 'BAJAJFINSV.NS',
    'BHARTIARTL': 'BHARTIARTL.NS',
    'HDFCBANK': 'HDFCBANK.NS',
    'INFY': 'INFY.NS',
    'TCS': 'TCS.NS',
    'HINDUNILVR': 'HINDUNILVR.NS',
    'ICICIBANK': 'ICICIBANK.NS',
    'HDFC': 'HDFC.NS',
    'RELIANCE': 'RELIANCE.NS',
    'KOTAKBANK': 'KOTAKBANK.NS',
    'ITC': 'ITC.NS',
    'HCLTECH': 'HCLTECH.NS',
    'WIPRO': 'WIPRO.NS',
    'AXISBANK': 'AXISBANK.NS',
    'ASIANPAINT': 'ASIANPAINT.NS',
    'MARUTI': 'MARUTI.NS',
    'ULTRACEMCO': 'ULTRACEMCO.NS',
    'NESTLEIND': 'NESTLEIND.NS',
    'TITAN': 'TITAN.NS',
    'BAJFINANCE': 'BAJFINANCE.NS',
    'HDFCLIFE': 'HDFCLIFE.NS',
    'BAJAJFINSV': 'BAJAJFINSV.NS',
    'ADANIENT': 'ADANIENT.NS',
    'ADANIPORTS': 'ADANIPORTS.NS',
    'ADANIPOWER': 'ADANIPOWER.NS',
    'BAJAJ-AUTO': 'BAJAJ-AUTO.NS',
    'BAJAJFINSV': 'BAJAJFINSV.NS',
    'BPCL': 'BPCL.NS',
    'BRITANNIA': 'BRITANNIA.NS',
    'CIPLA': 'CIPLA.NS',
    'COALINDIA': 'COALINDIA.NS',
    'DLF': 'DLF.NS',
    'DRREDDY': 'DRREDDY.NS',
    'EICHERMOT': 'EICHERMOT.NS',
    'GAIL': 'GAIL.NS',
    'GODREJCP': 'GODREJCP.NS',
    'HDFCBANK': 'HDFCBANK.NS',
    'HDFCLIFE': 'HDFCLIFE.NS',
    'HEROMOTOCO': 'HEROMOTOCO.NS',
    'HINDALCO': 'HINDALCO.NS',
    'HINDPETRO': 'HINDPETRO.NS',
    'HINDUNILVR': 'HINDUNILVR.NS',
    'ICICIBANK': 'ICICIBANK.NS',
    'ICICIGI': 'ICICIGI.NS',
    'ICICIPRULI': 'ICICIPRULI.NS',
    'IOC': 'IOC.NS',
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
    'SHREECEM': 'SHREECEM.NS',
    'SUNPHARMA': 'SUNPHARMA.NS',
    'TCS': 'TCS.NS',
    'TECHM': 'TECHM.NS',
    'TITAN': 'TITAN.NS',
    'ULTRACEMCO': 'ULTRACEMCO.NS',
    'UPL': 'UPL.NS',
    'WIPRO': 'WIPRO.NS',
    'ZEEL': 'ZEEL.NS'
}

# Load sentiment data
@st.cache_data
def load_sentiment_data():
    try:
        # Try relative path first
        sentiment_df = pd.read_csv("../../../Sentiment_Analysis/finbert_sentiment_scores.csv")
    except FileNotFoundError:
        try:
            # If that fails, try absolute path
            sentiment_df = pd.read_csv("D:/FYP_Major-Project/Sentiment_Analysis/finbert_sentiment_scores.csv")
        except FileNotFoundError:
            st.error("Could not find sentiment data file. Please ensure the file exists in the correct location.")
            return None
    
    try:
        # Check and rename columns if needed
        if 'company' in sentiment_df.columns:
            sentiment_df = sentiment_df.rename(columns={'company': 'ticker'})
        elif 'title' in sentiment_df.columns:
            sentiment_df = sentiment_df.rename(columns={'title': 'ticker'})
        
        # Ensure we have the required columns
        required_columns = ['ticker', 'Sentiment_Score']
        if not all(col in sentiment_df.columns for col in required_columns):
            st.error(f"Sentiment data file is missing required columns: {required_columns}")
            return None
        
        # Add date column if it doesn't exist
        if 'date' not in sentiment_df.columns:
            if 'published' in sentiment_df.columns:
                sentiment_df['date'] = sentiment_df['published']
            else:
                sentiment_df['date'] = pd.Timestamp.now()
        
        # Ensure sentiment scores are numeric
        sentiment_df['Sentiment_Score'] = pd.to_numeric(sentiment_df['Sentiment_Score'], errors='coerce')
        
        # Convert date column
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        
        # Group by ticker and date, taking the mean sentiment score
        sentiment_df = sentiment_df.groupby(['ticker', 'date'])['Sentiment_Score'].mean().reset_index()
        
        # Sort by date
        sentiment_df = sentiment_df.sort_values('date')
        
        return sentiment_df
    except Exception as e:
        st.error(f"Error processing sentiment data: {str(e)}")
        return None

# Calculate sentiment-based features
def calculate_sentiment_features(sentiment_df, ticker):
    # Get sentiment data for the specific ticker
    ticker_sentiment = sentiment_df[sentiment_df['ticker'] == ticker].copy()
    
    # Calculate rolling sentiment metrics
    ticker_sentiment['rolling_sentiment'] = ticker_sentiment['Sentiment_Score'].rolling(window=20).mean()
    ticker_sentiment['sentiment_volatility'] = ticker_sentiment['Sentiment_Score'].rolling(window=20).std()
    
    # Calculate sentiment momentum
    ticker_sentiment['sentiment_momentum'] = ticker_sentiment['Sentiment_Score'].diff(5)
    
    return ticker_sentiment

# Fetch historical data and calculate returns
def fetch_and_prepare_data(tickers, start_date, end_date):
    # Map tickers to Yahoo Finance symbols
    yf_tickers = [STOCK_SYMBOL_MAPPING.get(ticker, ticker) for ticker in tickers]
    
    # Fetch price data
    price_data = yf.download(yf_tickers, start=start_date, end=end_date)['Close']
    
    # Rename columns back to original tickers
    ticker_mapping_reverse = {v: k for k, v in STOCK_SYMBOL_MAPPING.items()}
    price_data.columns = [ticker_mapping_reverse.get(col, col) for col in price_data.columns]
    
    # Calculate returns
    returns = price_data.pct_change().dropna()
    
    # Calculate additional metrics
    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
    expected_returns = returns.mean() * 252     # Annualized returns
    
    return returns, volatility, expected_returns

# Optimize portfolio with sentiment
def optimize_portfolio_with_sentiment(returns, sentiment_features, risk_free_rate=0.03):
    """
    Optimizes portfolio weights using both financial and sentiment metrics
    """
    num_assets = len(returns.columns)
    
    # Calculate sentiment-adjusted returns
    sentiment_adjusted_returns = returns.copy()
    for ticker in returns.columns:
        if ticker in sentiment_features:
            sentiment_score = sentiment_features[ticker]['Sentiment_Score'].iloc[-1]
            # Adjust returns based on sentiment (positive sentiment increases expected return)
            sentiment_adjusted_returns[ticker] *= (1 + sentiment_score * 0.1)
    
    # Calculate sentiment-adjusted metrics
    mu = sentiment_adjusted_returns.mean() * 252
    cov_matrix = sentiment_adjusted_returns.cov() * 252
    
    def objective(weights):
        portfolio_return = np.dot(weights, mu)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol
        
        # Add sentiment-based penalty for negative sentiment
        sentiment_penalty = 0
        for i, ticker in enumerate(returns.columns):
            if ticker in sentiment_features:
                sentiment_score = sentiment_features[ticker]['Sentiment_Score'].iloc[-1]
                sentiment_penalty += weights[i] * max(0, -sentiment_score)
        
        return -(sharpe_ratio - sentiment_penalty)
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = np.ones(num_assets) / num_assets
    
    result = minimize(objective, initial_weights, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    return result.x

def plot_sentiment_portfolio_analysis(returns, sentiment_features, weights):
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add returns line
    portfolio_returns = returns.dot(weights)
    fig.add_trace(go.Scatter(
        x=portfolio_returns.index,
        y=portfolio_returns.cumsum(),
        name="Portfolio Returns",
        line=dict(color='blue')
    ))
    
    # Add sentiment line
    sentiment_scores = pd.DataFrame()
    for ticker in returns.columns:
        if ticker in sentiment_features:
            sentiment_scores[ticker] = sentiment_features[ticker]['Sentiment_Score']
    
    avg_sentiment = sentiment_scores.mean(axis=1)
    fig.add_trace(go.Scatter(
        x=avg_sentiment.index,
        y=avg_sentiment,
        name="Average Sentiment",
        line=dict(color='red'),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Portfolio Performance vs Market Sentiment",
        xaxis_title="Date",
        yaxis_title="Cumulative Returns",
        yaxis2=dict(
            title="Sentiment Score",
            overlaying="y",
            side="right",
            range=[-1, 1]
        ),
        template='plotly_white'
    )
    
    return fig

def main():
    st.title("Sentiment-Based Portfolio Optimization")
    
    # Load sentiment data
    sentiment_df = load_sentiment_data()
    
    if sentiment_df is None:
        st.error("Failed to load sentiment data. Please check the data file and try again.")
        return
    
    # Sidebar controls
    st.sidebar.header("Portfolio Parameters")
    
    # Date range selection
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    selected_start = st.sidebar.date_input("Start Date", start_date)
    selected_end = st.sidebar.date_input("End Date", end_date)
    
    # Risk-free rate
    risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 10.0, 3.0) / 100
    
    # Stock selection
    st.sidebar.header("Select Stocks")
    available_tickers = sentiment_df['ticker'].unique()
    if len(available_tickers) == 0:
        st.error("No tickers found in sentiment data")
        return
        
    selected_tickers = st.sidebar.multiselect(
        "Choose stocks for your portfolio",
        available_tickers,
        default=available_tickers[:5]
    )
    
    if not selected_tickers:
        st.warning("Please select at least one stock")
        return
    
    # Fetch and prepare data
    returns, volatility, expected_returns = fetch_and_prepare_data(
        selected_tickers, selected_start, selected_end
    )
    
    # Calculate sentiment features
    sentiment_features = {}
    for ticker in selected_tickers:
        sentiment_features[ticker] = calculate_sentiment_features(
            sentiment_df, ticker
        )
    
    # Optimize portfolio
    optimal_weights = optimize_portfolio_with_sentiment(
        returns, sentiment_features, risk_free_rate
    )
    
    # Display results
    st.header("Portfolio Analysis")
    
    # Create portfolio summary
    portfolio_summary = pd.DataFrame({
        'Ticker': selected_tickers,
        'Weight': optimal_weights,
        'Expected Return': expected_returns,
        'Volatility': volatility,
        'Sharpe Ratio': (expected_returns - risk_free_rate) / volatility
    })
    
    st.subheader("Portfolio Allocation")
    st.dataframe(portfolio_summary)
    
    # Plot portfolio analysis
    fig = plot_sentiment_portfolio_analysis(returns, sentiment_features, optimal_weights)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display sentiment analysis
    st.subheader("Sentiment Analysis")
    sentiment_summary = pd.DataFrame({
        'Ticker': selected_tickers,
        'Current Sentiment': [sentiment_features[ticker]['Sentiment_Score'].iloc[-1] 
                            for ticker in selected_tickers],
        'Sentiment Volatility': [sentiment_features[ticker]['sentiment_volatility'].iloc[-1] 
                               for ticker in selected_tickers],
        'Sentiment Momentum': [sentiment_features[ticker]['sentiment_momentum'].iloc[-1] 
                             for ticker in selected_tickers]
    })
    st.dataframe(sentiment_summary)
    
    # Risk metrics
    st.subheader("Risk Metrics")
    portfolio_returns = returns.dot(optimal_weights)
    var_95 = np.percentile(portfolio_returns, 5)
    cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Value at Risk (95%)", f"{var_95:.2%}")
    with col2:
        st.metric("Conditional VaR (95%)", f"{cvar_95:.2%}")

if __name__ == "__main__":
    main() 