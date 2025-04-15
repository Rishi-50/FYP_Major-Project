import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(page_title="Sentiment-Based Portfolio Optimization", layout="wide")

# Dictionary of Nifty 50 stocks with their Yahoo Finance symbols
NIFTY_50_STOCKS = {
    'ADANIENT': 'ADANIENT.NS',
    'ADANIPORTS': 'ADANIPORTS.NS',
    'APOLLOHOSP': 'APOLLOHOSP.NS',
    'ASIANPAINT': 'ASIANPAINT.NS',
    'AXISBANK': 'AXISBANK.NS',
    'BAJAJ-AUTO': 'BAJAJ-AUTO.NS',
    'BAJFINANCE': 'BAJFINANCE.NS',
    'BAJAJFINSV': 'BAJAJFINSV.NS',
    'BPCL': 'BPCL.NS',
    'BHARTIARTL': 'BHARTIARTL.NS',
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
    """Fetch and prepare historical price data for analysis"""
    try:
        # Map tickers to Yahoo Finance symbols
        yf_tickers = [NIFTY_50_STOCKS.get(ticker, ticker) for ticker in tickers]
        
        # Fetch price data with error handling
        price_data = pd.DataFrame()
        failed_downloads = []
        
        for ticker, yf_ticker in zip(tickers, yf_tickers):
            try:
                data = yf.download(yf_ticker, start=start_date, end=end_date, progress=False)['Close']
                if not data.empty:
                    price_data[ticker] = data
                else:
                    failed_downloads.append(ticker)
            except Exception as e:
                failed_downloads.append(ticker)
                st.warning(f"Could not download data for {ticker}: {str(e)}")
        
        if price_data.empty:
            st.error("Could not download price data for any selected stocks.")
            return None, None, None
        
        if failed_downloads:    
            st.warning(f"Could not download data for: {', '.join(failed_downloads)}")
        
        # Calculate returns
        returns = price_data.pct_change().dropna()
        
        # Calculate additional metrics
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        expected_returns = returns.mean() * 252     # Annualized returns
        
        return returns, volatility, expected_returns
    
    except Exception as e:
        st.error(f"Error preparing data: {str(e)}")
        return None, None, None

# Optimize portfolio with sentiment
def optimize_portfolio_with_sentiment(returns, sentiment_scores, risk_free_rate=0.05):
    """Optimize portfolio weights considering both returns and sentiment"""
    n_assets = len(returns.columns)
    
    # Calculate expected returns and covariance
    mu = returns.mean() * 252
    S = returns.cov() * 252
    
    # Incorporate sentiment scores into expected returns
    sentiment_adjustment = pd.Series(sentiment_scores, index=returns.columns)
    adjusted_mu = mu + (sentiment_adjustment * 0.02)  # Adjust impact of sentiment
    
    # Define optimization problem
    def portfolio_stats(weights):
        portfolio_return = np.sum(adjusted_mu * weights)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(S, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk
        return -sharpe_ratio  # Negative because we want to maximize
    
    # Constraints
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
    )
    bounds = tuple((0, 1) for _ in range(n_assets))  # Weights between 0 and 1
    
    # Initial guess (equal weights)
    initial_weights = np.array([1/n_assets] * n_assets)
    
    # Optimize
    result = minimize(
        portfolio_stats,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    optimal_weights = pd.Series(result.x, index=returns.columns)
    
    # Calculate portfolio metrics
    portfolio_return = np.sum(adjusted_mu * optimal_weights)
    portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(S, optimal_weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk
    
    return optimal_weights, portfolio_return, portfolio_risk, sharpe_ratio

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
    # Title and Introduction with better styling
    st.markdown("""
    <style>
    .big-font {
        font-size:40px !important;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 20px;
    }
    .section-header {
        font-size:24px !important;
        font-weight: bold;
        color: #333;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    .feature-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #1E88E5;
    }
    .explanation-box {
        background-color: #f0f7ff;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #1E88E5;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="big-font">📊 Sentiment-Based Portfolio Optimization</p>', unsafe_allow_html=True)
    
    # Introduction and Feature Overview
    st.markdown("""
    Welcome to the Sentiment-Based Portfolio Optimization tool! This advanced platform combines traditional financial metrics 
    with market sentiment analysis to create optimized investment portfolios.
    """)
    
    # Load sentiment data first
    with st.spinner("Loading market sentiment data..."):
        sentiment_df = load_sentiment_data()
    
    if sentiment_df is None:
        st.error("⚠️ Could not load sentiment data. Please ensure the file exists in the correct location.")
        return
    
    # Create expandable sections for feature explanations
    with st.expander("ℹ️ How Sentiment-Based Portfolio Optimization Works", expanded=True):
        st.markdown("""
        ### Key Components
        
        1. **Market Sentiment Analysis** 🔍
           - Analyzes news and social media sentiment for each stock
           - Converts qualitative data into quantitative sentiment scores
           - Tracks sentiment trends and momentum
        
        2. **Portfolio Optimization** 📈
           - Combines traditional metrics with sentiment data
           - Optimizes for risk-adjusted returns
           - Considers both historical performance and market sentiment
        
        3. **Risk Management** ⚖️
           - Monitors portfolio volatility
           - Tracks sentiment-based risk indicators
           - Provides diversification recommendations
        """)
    
    # Settings Section
    st.markdown('<p class="section-header">Portfolio Settings</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        #### 📅 Analysis Period
        Select the time range for your portfolio analysis:
        """)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        selected_start = st.date_input(
            "Start Date",
            start_date,
            help="Historical data will be analyzed from this date"
        )
        selected_end = st.date_input(
            "End Date",
            end_date,
            help="Analysis will consider data up to this date"
        )
    
    with col2:
        st.markdown("""
        #### ⚖️ Risk Parameters
        Adjust your risk preferences:
        """)
        risk_free_rate = st.slider(
            "Risk-Free Rate (%)",
            0.0, 10.0, 3.0,
            help="The return rate of a 'risk-free' investment (e.g., government bonds)"
        ) / 100
    
    # Stock Selection
    st.markdown("""
    #### 🎯 Stock Selection
    Choose stocks for your portfolio:
    """)
    available_tickers = list(NIFTY_50_STOCKS.keys())
    if len(available_tickers) == 0:
        st.error("No stocks found in the sentiment data")
        return
    
    selected_tickers = st.multiselect(
        "Select stocks",
        available_tickers,
        default=available_tickers[:5],
        help="Select multiple stocks to create a diversified portfolio"
    )
    
    if not selected_tickers:
        st.warning("⚠️ Please select at least one stock to continue")
        return
    
    # Analysis Section
    st.markdown('<p class="section-header">Portfolio Analysis</p>', unsafe_allow_html=True)
    
    with st.spinner("Analyzing data and optimizing portfolio..."):
        # Fetch and prepare data
        result = fetch_and_prepare_data(selected_tickers, selected_start, selected_end)
        
        if result[0] is None:
            st.error("Could not proceed with portfolio optimization due to data issues.")
            return
        
        returns, volatility, expected_returns = result
        
        # Calculate sentiment features
        sentiment_features = {}
        latest_sentiments = {}
        
        for ticker in selected_tickers:
            ticker_sentiment = calculate_sentiment_features(sentiment_df, ticker)
            if not ticker_sentiment.empty:
                sentiment_features[ticker] = ticker_sentiment
                latest_sentiments[ticker] = ticker_sentiment['Sentiment_Score'].iloc[-1]
            else:
                latest_sentiments[ticker] = 0
        
        # Optimize portfolio
        sentiment_scores = pd.Series(latest_sentiments)
        optimal_weights, portfolio_return, portfolio_risk, sharpe_ratio = optimize_portfolio_with_sentiment(
            returns, sentiment_scores, risk_free_rate
        )
        
        # Create tabs for organized display
        tab1, tab2, tab3 = st.tabs([
            "📈 Portfolio Analysis",
            "🗞️ Sentiment Analysis",
            "⚠️ Risk Assessment"
        ])
        
        with tab1:
            st.markdown('<p class="section-header">Optimized Portfolio Allocation</p>', unsafe_allow_html=True)
            
            # Enhanced portfolio summary
            portfolio_summary = pd.DataFrame({
                'Stock': selected_tickers,
                'Weight (%)': optimal_weights * 100,
                'Expected Return (%)': expected_returns * 100,
                'Volatility (%)': volatility * 100,
                'Sentiment Score': [latest_sentiments[ticker] for ticker in selected_tickers]
            })
            
            # Display formatted summary with improved styling
            st.dataframe(
                portfolio_summary.style
                .format({
                    'Weight (%)': '{:.2f}%',
                    'Expected Return (%)': '{:.2f}%',
                    'Volatility (%)': '{:.2f}%',
                    'Sentiment Score': '{:.3f}'
                })
                .background_gradient(subset=['Weight (%)'], cmap='Blues')
                .background_gradient(subset=['Sentiment Score'], cmap='RdYlGn', vmin=-1, vmax=1)
            )
            
            # Detailed explanation of portfolio metrics
            st.markdown("""
            <div class="explanation-box">
            ### 📊 Understanding Portfolio Metrics
            
            1. **Portfolio Weights**
               - Shows the optimal allocation for each stock
               - Higher weights indicate greater importance in the portfolio
               - Weights sum to 100% across all stocks
            
            2. **Expected Returns**
               - Projected annual returns for each stock
               - Based on historical performance and sentiment
               - Higher values indicate better return potential
            
            3. **Volatility**
               - Measures the risk of each stock
               - Higher values indicate more price fluctuation
               - Lower values suggest more stable returns
            
            4. **Sentiment Score**
               - Current market sentiment (-1 to +1)
               - Positive values indicate positive market sentiment
               - Negative values suggest negative market sentiment
            </div>
            """, unsafe_allow_html=True)
            
            # Enhanced portfolio performance visualization
            st.markdown("### 📈 Portfolio Performance vs. Market Sentiment")
            
            # Create an enhanced figure with better styling
            fig = go.Figure()
            
            # Add cumulative returns line with improved styling
            portfolio_returns = returns.dot(optimal_weights)
            fig.add_trace(go.Scatter(
                x=portfolio_returns.index,
                y=portfolio_returns.cumsum(),
                name="Portfolio Returns",
                line=dict(color='#1E88E5', width=2),
                fill='tozeroy',
                fillcolor='rgba(30,136,229,0.1)'
            ))
            
            # Add sentiment line with improved styling
            avg_sentiment = pd.DataFrame()
            for ticker in returns.columns:
                if ticker in sentiment_features:
                    weight = optimal_weights[ticker]
                    if ticker not in avg_sentiment.columns:
                        avg_sentiment[ticker] = sentiment_features[ticker]['Sentiment_Score'] * weight
            
            avg_sentiment = avg_sentiment.sum(axis=1)
            
            fig.add_trace(go.Scatter(
                x=avg_sentiment.index,
                y=avg_sentiment,
                name="Weighted Sentiment",
                line=dict(color='#43A047', width=2, dash='dot'),
                yaxis='y2'
            ))
            
            # Update layout with better styling
            fig.update_layout(
                title={
                    'text': 'Portfolio Performance vs Market Sentiment',
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                xaxis_title="Date",
                yaxis_title="Cumulative Returns",
                yaxis2=dict(
                    title="Sentiment Score",
                    overlaying="y",
                    side="right",
                    range=[-1, 1],
                    showgrid=False
                ),
                template='plotly_white',
                hovermode='x unified',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor='rgba(255,255,255,0.8)'
                ),
                margin=dict(l=60, r=60, t=50, b=50)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Key metrics with explanations
            st.markdown("### 📊 Key Portfolio Metrics")
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            total_return = (1 + portfolio_return) ** (252/len(returns)) - 1
            annual_return = portfolio_return
            
            with metric_col1:
                st.metric(
                    "Expected Annual Return",
                    f"{annual_return:.2%}",
                    help="Projected yearly return based on historical data and sentiment"
                )
            with metric_col2:
                st.metric(
                    "Portfolio Risk",
                    f"{portfolio_risk:.2%}",
                    help="Annual portfolio volatility (standard deviation of returns)"
                )
            with metric_col3:
                st.metric(
                    "Sharpe Ratio",
                    f"{sharpe_ratio:.2f}",
                    help="Risk-adjusted return metric (higher is better)"
                )
            
            # Detailed explanation of key metrics
            st.markdown("""
            <div class="explanation-box">
            ### 📈 Understanding Key Metrics
            
            1. **Expected Annual Return**
               - Projected yearly return of the portfolio
               - Combines historical returns with sentiment analysis
               - Higher values indicate better return potential
            
            2. **Portfolio Risk**
               - Measures the volatility of the portfolio
               - Based on historical price movements
               - Lower values indicate more stable returns
            
            3. **Sharpe Ratio**
               - Risk-adjusted return metric
               - Higher values indicate better risk-adjusted returns
               - Compares returns to risk-free rate
            </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<p class="section-header">Market Sentiment Analysis</p>', unsafe_allow_html=True)
            
            # Enhanced sentiment summary with explanations
            st.markdown("""
            <div class="explanation-box">
            ### Understanding Sentiment Metrics
            
            1. **Current Sentiment**
               - Latest market sentiment (-1 to +1)
               - Positive values indicate positive news/market sentiment
               - Negative values suggest negative market sentiment
            
            2. **Sentiment Trend**
               - 20-day moving average of sentiment
               - Shows the overall direction of market sentiment
               - Helps identify persistent positive/negative sentiment
            
            3. **Sentiment Volatility**
               - Measures stability of market sentiment
               - Higher values indicate more uncertain or mixed sentiment
               - Lower values suggest more stable market perception
            
            4. **Momentum**
               - Rate of sentiment change
               - Positive values show improving sentiment
               - Negative values show deteriorating sentiment
            </div>
            """, unsafe_allow_html=True)
            
            sentiment_summary = pd.DataFrame({
                'Stock': selected_tickers,
                'Current Sentiment': [latest_sentiments[ticker] for ticker in selected_tickers],
                'Sentiment Trend': [sentiment_features[ticker]['rolling_sentiment'].iloc[-1] if ticker in sentiment_features else 0 for ticker in selected_tickers],
                'Sentiment Volatility': [sentiment_features[ticker]['sentiment_volatility'].iloc[-1] if ticker in sentiment_features else 0 for ticker in selected_tickers],
                'Momentum': [sentiment_features[ticker]['sentiment_momentum'].iloc[-1] if ticker in sentiment_features else 0 for ticker in selected_tickers]
            })
            
            st.dataframe(
                sentiment_summary.style
                .format({
                    'Current Sentiment': '{:.3f}',
                    'Sentiment Trend': '{:.3f}',
                    'Sentiment Volatility': '{:.3f}',
                    'Momentum': '{:.3f}'
                })
                .background_gradient(subset=['Current Sentiment'], cmap='RdYlGn', vmin=-1, vmax=1)
                .background_gradient(subset=['Momentum'], cmap='RdYlGn')
            )
        
        with tab3:
            st.markdown('<p class="section-header">Risk Assessment</p>', unsafe_allow_html=True)
            
            # Risk metrics explanation
            st.markdown("""
            <div class="explanation-box">
            ### Understanding Risk Metrics
            
            1. **Portfolio Volatility**: {:.2%}
               - Measures the overall portfolio risk
               - Based on historical price movements
               - Lower values indicate more stable returns
            
            2. **Sentiment Volatility**
               - Tracks stability of market sentiment
               - Higher values suggest uncertain market perception
               - Used to adjust position sizes
            
            3. **Diversification Analysis**
               - Shows correlations between assets
               - Helps identify concentration risks
               - Guides portfolio rebalancing decisions
            </div>
            """.format(portfolio_risk), unsafe_allow_html=True)
            
            # Display correlation heatmap
            st.markdown("### 📊 Portfolio Correlation Analysis")
            corr_matrix = returns.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=corr_matrix.index,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmin=-1,
                zmax=1
            ))
            
            fig.update_layout(
                title={
                    'text': "Stock Correlation Heatmap",
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                width=700,
                height=700,
                margin=dict(l=60, r=60, t=50, b=50)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="explanation-box">
            ### 📈 Interpreting the Correlation Heatmap
            
            1. **Correlation Values**
               - Dark Red (1.0): Perfect positive correlation
               - Dark Blue (-1.0): Perfect negative correlation
               - White (0): No correlation
            
            2. **Diversification Impact**
               - Lower correlations between stocks indicate better diversification
               - Helps identify potential concentration risks
               - Guides portfolio rebalancing decisions
            
            3. **Risk Management**
               - Use correlations to identify similar stocks
               - Avoid overexposure to highly correlated assets
               - Balance portfolio with uncorrelated assets
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 