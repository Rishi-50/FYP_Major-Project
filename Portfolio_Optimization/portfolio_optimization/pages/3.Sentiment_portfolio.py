import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import warnings
import plotly.express as px
warnings.filterwarnings('ignore')

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
        sentiment_df = pd.read_csv("../../Sentiment_Analysis/finbert_sentiment_scores.csv")
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

def create_sequences(X, y, time_steps=5):
    """Create sequences for LSTM model"""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def build_lstm_model(input_shape, learning_rate=0.001):
    """Build and compile LSTM model with optimized architecture"""
    try:
        # Set memory growth to prevent OOM errors
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
    except:
        pass

    # More efficient architecture with better regularization
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=input_shape,
             kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.2),
        LSTM(16, return_sequences=False,
             kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.2),
        Dense(16, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dense(1)
    ])
    
    # Use Adam optimizer with clipnorm only
    optimizer = Adam(
        learning_rate=learning_rate,
        clipnorm=1.0  # Remove clipvalue, use only clipnorm
    )
    
    model.compile(
        optimizer=optimizer,
        loss='huber',  # More robust to outliers than MSE
        metrics=['mae']
    )
    return model

def prepare_lstm_data(returns, sentiment_df, ticker, time_steps=5):
    """Prepare data for LSTM model with better preprocessing"""
    try:
        # Get sentiment data for the ticker
        ticker_sentiment = sentiment_df[sentiment_df['ticker'] == ticker].copy()
        
        # Convert index to datetime if it's not already
        if not isinstance(ticker_sentiment.index, pd.DatetimeIndex):
            ticker_sentiment.index = pd.to_datetime(ticker_sentiment['date'])
        
        # Ensure returns index is datetime
        returns.index = pd.to_datetime(returns.index)
        
        # Align dates and resample to business days
        start_date = max(returns.index.min(), ticker_sentiment.index.min())
        end_date = min(returns.index.max(), ticker_sentiment.index.max())
        
        # Filter data to common date range
        returns_filtered = returns.loc[start_date:end_date][ticker]
        ticker_sentiment_filtered = ticker_sentiment.loc[start_date:end_date]
        
        # Create DataFrame with aligned data
        data = pd.DataFrame({
            'returns': returns_filtered,
            'sentiment': ticker_sentiment_filtered['Sentiment_Score']
        })
        
        # Handle missing data more robustly
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Add technical indicators with error handling
        try:
            data['rolling_mean'] = data['returns'].rolling(window=5, min_periods=1).mean()
            data['rolling_std'] = data['returns'].rolling(window=5, min_periods=1).std()
            data['momentum'] = data['returns'].diff(periods=5)
        except Exception as e:
            st.warning(f"Could not calculate some technical indicators for {ticker}: {str(e)}")
            # Provide default values if calculation fails
            data['rolling_mean'] = 0
            data['rolling_std'] = 0
            data['momentum'] = 0
        
        # Fill any remaining NaN values with 0
        data = data.fillna(0)
        
        # Remove outliers more conservatively
        for col in data.columns:
            lower = data[col].quantile(0.005)
            upper = data[col].quantile(0.995)
            data[col] = data[col].clip(lower=lower, upper=upper)
        
        # Ensure we have enough data
        if len(data) < max(30, time_steps * 2):  # Minimum 30 samples or 2x time_steps
            st.warning(f"Insufficient data points for {ticker}")
            return None
        
        # Scale the features
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Create sequences
        X, y = create_sequences(scaled_data, scaled_data[:, 0], time_steps)
        
        # Ensure we have enough sequences
        if len(X) < 30:
            st.warning(f"Insufficient sequences for {ticker}")
            return None
        
        # Split into train and test sets
        train_size = int(len(X) * 0.8)
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]
        
        # Verify shapes
        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            st.warning(f"Invalid split sizes for {ticker}")
            return None
        
        return (X_train, y_train, X_test, y_test, scaler)
    except Exception as e:
        st.error(f"Error preparing LSTM data for {ticker}: {str(e)}")
        return None

def train_lstm_model(X_train, y_train, input_shape):
    """Train LSTM model with optimized training process"""
    try:
        # Verify input shapes
        if X_train.shape[0] == 0 or y_train.shape[0] == 0:
            st.error("Empty training data")
            return None, None
            
        model = build_lstm_model(input_shape)
        
        # Enhanced early stopping with reduced patience
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            min_delta=0.001
        )
        
        # Reduce learning rate on plateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.0001
        )
        
        try:
            # Train model with optimized parameters and error handling
            history = model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=16,
                validation_split=0.2,
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # Verify training completed successfully
            if history.history['loss'][-1] != history.history['loss'][-1]:  # Check for NaN
                st.error("Training failed: NaN loss detected")
                return None, None
                
            return model, history
            
        except Exception as e:
            st.error(f"Error during model training: {str(e)}")
            return None, None
            
    except Exception as e:
        st.error(f"Error setting up model training: {str(e)}")
        return None, None

def predict_returns_with_lstm(model, X_test, scaler):
    """Make predictions with error handling and confidence estimation"""
    try:
        if len(X_test) == 0:
            st.error("Empty test data")
            return None
            
        # Make predictions in smaller batches
        batch_size = 16  # Reduced batch size
        predictions = []
        
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i + batch_size]
            try:
                batch_pred = model.predict(batch, verbose=0)
                predictions.extend(batch_pred.flatten())
            except Exception as e:
                st.error(f"Error in batch prediction: {str(e)}")
                return None
        
        predictions = np.array(predictions)
        
        # Verify predictions
        if np.any(np.isnan(predictions)):
            st.error("NaN values in predictions")
            return None
        
        # Create dummy array for inverse transform
        dummy = np.zeros((len(predictions), scaler.n_features_in_))
        dummy[:, 0] = predictions
        
        # Inverse transform predictions
        try:
            predictions_unscaled = scaler.inverse_transform(dummy)[:, 0]
        except Exception as e:
            st.error(f"Error in inverse transform: {str(e)}")
            return None
        
        # Clip extreme predictions
        predictions_unscaled = np.clip(
            predictions_unscaled,
            np.percentile(predictions_unscaled, 1),
            np.percentile(predictions_unscaled, 99)
        )
        
        return predictions_unscaled
    except Exception as e:
        st.error(f"Error in prediction pipeline: {str(e)}")
        return None

def analyze_stock_with_lstm(returns, sentiment_df, ticker, time_steps=5):
    """Complete LSTM analysis pipeline with enhanced error handling"""
    try:
        # Check if we have enough data
        if len(returns) < 30:
            st.warning(f"Insufficient data for {ticker} to perform LSTM analysis")
            return None
            
        # Prepare data
        data = prepare_lstm_data(returns, sentiment_df, ticker, time_steps)
        if data is None:
            return None
        
        X_train, y_train, X_test, y_test, scaler = data
        
        # Train model
        model, history = train_lstm_model(X_train, y_train, X_train.shape[1:])
        if model is None:
            return None
        
        # Make predictions
        predictions = predict_returns_with_lstm(model, X_test, scaler)
        if predictions is None:
            return None
        
        # Calculate error metrics
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        
        # Calculate prediction confidence
        confidence = 1 / (1 + rmse)
        
        return {
            'predictions': predictions,
            'actual': y_test,
            'rmse': rmse,
            'confidence': confidence,
            'model': model,
            'scaler': scaler,
            'history': history.history if history else None
        }
    except Exception as e:
        st.error(f"Error in LSTM analysis for {ticker}: {str(e)}")
        return None

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
        tab1, tab2 = st.tabs([
            "📈 Portfolio Analysis & Insights",
            "🤖 LSTM Predictions"
        ])
        
        with tab1:
            st.markdown('<p class="section-header">Comprehensive Portfolio Analysis</p>', unsafe_allow_html=True)
            
            # Portfolio Allocation Section
            st.markdown("### 📊 Portfolio Allocation and Performance")
            
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
            
            # Key Portfolio Metrics
            st.markdown("### 📈 Key Portfolio Metrics")
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

            # Sentiment Analysis Section
            st.markdown("### 🗞️ Market Sentiment Analysis")
            
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

            # Portfolio Performance Visualization
            st.markdown("### 📊 Portfolio Performance and Sentiment Trends")
            
            # Create enhanced figure with both returns and sentiment
            fig = go.Figure()
            
            # Add cumulative returns line
            portfolio_returns = returns.dot(optimal_weights)
            fig.add_trace(go.Scatter(
                x=portfolio_returns.index,
                y=portfolio_returns.cumsum(),
                name="Portfolio Returns",
                line=dict(color='#1E88E5', width=2),
                fill='tozeroy',
                fillcolor='rgba(30,136,229,0.1)'
            ))
            
            # Add weighted sentiment line
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
                height=500,
                hovermode='x unified',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor='rgba(255,255,255,0.8)'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk Assessment Section
            st.markdown("### ⚠️ Risk Assessment and Correlation Analysis")
            
            # Display correlation heatmap
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
                height=600,
                margin=dict(l=60, r=60, t=50, b=50)
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # Comprehensive Analysis Insights
            st.markdown("""
            <div class="explanation-box">
            ### 🎯 Portfolio Insights Summary
            
            #### Portfolio Composition
            - Diversification Level: {:.0%} of maximum possible
            - Number of Significant Positions: {}
            - Highest Weight Position: {} ({:.2%})
            
            #### Risk-Return Profile
            - Expected Annual Return: {:.2%}
            - Portfolio Risk Level: {:.2%}
            - Risk-Adjusted Return (Sharpe): {:.2f}
            
            #### Market Sentiment
            - Overall Portfolio Sentiment: {:.2f}
            - Sentiment Trend: {}
            - Number of Positive Sentiment Stocks: {}
            
            #### Risk Analysis
            - Average Stock Correlation: {:.2f}
            - Diversification Benefit: {:.2%}
            - Risk Concentration: {:.2%}
            </div>
            """.format(
                1 - optimal_weights.max(),  # Diversification level
                sum(optimal_weights > 0.05),  # Significant positions
                optimal_weights.idxmax(),  # Highest weight stock
                optimal_weights.max(),  # Highest weight
                portfolio_return,  # Expected return
                portfolio_risk,  # Portfolio risk
                sharpe_ratio,  # Sharpe ratio
                np.mean(list(latest_sentiments.values())),  # Overall sentiment
                "Improving" if np.mean(list(latest_sentiments.values())) > 0 else "Declining",
                sum(1 for s in latest_sentiments.values() if s > 0),  # Positive sentiment count
                corr_matrix.mean().mean(),  # Average correlation
                1 - portfolio_risk / (volatility * optimal_weights).sum(),  # Diversification benefit
                (optimal_weights ** 2).sum()  # Risk concentration
            ), unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<p class="section-header">LSTM Return Predictions</p>', unsafe_allow_html=True)
            
            # LSTM Analysis explanation
            st.markdown("""
            <div class="explanation-box">
            ### 🤖 LSTM Model Analysis
            
            This section uses Long Short-Term Memory (LSTM) neural networks to predict future returns based on:
            1. Historical price movements
            2. Market sentiment scores
            3. Technical indicators
            
            The model analyzes patterns in both price movements and sentiment to forecast potential future returns.
            </div>
            """, unsafe_allow_html=True)
            
            # Progress bar for LSTM analysis
            lstm_progress = st.progress(0)
            lstm_status = st.empty()
            
            # Store LSTM results
            lstm_results = {}
            
            # Analyze each stock with LSTM
            for idx, ticker in enumerate(selected_tickers):
                lstm_status.text(f"Training LSTM model for {ticker}...")
                lstm_progress.progress((idx + 1) / len(selected_tickers))
                
                result = analyze_stock_with_lstm(returns, sentiment_df, ticker)
                if result is not None:
                    lstm_results[ticker] = result
            
            lstm_status.empty()
            lstm_progress.empty()
            
            if lstm_results:
                # Add toggle for combined vs individual views
                view_option = st.radio(
                    "Prediction View",
                    ["Combined View", "Individual Views"],
                    horizontal=True
                )
                
                # Calculate average predicted returns (moved outside the views logic)
                avg_predictions = {}
                for ticker in lstm_results:
                    avg_predictions[ticker] = np.mean(lstm_results[ticker]['predictions'][-5:])  # Last 5 predictions
                
                if view_option == "Combined View":
                    # Create a combined plot for all stocks
                    st.markdown("### Combined LSTM Predictions for All Selected Stocks")
                    
                    # Add aggregate predictions analysis
                    st.markdown("### 📊 Aggregate Predictions Analysis")
                    
                    prediction_df = pd.DataFrame({
                        'Stock': list(avg_predictions.keys()),
                        'Predicted Return': list(avg_predictions.values()),
                        'Model RMSE': [lstm_results[ticker]['rmse'] for ticker in avg_predictions],
                        'Weight in Portfolio': [optimal_weights[ticker] for ticker in avg_predictions]
                    })
                    
                    st.dataframe(
                        prediction_df.style
                        .format({
                            'Predicted Return': '{:.2%}',
                            'Model RMSE': '{:.4f}',
                            'Weight in Portfolio': '{:.2%}'
                        })
                        .background_gradient(subset=['Predicted Return'], cmap='RdYlGn')
                    )
                    
                    # Add bar chart of predicted returns by stock
                    st.markdown("### Predicted Returns by Stock")
                    
                    fig_bar = go.Figure()
                    
                    # Sort stocks by predicted return
                    sorted_stocks = sorted(avg_predictions.items(), key=lambda x: x[1], reverse=True)
                    sorted_tickers = [item[0] for item in sorted_stocks]
                    sorted_returns = [item[1] for item in sorted_stocks]
                    
                    # Set color based on return (positive/negative)
                    colors = ['#4CAF50' if ret > 0 else '#F44336' for ret in sorted_returns]
                    
                    # Create bar chart
                    fig_bar.add_trace(go.Bar(
                        x=sorted_tickers,
                        y=sorted_returns,
                        marker_color=colors,
                        text=[f"{ret:.2%}" for ret in sorted_returns],
                        textposition='auto',
                        hovertemplate=(
                            "<b>%{x}</b><br>" +
                            "Predicted Return: %{y:.2%}<br>" +
                            "Portfolio Weight: %{customdata:.2%}<br>"
                        ),
                        customdata=[optimal_weights[ticker] for ticker in sorted_tickers]
                    ))
                    
                    # Add a horizontal line at y=0 (zero return)
                    fig_bar.add_hline(
                        y=0, 
                        line_dash="dash", 
                        line_color="gray"
                    )
                    
                    fig_bar.update_layout(
                        title="Stock Predicted Returns Ranking",
                        xaxis_title="Stock",
                        yaxis_title="Predicted Return",
                        yaxis=dict(tickformat='.2%'),
                        template='plotly_white',
                        height=400,
                        margin=dict(l=50, r=50, t=80, b=80)
                    )
                    
                    # Rotate x-axis labels if many stocks
                    if len(sorted_tickers) > 5:
                        fig_bar.update_layout(
                            xaxis=dict(tickangle=-45)
                        )
                    
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # Prediction-based insights
                    weighted_pred_return = sum(avg_predictions[ticker] * optimal_weights[ticker] 
                                           for ticker in avg_predictions)
                    
                    st.markdown(f"""
                    <div class="explanation-box">
                    ### 🎯 Portfolio Prediction Insights
                    
                    - Predicted Portfolio Return: {weighted_pred_return:.2%}
                    - Number of Stocks with Positive Predictions: {sum(1 for v in avg_predictions.values() if v > 0)}
                    - Number of Stocks with Negative Predictions: {sum(1 for v in avg_predictions.values() if v < 0)}
                    
                    These predictions combine both technical analysis and sentiment data to provide a comprehensive forecast.
                    </div>
                    """, unsafe_allow_html=True)
                
                if view_option == "Individual Views":
                    # Display individual LSTM predictions for each stock
                    for ticker in selected_tickers:
                        if ticker in lstm_results:
                            st.markdown(f"### {ticker} Predictions")
                            
                            result = lstm_results[ticker]
                            
                            # Create prediction plot
                            fig = go.Figure()
                            
                            # Create x-axis values (time steps)
                            x_values = list(range(len(result['actual'])))
                
                            # Add actual returns
                            fig.add_trace(go.Scatter(
                                x=x_values,
                                y=result['actual'],
                                name='Actual Returns',
                                mode='lines',
                                line=dict(color='#1E88E5', width=2)
                            ))
                            
                            # Add predictions
                            fig.add_trace(go.Scatter(
                                x=x_values,
                                y=result['predictions'],
                                name='Predicted Returns',
                                mode='lines',
                                line=dict(color='#43A047', width=2, dash='dash')
                            ))
                            
                            # Add zero reference line
                            fig.add_hline(
                                y=0, 
                                line_dash="dash", 
                                line_color="gray", 
                                annotation_text="Zero Return",
                                annotation_position="bottom right"
                            )
                
                            fig.update_layout(
                                title=f"{ticker} - LSTM Return Predictions",
                                xaxis_title="Time Steps",
                                yaxis_title="Returns",
                                template='plotly_white',
                                height=400,
                                yaxis=dict(tickformat='.2%'),
                                hovermode='x unified'
                            )
                
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display metrics
                            st.metric(
                                "Prediction RMSE",
                                f"{result['rmse']:.4f}",
                                help="Root Mean Square Error of predictions (lower is better)"
                            )
                            
                            # Add prediction insights
                            last_prediction = result['predictions'][-1]
                            prediction_direction = "positive" if last_prediction > 0 else "negative"
                            
                            st.markdown(f"""
                            <div class="feature-box">
                            #### Prediction Insights
                            - Latest predicted return: {last_prediction:.2%}
                            - Predicted trend: {prediction_direction.title()}
                            - Model confidence (based on RMSE): {result['confidence']:.2%}
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.warning("Could not generate LSTM predictions for any selected stocks. Please try with different stocks or time periods.")

if __name__ == "__main__":
    main() 