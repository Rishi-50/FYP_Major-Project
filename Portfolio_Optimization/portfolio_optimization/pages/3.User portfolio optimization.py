import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Check if TensorFlow can be imported, otherwise use fallback prediction
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
    
    # Only configure TensorFlow if it loaded successfully
    try:
        # Configure TensorFlow for better performance
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        else:
            # Limit CPU usage if no GPU
            tf.config.threading.set_intra_op_parallelism_threads(2)
            tf.config.threading.set_inter_op_parallelism_threads(2)
    except:
        # If configuration fails, continue without it
        pass
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Fallback prediction function that doesn't require TensorFlow
def simple_prediction(data, ticker, future_days=30):
    """Simple prediction model that's used when TensorFlow has issues"""
    prices = data[ticker].values
    last_price = prices[-1]
    
    # Calculate recent trend (last 30 days)
    if len(prices) >= 30:
        recent_trend = (prices[-1] / prices[-30]) - 1
    else:
        recent_trend = 0.001  # Default small positive trend
    
    # Generate future prices based on simple trend
    predicted_prices = []
    for i in range(future_days):
        # Dampen the trend over time
        dampen_factor = np.exp(-i * 0.05)
        next_price = last_price * (1 + (recent_trend * dampen_factor/30))
        predicted_prices.append(next_price)
        last_price = next_price
    
    # Create future dates
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=future_days)
    
    return future_dates, np.array(predicted_prices).reshape(-1, 1)

# Optimized LSTM prediction function with error handling
@st.cache_data(ttl=3600)  # Cache results for 1 hour
def stock_price_prediction(data, ticker, future_days=30):
    """Prediction function that tries LSTM first, falls back to simple prediction if needed"""
    # Use simple prediction if TensorFlow is not available
    if not TENSORFLOW_AVAILABLE:
        st.warning("TensorFlow not available. Using simplified prediction model.")
        return simple_prediction(data, ticker, future_days)
    
    # Extract the data for the specific ticker
    if ticker not in data.columns:
        st.error(f"Ticker {ticker} not found in data")
        return None, None
    
    try:
        # Minimum data needed for prediction
        min_data_needed = 60
        prediction_days = min(min_data_needed, len(data) - 1)
        
        # Check if we have enough data
        if len(data) <= min_data_needed:
            st.warning(f"Limited data available ({len(data)} days). Using simplified model.")
            return simple_prediction(data, ticker, future_days)
        
        # Extract price data
        price_data = data[ticker].values.reshape(-1, 1)
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(price_data)
        
        # Prepare training data
        x_train = []
        y_train = []
        
        for i in range(prediction_days, len(scaled_data)):
            x_train.append(scaled_data[i-prediction_days:i, 0])
            y_train.append(scaled_data[i, 0])
        
        # Convert to numpy arrays
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        
        # Reshape for LSTM
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        # Build model with try-except to handle any TensorFlow errors
        try:
            # Create a smaller LSTM model for better performance
            model = Sequential()
            model.add(LSTM(units=30, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(Dropout(0.1))
            model.add(LSTM(units=30, return_sequences=False))
            model.add(Dropout(0.1))
            model.add(Dense(units=1))
            
            # Compile the model
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Train with smaller epochs, larger batch for performance
            model.fit(
                x_train, y_train, 
                epochs=10,
                batch_size=64, 
                verbose=0.1,
                shuffle=True
            )
            
            # Prepare test data
            test_data = scaled_data[-prediction_days:]
            x_test = np.array([test_data[:, 0]])
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
            
            # Make predictions for future days
            predicted_prices = []
            current_batch = x_test[0]
            
            for i in range(future_days):
                # Predict next price
                current_pred = model.predict(np.array([current_batch]), verbose=0)[0]
                predicted_prices.append(current_pred[0])
                
                # Update batch for next prediction
                current_batch = np.append(current_batch[1:], current_pred)
                current_batch = current_batch.reshape((prediction_days, 1))
            
            # Convert predictions back to original scale
            predicted_prices = np.array(predicted_prices).reshape(-1, 1)
            predicted_prices = scaler.inverse_transform(predicted_prices)
            
            # Clean up TensorFlow session to prevent memory issues
            tf.keras.backend.clear_session()
            
            # Create future dates
            last_date = data.index[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=future_days)
            
            return future_dates, predicted_prices
            
        except Exception as e:
            # If any TensorFlow error occurs, fall back to simple prediction
            print(f"LSTM Error: {str(e)}")
            st.warning("LSTM model encountered an error. Using simplified prediction model.")
            return simple_prediction(data, ticker, future_days)
            
    except Exception as e:
        # If any unexpected error occurs
        print(f"Prediction error: {str(e)}")
        st.error(f"Error in prediction: {str(e)}")
        return simple_prediction(data, ticker, future_days)

# Cache data fetching to improve performance
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_data(tickers, start_date, end_date):
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
        # Handle single ticker case
        if isinstance(data, pd.Series):
            data = pd.DataFrame(data)
            data.columns = [tickers]
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Portfolio optimization function
def optimize_portfolio(returns, risk_free_rate=0.05):
    try:
        n_assets = len(returns.columns)
        
        # Calculate expected returns and covariance
        mu = returns.mean() * 252
        S = returns.cov() * 252
        
        # Define optimization problem
        def portfolio_stats(weights):
            portfolio_return = np.sum(mu * weights)
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
        portfolio_return = np.sum(mu * optimal_weights)
        portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(S, optimal_weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk
        
        return optimal_weights, portfolio_return, portfolio_risk, sharpe_ratio
    except Exception as e:
        st.error(f"Error in portfolio optimization: {str(e)}")
        # Return safe fallback values
        weights = pd.Series([1/len(returns.columns)] * len(returns.columns), index=returns.columns)
        return weights, 0.05, 0.10, 0.50

# Main function
def main():
    st.title("Interactive Portfolio Analysis and Optimization")

    # Set up sidebar for user inputs
    st.sidebar.title("Portfolio Settings")
    
    # Date range selection
    st.sidebar.subheader("Analysis Period")
    start_date = st.sidebar.date_input("Start Date", pd.Timestamp("2022-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.Timestamp("today"))
    if start_date >= end_date:
        st.sidebar.error("End date must be after start date")
        return
    
    # Risk-free rate
    risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0) / 100
    
    # Ticker input
    st.sidebar.subheader("Portfolio Stocks")
    tickers_input = st.sidebar.text_input("Enter stock tickers (comma-separated)", "RELIANCE.NS,TCS.NS,HDFCBANK.NS,INFY.NS,ITC.NS")
    
    if tickers_input:
        tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]
        quantities = []
        prices = []

        st.sidebar.subheader("Enter Portfolio Details:")
        total_investments = 0.0

        for ticker in tickers:
            quantity = st.sidebar.number_input(f"Quantity of {ticker}:", min_value=0.0, value=1.0, step=1.0)
            price = st.sidebar.number_input(f"Buying Price of {ticker} (₹):", min_value=0.0, value=100.0, step=1.0)
            quantities.append(quantity)
            prices.append(price)
            total_investments += quantity * price

        if total_investments > 0:
            # Calculate total capital and weights
            investments = [q * p for q, p in zip(quantities, prices)]
            weights = [(inv / total_investments) * 100 for inv in investments]

            # Display portfolio allocation
            portfolio_df = pd.DataFrame({
                "Ticker": tickers,
                "Quantity": quantities,
                "Buying Price (₹)": prices,
                "Investment (₹)": investments,
                "Weight (%)": weights
            })

            st.subheader("Portfolio Allocation")
            st.dataframe(portfolio_df.style.format({
                "Investment (₹)": "{:.2f}",
                "Weight (%)": "{:.2f}"
            }))

            st.write(f"**Total Investment:** ₹{total_investments:.2f}")

            # Fetch historical data
            with st.spinner("Fetching historical data..."):
                data = fetch_data(tickers, start_date, end_date)
            
            if data is None or data.empty:
                st.error("Could not fetch historical data. Please check ticker symbols.")
                return

            # Calculate daily returns
            returns = data.pct_change().dropna()
            
            # Create tabs for different analyses
            tabs = st.tabs(["Portfolio Analysis", "Price Predictions", "Optimization"])
            
            with tabs[0]:
                st.subheader("Portfolio Performance")
                
                # Calculate portfolio value over time
                portfolio_value = pd.DataFrame(index=data.index)
                for i, ticker in enumerate(tickers):
                    if ticker in data.columns:
                        portfolio_value[ticker] = data[ticker] * quantities[i]
                
                portfolio_value['Total'] = portfolio_value.sum(axis=1)
                
                # Plot portfolio value
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=portfolio_value.index,
                    y=portfolio_value['Total'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='blue', width=2)
                ))
                
                fig.update_layout(
                    title='Portfolio Value Over Time',
                    xaxis_title='Date',
                    yaxis_title='Value (₹)',
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate and display performance metrics
                if len(portfolio_value) > 1:
                    initial_value = portfolio_value['Total'].iloc[0]
                    final_value = portfolio_value['Total'].iloc[-1]
                    total_return = (final_value / initial_value - 1) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Return", f"{total_return:.2f}%")
                    col2.metric("Initial Value", f"₹{initial_value:.2f}")
                    col3.metric("Current Value", f"₹{final_value:.2f}")
                
                # Display correlation matrix
                st.subheader("Asset Correlation Matrix")
                correlation = data.corr()
                fig = go.Figure(data=go.Heatmap(
                    z=correlation.values,
                    x=correlation.columns,
                    y=correlation.index,
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1
                ))
                
                fig.update_layout(
                    title='Asset Correlation Matrix',
                    width=700,
                    height=700
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tabs[1]:
                st.subheader("Price Predictions")
                
                # Stock selection for prediction
                prediction_ticker = st.selectbox("Select stock for prediction", tickers)
                
                # Show warning if data is limited
                if len(data) < 100:
                    st.warning("Limited historical data available. Predictions may be less accurate.")
                
                # Show progress during prediction
                with st.spinner("Generating price predictions..."):
                    future_dates, predicted_prices = stock_price_prediction(
                        data, prediction_ticker, future_days=7
                    )
                
                if future_dates is None or predicted_prices is None:
                    st.error("Prediction failed. Please try another stock or time period.")
                else:
                    # Create prediction DataFrame
                    pred_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted_Price': predicted_prices.flatten()
                    }).set_index('Date')
                    
                    # Concatenate historical and predicted data for plotting
                    historical = data[[prediction_ticker]].copy()
                    historical.columns = ['Actual_Price']
                    
                    # Create prediction plot
                    fig = go.Figure()
                    
                    # Add historical prices
                    fig.add_trace(go.Scatter(
                        x=historical.index,
                        y=historical['Actual_Price'],
                        mode='lines',
                        name='Historical Prices',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Add predicted prices
                    fig.add_trace(go.Scatter(
                        x=pred_df.index,
                        y=pred_df['Predicted_Price'],
                        mode='lines',
                        name='Predicted Prices',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                    
                    # Add confidence interval (simple approach)
                    historical_volatility = max(0.01, historical['Actual_Price'].pct_change().std() * np.sqrt(252))
                    
                    upper_bound = []
                    lower_bound = []
                    
                    for i, price in enumerate(predicted_prices.flatten()):
                        # Increase uncertainty with time
                        day_factor = np.sqrt(i + 1) * historical_volatility * 0.25
                        upper_bound.append(price * (1 + day_factor))
                        lower_bound.append(price * (1 - day_factor))
                    
                    # Add confidence intervals
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=upper_bound,
                        fill=None,
                        mode='lines',
                        line=dict(color='rgba(255, 0, 0, 0)'),
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=lower_bound,
                        fill='tonexty',
                        mode='lines',
                        name='Confidence Interval',
                        line=dict(color='rgba(255, 0, 0, 0)'),
                        fillcolor='rgba(255, 0, 0, 0.2)'
                    ))
                    
                    fig.update_layout(
                        title=f'Price Prediction for {prediction_ticker}',
                        xaxis_title='Date',
                        yaxis_title='Price (₹)',
                        template='plotly_white',
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Predicted price table
                    st.subheader("Price Predictions")
                    weekly_pred = pred_df.resample('W').first()
                    weekly_pred.index = weekly_pred.index.strftime('%Y-%m-%d')
                    weekly_pred.columns = ['Predicted Price (₹)']
                    st.dataframe(weekly_pred.style.format({"Predicted Price (₹)": "{:.2f}"}))
                    
                    # Prediction insights
                    current_price = historical['Actual_Price'].iloc[-1]
                    future_price = predicted_prices[-1][0]
                    predicted_return = (future_price / current_price - 1) * 100
                    
                    st.subheader("Prediction Insights")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Current Price", f"₹{current_price:.2f}")
                    col2.metric(f"Predicted Price (7 days)", f"₹{future_price:.2f}")
                    col3.metric("Predicted Return", f"{predicted_return:.2f}%", f"{predicted_return:.2f}%")
                    
                    # Add prediction explanation
                    if TENSORFLOW_AVAILABLE:
                        st.info("""
                        **About the Prediction Model**
                        
                        - The prediction uses a Long Short-Term Memory (LSTM) neural network
                        - LSTM can capture complex patterns and trends in time series data
                        - Confidence interval expands over time to reflect increasing uncertainty
                        - Predictions should be used as one of many inputs for investment decisions
                        """)
                    else:
                        st.info("""
                        **About the Prediction Model**
                        
                        - The prediction uses a simplified time series forecasting approach
                        - The model analyzes recent price trends and historical volatility
                        - Confidence interval expands over time to reflect increasing uncertainty
                        - Predictions should be used as one of many inputs for investment decisions
                        """)
                
            with tabs[2]:
                st.subheader("Portfolio Optimization")
                
                # Run optimization with better error handling
                try:
                    weights, portfolio_return, portfolio_risk, sharpe_ratio = optimize_portfolio(returns, risk_free_rate)
                    
                    # Display optimization results
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Expected Annual Return", f"{portfolio_return*100:.2f}%")
                    col2.metric("Annual Risk", f"{portfolio_risk*100:.2f}%")
                    col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                    
                    # Display optimized weights
                    st.subheader("Optimized Portfolio Weights")
                    
                    # Calculate current weights from portfolio
                    current_weights = [(inv / total_investments) for inv in investments]
                    
                    # Create DataFrame for weights
                    weights_df = pd.DataFrame({
                        'Ticker': weights.index,
                        'Current Weight (%)': [cw * 100 for cw in current_weights],
                        'Optimized Weight (%)': [weights[ticker] * 100 for ticker in weights.index],
                    })
                    
                    # Display as bar chart
                    fig = go.Figure()
                    
                    # Add current weights
                    fig.add_trace(go.Bar(
                        x=weights.index,
                        y=weights_df['Current Weight (%)'],
                        name='Current Weights',
                        marker_color='blue',
                        opacity=0.6
                    ))
                    
                    # Add optimized weights
                    fig.add_trace(go.Bar(
                        x=weights.index,
                        y=weights_df['Optimized Weight (%)'],
                        name='Optimized Weights',
                        marker_color='green',
                    ))
                    
                    fig.update_layout(
                        title='Portfolio Allocation Comparison',
                        xaxis_title='Stock',
                        yaxis_title='Weight (%)',
                        template='plotly_white',
                        barmode='group'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create series for current weights
                    current_weights_series = pd.Series(current_weights, index=tickers)
                    
                    # Generate the efficient frontier
                    st.subheader("Efficient Frontier")
                    
                    # Number of portfolios (use fewer for better performance)
                    num_portfolios = 1000
                    
                    # Create random weights
                    np.random.seed(42)
                    weights_record = np.zeros((num_portfolios, len(returns.columns)))
                    returns_record = np.zeros(num_portfolios)
                    volatility_record = np.zeros(num_portfolios)
                    sharpe_record = np.zeros(num_portfolios)
                    
                    with st.spinner("Generating efficient frontier..."):
                        for i in range(num_portfolios):
                            # Generate random weights
                            weights_i = np.random.random(len(returns.columns))
                            weights_i = weights_i / np.sum(weights_i)
                            weights_record[i] = weights_i
                            
                            # Calculate returns
                            returns_i = np.sum(returns.mean() * weights_i) * 252
                            returns_record[i] = returns_i
                            
                            # Calculate volatility
                            volatility_i = np.sqrt(np.dot(weights_i.T, np.dot(returns.cov() * 252, weights_i)))
                            volatility_record[i] = volatility_i
                            
                            # Calculate Sharpe Ratio
                            sharpe_record[i] = (returns_i - risk_free_rate) / volatility_i
                    
                    # Create a scatter plot of the generated portfolios
                    fig = go.Figure()
                    
                    # Add random portfolios
                    fig.add_trace(go.Scatter(
                        x=volatility_record,
                        y=returns_record,
                        mode='markers',
                        marker=dict(
                            size=5,
                            color=sharpe_record,
                            colorscale='Viridis',
                            colorbar=dict(title='Sharpe Ratio'),
                            showscale=True
                        ),
                        name='Possible Portfolios'
                    ))
                    
                    # Add optimized portfolio
                    fig.add_trace(go.Scatter(
                        x=[portfolio_risk],
                        y=[portfolio_return],
                        mode='markers',
                        marker=dict(
                            size=15,
                            color='red',
                            symbol='star'
                        ),
                        name='Optimized Portfolio'
                    ))
                    
                    # Add current portfolio
                    current_return = np.sum(returns.mean() * current_weights_series) * 252
                    current_volatility = np.sqrt(np.dot(current_weights_series.T, np.dot(returns.cov() * 252, current_weights_series)))
                    
                    fig.add_trace(go.Scatter(
                        x=[current_volatility],
                        y=[current_return],
                        mode='markers',
                        marker=dict(
                            size=15,
                            color='blue',
                            symbol='diamond'
                        ),
                        name='Current Portfolio'
                    ))
                    
                    fig.update_layout(
                        title='Portfolio Efficient Frontier',
                        xaxis_title='Annual Volatility',
                        yaxis_title='Annual Return',
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Optimization explanation
                    st.info("""
                    **About Portfolio Optimization**
                    
                    - The optimizer maximizes the Sharpe ratio (risk-adjusted return)
                    - Each dot in the efficient frontier represents a possible portfolio allocation
                    - The optimized portfolio (red star) offers the best risk-adjusted return
                    - Consider transaction costs and taxes when rebalancing to the optimized weights
                    """)
                    
                except Exception as e:
                    st.error(f"Error in optimization: {str(e)}")
                    st.info("Try selecting different stocks or date ranges.")
                
        else:
            st.info("Please enter portfolio details in the sidebar to view analysis")
    else:
        st.info("Please enter stock tickers in the sidebar to get started")

if __name__ == "__main__":
    main()
