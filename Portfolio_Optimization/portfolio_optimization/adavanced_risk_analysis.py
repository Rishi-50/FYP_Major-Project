import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from arch import arch_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import sys
import os
# Import the Nifty 50 stocks
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nifty50_stocks import NIFTY_50_STOCKS

# Fetch historical data for given tickers
def fetch_data(tickers, start_date, end_date):
    return yf.download(tickers, start=start_date, end=end_date)['Close']

# Calculate daily returns
def calculate_returns(data):
    return data.pct_change().dropna()

# Debugged: Calculate portfolio weights using Risk Parity
def risk_parity_portfolio(returns):
    cov_matrix = returns.cov() * 252  # Annualized covariance
    n_assets = returns.shape[1]

    # Objective function for equal risk contribution
    def objective(weights):
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        marginal_risks = np.dot(cov_matrix, weights)
        risk_contribution = weights * marginal_risks
        return np.sum((risk_contribution / (portfolio_variance + 1e-8) - 1 / n_assets) ** 2)

    # Initial equal weights
    init_weights = np.ones(n_assets) / n_assets

    # Constraints and bounds
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Sum of weights = 1
    bounds = [(0, 1) for _ in range(n_assets)]  # No short selling

    # Optimize
    result = minimize(objective, init_weights, bounds=bounds, constraints=constraints)
    if not result.success:
        raise ValueError("Optimization failed!")

    weights = result.x
    return pd.DataFrame({"Asset": returns.columns, "Weight": weights})

def generate_random_portfolios(returns, num_portfolios=5000, risk_free_rate=0.03):
    """Generate random portfolios and calculate their risk, return, and Sharpe ratio."""
    mean_returns = returns.mean() * 252  # Annualized returns
    cov_matrix = returns.cov() * 252    # Annualized covariance
    n_assets = returns.shape[1]

    results = {"Returns": [], "Volatility": [], "Sharpe Ratio": [], "Weights": []}

    for _ in range(num_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)  # Normalize weights to sum to 1

        # Portfolio performance metrics
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

        # Save results
        results["Returns"].append(portfolio_return)
        results["Volatility"].append(portfolio_volatility)
        results["Sharpe Ratio"].append(sharpe_ratio)
        results["Weights"].append(weights)

    return pd.DataFrame(results)

# Apply PCA for dimensionality reduction
def apply_pca(returns, n_components=2):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(returns)
    explained_variance = pca.explained_variance_ratio_
    return pd.DataFrame(principal_components, columns=[f"PC{i+1}" for i in range(n_components)]), explained_variance



def forecast_volatility(data, asset, forecast_horizon=30, model_type="GARCH"):
    """
    Forecast volatility using ARCH or GARCH models.

    Parameters:
        data (pd.DataFrame): A DataFrame containing the asset price data.
        asset (str): The column name of the asset in the DataFrame.
        forecast_horizon (int): Number of days to forecast.
        model_type (str): "ARCH" or "GARCH" to specify the model.

    Returns:
        tuple: (forecasted_variances, evaluation_metrics)
    """
    try:
        # Calculate returns
        asset_returns = data[asset].pct_change().dropna() * 100  # Convert to percentage returns
        
        if len(asset_returns) < 60:  # Minimum required data points
            raise ValueError("Insufficient data points for forecasting")

        # Split data for evaluation
        train_size = int(len(asset_returns) * 0.8)
        train_returns = asset_returns[:train_size]
        test_returns = asset_returns[train_size:]

        # Choose and fit the model
        if model_type == "ARCH":
            model = arch_model(train_returns, vol="Arch", p=1, rescale=False)
        elif model_type == "GARCH":
            model = arch_model(train_returns, vol="Garch", p=1, q=1, rescale=False)
        else:
            raise ValueError("Invalid model type. Choose 'ARCH' or 'GARCH'.")

        # Fit the model with error handling
        try:
            res = model.fit(disp="off", show_warning=False, update_freq=0)
        except Exception as e:
            raise ValueError(f"Model fitting failed: {str(e)}")

        # Generate predictions for the test period
        predictions = []
        for i in range(len(test_returns)):
            try:
                forecast = res.forecast(horizon=1, start=train_size + i - 1)
                pred_var = np.sqrt(forecast.variance.values[-1, -1])
                predictions.append(pred_var)
            except Exception as e:
                continue
        
        predictions = np.array(predictions)

        # Ensure predictions array is not empty
        if len(predictions) == 0:
            raise ValueError("No predictions generated")

        # Calculate actual volatility (using rolling standard deviation of returns)
        actual_vol = test_returns.rolling(window=5).std().dropna().values
        pred_vol = predictions[:len(actual_vol)]  # Match lengths

        # Ensure we have matching lengths for evaluation
        min_len = min(len(actual_vol), len(pred_vol))
        if min_len == 0:
            raise ValueError("No overlapping data for evaluation")

        actual_vol = actual_vol[:min_len]
        pred_vol = pred_vol[:min_len]

        # Calculate evaluation metrics
        mse = np.mean((actual_vol - pred_vol) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual_vol - pred_vol))

        # Forecast future volatility
        try:
            future_forecast = res.forecast(horizon=forecast_horizon)
            if future_forecast.variance.empty:
                raise ValueError("Empty forecast generated")
                
            forecasted_variances = np.sqrt(future_forecast.variance.iloc[-1].values)
            
            # Ensure we have at least one valid forecast
            if len(forecasted_variances) == 0:
                raise ValueError("No valid forecasts generated")
                
        except Exception as e:
            # If forecasting fails, use the last known volatility as a simple forecast
            last_vol = test_returns.rolling(window=5).std().iloc[-1]
            forecasted_variances = np.array([last_vol] * forecast_horizon)

        evaluation_metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae
        }

        return forecasted_variances, evaluation_metrics

    except Exception as e:
        raise Exception(f"Error in volatility forecasting: {str(e)}")


# Detect anomalies using Isolation Forest for each asset
def detect_anomalies(returns):
    anomalies = []
    for asset in returns.columns:
        asset_returns = returns[asset].values.reshape(-1, 1)  # Reshape for IsolationForest
        model = IsolationForest(random_state=42)
        asset_anomalies = model.fit_predict(asset_returns)
        anomalies.append(asset_anomalies[-1])  # Get anomaly for the most recent data point
    return anomalies

# Simplified version with only Isolation Forest:
def enhanced_anomaly_detection(returns, n_periods=30, contamination=0.05):
    """
    Perform anomaly detection on return series using Isolation Forest.
    
    Parameters:
    -----------
    returns : pandas.DataFrame
        DataFrame containing asset returns with assets as columns
    n_periods : int
        Number of most recent periods to analyze for anomalies
    contamination : float
        Expected proportion of outliers in the data
        
    Returns:
    --------
    dict
        Dictionary containing anomaly results
    """
    # Make sure we have enough data
    if len(returns) < n_periods:
        n_periods = len(returns)
    
    # Only use the most recent n_periods for detection
    recent_returns = returns.iloc[-n_periods:]
    
    results = {
        'asset_names': returns.columns.tolist(),
        'anomalies': {},  # Will store detection results 
        'anomaly_scores': {},  # Will store anomaly scores
        'threshold_values': {},  # Will store threshold values
        'historical_anomalies': {}  # Will store historical anomalies (for time series view)
    }
    
    # Run Isolation Forest detection
    iso_results = []
    iso_scores = []
    iso_historical = []
    
    for asset in returns.columns:
        # Get asset returns as array
        asset_returns = recent_returns[asset].values.reshape(-1, 1)
        full_history = returns[asset].values.reshape(-1, 1)
        
        # Train Isolation Forest model
        iso_model = IsolationForest(
            n_estimators=100,
            max_samples='auto',
            contamination=contamination,
            random_state=42
        )
        
        # Fit and predict
        iso_model.fit(asset_returns)
        anomaly = iso_model.predict(asset_returns[-1:])  # Most recent point
        
        # Get anomaly scores (-1 for anomalies, 1 for normal points by default)
        # Convert to a 0-100 scale where higher values = more anomalous
        anomaly_score = -1 * iso_model.score_samples(asset_returns[-1:])
        anomaly_score = (anomaly_score[0] + 0.5) * 100  # Scale to 0-100
        
        # Get historical anomalies for time series visualization
        historical_predictions = iso_model.predict(full_history)
        
        iso_results.append(anomaly[0])
        iso_scores.append(anomaly_score)
        iso_historical.append(historical_predictions)
    
    # Store results
    results['anomalies'] = {
        'isolation_forest': iso_results
    }
    
    results['anomaly_scores'] = {
        'isolation_forest': iso_scores
    }
    
    results['threshold_values'] = {
        'isolation_forest': contamination * 100  # As percentage
    }
    
    results['historical_anomalies'] = {
        'isolation_forest': iso_historical,
        # Dates for the historical data
        'dates': returns.index.tolist()
    }
    
    return results

def plot_anomaly_detection_results(results, returns):
    """
    Create interactive visualizations for Isolation Forest anomaly detection results.
    
    Parameters:
    -----------
    results : dict
        Results from enhanced_anomaly_detection function
    returns : pandas.DataFrame
        Original returns DataFrame
        
    Returns:
    --------
    tuple
        (summary_fig, status_fig, timeseries_fig) - Plotly figures
    """
    assets = results['asset_names']
    
    # 1. Summary bar chart - Anomaly scores
    summary_fig = go.Figure()
    
    # Get Isolation Forest scores
    scores = results['anomaly_scores']['isolation_forest']
    
    # Add bars for scores
    summary_fig.add_trace(go.Bar(
        x=assets,
        y=scores,
        name="Anomaly Score",
        marker_color='rgba(31, 119, 180, 0.7)'
    ))
    
    # Add threshold line for score
    threshold = results['threshold_values']['isolation_forest'] * 10  # Adjust to same scale
    summary_fig.add_shape(
        type="line",
        x0=-0.5, 
        x1=len(assets)-0.5,
        y0=threshold,  # Threshold for anomaly
        y1=threshold,
        line=dict(color="red", width=2, dash="dash")
    )
    
    summary_fig.add_annotation(
        x=len(assets)-1,
        y=threshold,
        text="Anomaly Threshold",
        showarrow=False,
        yshift=10
    )
    
    # Update layout
    summary_fig.update_layout(
        title="Isolation Forest Anomaly Scores",
        xaxis_title="Asset",
        yaxis_title="Anomaly Score (Higher = More Anomalous)",
        template="plotly_white"
    )
    
    # 2. Status visualization
    iso_results = results['anomalies']['isolation_forest']
    
    status_data = []
    for i, asset in enumerate(assets):
        status_data.append({
            "Asset": asset,
            "Status": "Anomaly" if iso_results[i] == -1 else "Normal",
            "Score": scores[i]
        })
    
    status_df = pd.DataFrame(status_data)
    
    # Create colormap where red = anomaly, green = normal
    colors = ['green' if status == "Normal" else 'red' 
              for status in status_df['Status']]
    
    status_fig = go.Figure()
    
    status_fig.add_trace(go.Bar(
        x=status_df['Asset'],
        y=[1] * len(status_df),
        marker_color=colors,
        text=status_df['Status'],
        hovertemplate="<b>%{x}</b><br>" +
                      "Status: %{text}<br>" +
                      "Anomaly Score: %{customdata:.1f}<extra></extra>",
        customdata=status_df['Score'],
        name="",
        showlegend=False
    ))
    
    status_fig.update_layout(
        title="Isolation Forest Anomaly Detection Results",
        xaxis_title="Asset",
        yaxis_title="",
        template="plotly_white",
        yaxis=dict(showticklabels=False)
    )
    
    # 3. Time series with anomalies for an example asset (first asset)
    asset_idx = 0
    asset_name = assets[asset_idx]
    
    # Get historical returns
    hist_returns = returns[asset_name].iloc[-min(100, len(returns)):]
    
    # Get historical anomalies for Isolation Forest
    hist_anomalies = results['historical_anomalies']['isolation_forest'][asset_idx]
    
    # Prepare data for time series plot
    if len(hist_anomalies) > len(hist_returns):
        hist_anomalies = hist_anomalies[-len(hist_returns):]
        
    anomaly_indices = np.where(hist_anomalies == -1)[0]
    
    if len(anomaly_indices) > 0:
        anomaly_returns = hist_returns.iloc[anomaly_indices]
    else:
        anomaly_returns = pd.Series()
    
    # Create time series figure
    timeseries_fig = go.Figure()
    
    # Add returns line
    timeseries_fig.add_trace(go.Scatter(
        x=hist_returns.index,
        y=hist_returns.values,
        mode='lines',
        name='Returns',
        line=dict(color='blue')
    ))
    
    # Add anomaly points if any exist
    if len(anomaly_returns) > 0:
        timeseries_fig.add_trace(go.Scatter(
            x=anomaly_returns.index,
            y=anomaly_returns.values,
            mode='markers',
            name='Anomalies',
            marker=dict(
                size=10,
                color='red',
                symbol='circle-open'
            )
        ))
    
    # Update layout
    timeseries_fig.update_layout(
        title=f"Historical Returns with Anomalies: {asset_name}",
        xaxis_title="Date",
        yaxis_title="Return",
        template="plotly_white"
    )
    
    return summary_fig, status_fig, timeseries_fig

def plot_pca_results(pca_data, explained_variance):
    """
    Creates an interactive PCA scatter plot.

    Parameters:
    - pca_data: DataFrame with 'PC1', 'PC2', and optional labels for tooltips.
    - explained_variance: Percentage of variance explained by the components.

    Returns:
    - fig: Plotly interactive figure.
    """
    # Ensure 'Label' column exists for tooltips (if not, create a default)
    if 'Label' not in pca_data.columns:
        pca_data['Label'] = pca_data.index

    # Create a scatter plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=pca_data['PC1'],
        y=pca_data['PC2'],
        mode='markers',
        marker=dict(
            size=10,
            color=pca_data['PC1'],  # Color by PC1 for better visual effect
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="PC1 Value")
        ),
        text=pca_data['Label'],  # Add labels to tooltips
        hovertemplate=(
            "<b>Label:</b> %{text}<br>"
            "<b>PC1:</b> %{x:.2f}<br>"
            "<b>PC2:</b> %{y:.2f}<br>"
            "<extra></extra>"
        ),
        name='PCA Components'
    ))

    # Add explained variance to the title
    fig.update_layout(
        title=f"PCA Results<br> Explained Variance: PC1: {explained_variance[0]*100:.2f}%, PC2: {explained_variance[1]*100:.2f}%",
        xaxis=dict(title='Principal Component 1'),
        yaxis=dict(title='Principal Component 2'),
        template='plotly_white',
        hovermode='closest',
    )

    # Add interactivity: Show a reference line at the origin
    fig.add_trace(go.Scatter(
        x=[0, 0],
        y=[pca_data['PC2'].min(), pca_data['PC2'].max()],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=[pca_data['PC1'].min(), pca_data['PC1'].max()],
        y=[0, 0],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        showlegend=False
    ))

    return fig




def plot_efficient_frontier_with_risk_parity(returns, risk_free_rate=0.04):
    """Plot efficient frontier with random portfolios and Risk Parity Portfolio."""
    random_portfolios = generate_random_portfolios(returns, risk_free_rate=risk_free_rate)
    risk_parity_weights = risk_parity_portfolio(returns)
    mean_returns = returns.mean() * 252
    risk_parity_return = np.dot(mean_returns, risk_parity_weights["Weight"])
    risk_parity_volatility = np.sqrt(
        np.dot(risk_parity_weights["Weight"].T, np.dot(returns.cov() * 252, risk_parity_weights["Weight"]))
    )

    # Create the plot
    fig = go.Figure()

    # Add random portfolios
    fig.add_trace(go.Scatter(
        x=random_portfolios["Volatility"],
        y=random_portfolios["Returns"],
        mode='markers',
        marker=dict(
            size=6,
            color=random_portfolios["Sharpe Ratio"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Sharpe Ratio"),
        ),
        name="Random Portfolios"
    ))

    # Add the Risk Parity Portfolio
    fig.add_trace(go.Scatter(
        x=[risk_parity_volatility],
        y=[risk_parity_return],
        mode='markers',
        marker=dict(color='red', size=12),
        name='Risk Parity Portfolio'
    ))

    # Update layout
    fig.update_layout(
        title="Efficient Frontier with Risk Parity Portfolio",
        xaxis_title="Volatility (Risk)",
        yaxis_title="Return",
        template='plotly_white'
    )

    return fig


# Train a regression model to predict future returns
def train_return_predictor(returns, lookback=5, forecast_horizon=1):
    X, y = [], []
    for i in range(lookback, len(returns) - forecast_horizon):
        X.append(returns.iloc[i-lookback:i].values.flatten())
        y.append(returns.iloc[i + forecast_horizon].values.flatten())
    X, y = np.array(X), np.array(y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return model, mse

def train_return_predictor_linear(returns, lookback=5, forecast_horizon=1):
    X, y = [], []
    for i in range(lookback, len(returns) - forecast_horizon):
        X.append(returns.iloc[i-lookback:i].values.flatten())
        y.append(returns.iloc[i + forecast_horizon].values.flatten())
    X, y = np.array(X), np.array(y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return model, mse

def train_return_predictor_lstm(returns, lookback=5, forecast_horizon=1):
    """
    Train an LSTM model to predict returns.
    
    Parameters:
        returns (pd.DataFrame): DataFrame containing asset returns
        lookback (int): Number of past days to use for prediction
        forecast_horizon (int): Number of days to forecast ahead
        
    Returns:
        model: Trained LSTM model
        mse: Mean squared error on test set
    """
    # Prepare data for LSTM
    X, y = [], []
    for i in range(len(returns) - lookback - forecast_horizon + 1):
        X.append(returns.iloc[i:(i + lookback)].values)
        y.append(returns.iloc[i + lookback + forecast_horizon - 1].values)
    
    X = np.array(X)
    y = np.array(y)
    
    # Split data into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, returns.shape[1])),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(returns.shape[1])
    ])
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # Train model
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        verbose=0
    )
    
    # Evaluate model
    mse = mean_squared_error(y_test, model.predict(X_test))
    
    return model, mse

# Main app
def main():
    st.title("Advanced Portfolio Optimization and Risk Analysis")
    st.write("Explore PCA, time-series forecasting, risk parity, and anomaly detection methods for portfolio analysis.")

    # Sidebar inputs
    st.sidebar.header("User Input")
    
    # Replace text input with multiselect dropdown
    st.sidebar.subheader("Select Stocks (Nifty 50)")
    default_stocks = ["Reliance Industries", "TCS", "HDFC Bank", "Infosys", "ITC"]
    selected_stock_names = st.sidebar.multiselect(
        "Choose stocks for your portfolio:",
        options=list(NIFTY_50_STOCKS.keys()),
        default=default_stocks,
        help="Select multiple stocks from Nifty 50 to create your portfolio"
    )
    
    # Convert selected stock names to their tickers
    tickers = [NIFTY_50_STOCKS[stock] for stock in selected_stock_names] if selected_stock_names else []
    
    # Show selected tickers
    if tickers:
        st.sidebar.caption(f"Selected tickers: {', '.join(tickers)}")
        
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))
    risk_free_rate = st.sidebar.number_input("Risk-Free Rate (in decimal form):", value=0.03, step=0.01)

    if tickers:
        data = fetch_data(tickers, start_date, end_date)
        returns = calculate_returns(data)
        
        # Tabs for different functionalities
        tab1, tab2, tab3, tab4 = st.tabs([
            "PCA Analysis", "Risk Parity", "Volatility and Return Prediction", "Anomaly Detection"
        ])

        with tab1:
            st.subheader("PCA Analysis")
            pca_data, explained_variance = apply_pca(returns)
            st.dataframe(pca_data)
            pca_fig = plot_pca_results(pca_data, explained_variance)
            st.plotly_chart(pca_fig)

            st.markdown("""
            ### What is PCA?
            Principal Component Analysis (PCA) reduces the dimensionality of data while retaining most of its variance. 
            It transforms correlated variables into a smaller number of uncorrelated variables called principal components.

            ### How is PCA Helpful?
            - **Understanding Drivers of Variance**: PCA identifies the main factors driving portfolio returns.
            - **Risk Management**: By focusing on major components, you can better understand where risks are concentrated.
            - **Data Simplification**: Reduces the complexity of large datasets, making analysis more manageable.

            ### Interpretation:
            - PC1 explains the majority of variance in the portfolio, showing where most of the movement in asset prices originates.
            - Lower components like PC2 and PC3 explain residual variance, which may relate to secondary factors or noise.

            ### **Example Use Case**:
            1. **Financial Data Analysis**:
            - In financial markets, PCA can be applied to analyze correlations between assets.
            - For example, returns of multiple stocks can be reduced to a few principal components, where PC1 may capture the overall market movement, and subsequent components may reflect sector-specific trends.

            2. **Customer Segmentation**:
            - In marketing, PCA helps reduce customer behavior data into fewer dimensions, enabling clustering and segmentation analysis.

            ---

            ### **Applications of PCA**:
            - **Exploratory Data Analysis (EDA)**: Identify patterns and clusters in datasets.
            - **Preprocessing**: Reduce dimensionality before applying machine learning algorithms.
            - **Visualization**: Plot high-dimensional data in 2D or 3D space for easier interpretation.
            - **Feature Engineering**: Extract meaningful features for modeling.

            ---

            ### **Summary of PCA Analysis**:
            - In this analysis:
            - The data is transformed into principal components, with **Principal Component 1 (PC1)** and **Principal Component 2 (PC2)** capturing the most variance.
            - The **explained variance** metric indicates how much information is preserved in the reduced dimensions.
            - The first two components explain the majority of the variance, allowing for a simplified yet informative representation of the dataset.
            """)

        with tab2:
            st.subheader("Risk Parity Portfolio")
            risk_parity_weights = risk_parity_portfolio(returns)
            st.dataframe(risk_parity_weights)

            st.subheader("Efficient Frontier with Risk Parity Portfolio")
            efficient_frontier_fig = plot_efficient_frontier_with_risk_parity(returns, risk_free_rate)
            st.plotly_chart(efficient_frontier_fig)

            st.markdown("""
            ### What is Risk Parity?
            Risk Parity balances the contribution of risk from each asset in the portfolio. 
            It ensures no single asset dominates the portfolio's overall risk.

            ### How is Risk Parity Helpful?
            - **Diversification**: Creates a portfolio where risk is spread evenly across assets.
            - **Risk Management**: Mitigates the impact of highly volatile assets on the portfolio.
            - **Stability**: Often leads to more stable returns over time compared to traditional allocation methods.

            ### Interpretation:
            - The risk parity weights indicate the proportion of investment in each asset to equalize risk contribution.
            - The efficient frontier shows the risk-return tradeoff and how the risk parity portfolio compares to other allocations.
            """)

        
        
        
        with tab3:
            st.subheader("Volatility and Return Prediction")

            # Option to choose between Volatility and Return Prediction
            prediction_type = st.radio("Select Prediction Type:", options=["Volatility Forecasting", "Return Prediction"], index=0)

            if prediction_type == "Volatility Forecasting":
                # Volatility Forecasting
                st.markdown("### Volatility Forecasting")
                model_type = st.radio("Select Volatility Model:", options=["ARCH", "GARCH"], index=1)
                forecast_results = {}

                for asset in tickers:
                    try:
                        forecast, metrics = forecast_volatility(data, asset, forecast_horizon=10, model_type=model_type)
                        forecast_results[asset] = {
                            "Forecasted Volatility": forecast,
                            "MSE": metrics['MSE'],
                            "RMSE": metrics['RMSE'],
                            "MAE": metrics['MAE']
                        }
                    except Exception as e:
                        st.warning(f"Failed to forecast for {asset}: {e}")

                st.dataframe(pd.DataFrame(forecast_results).T.rename(columns={0: "Forecasted Volatility"}))

                # Interpretation of Volatility Forecast:
                st.markdown("#### Volatility Forecast Interpretation:")
                st.write(
                    """
                    - **Volatility** is a statistical measure that describes the extent to which the price of an asset fluctuates over time. In simple terms, it represents the level of uncertainty or risk associated with the asset's price movement. 
                    - High **volatility** implies that the asset's price is likely to experience large fluctuations in a short period, which can result in significant gains or losses. Conversely, low volatility means the asset's price changes are smaller and more predictable, typically implying a lower level of risk.
                    
                    - **Forecasted volatility** is the estimated measure of an asset's future price fluctuations, typically derived from past price data. It provides an outlook on how much the asset's price could vary over the forecast period, which is particularly useful for risk management and investment strategy. 
                    - Forecasting volatility is crucial because it helps investors anticipate the likelihood of significant price swings and assess whether the asset's risk profile aligns with their investment objectives.

                    - **Interpretation of high forecasted volatility:**
                        - If the forecasted volatility is high, it indicates that there is a higher likelihood of large price movements in the forecast horizon (e.g., 10 days). This suggests that the asset may experience significant price changes, either upwards or downwards.
                        - High volatility can represent both risk and opportunity: it could lead to substantial returns, but it also increases the potential for substantial losses. Investors seeking greater profits may find high volatility attractive, but those with a lower risk tolerance may want to avoid such assets or hedge their positions.

                    - **Interpretation of low forecasted volatility:**
                        - On the other hand, if forecasted volatility is low, it signals that the asset's price is expected to remain relatively stable over the forecast horizon.
                        - Low volatility is typically more attractive to risk-averse investors who prefer to invest in assets with less price fluctuation. Such assets can offer more predictable returns, making them ideal for conservative or long-term investors who prioritize stability over high returns.
                        
                    - Understanding forecasted volatility helps in shaping investment decisions. For instance, an investor might choose high-volatility assets for short-term speculative strategies, whereas low-volatility assets might be favored for long-term portfolio stability.
                    """
                )

            elif prediction_type == "Return Prediction":
                # Return Prediction
                st.markdown("### Return Prediction")

                # Options to choose between different models for return prediction
                return_model_type = st.radio("Select Return Prediction Model:", options=["Linear Regression", "Random Forest", "LSTM"], index=0)

                # Forecast horizon and lookback options for prediction
                forecast_horizon = st.slider("Forecast Horizon (Days):", 1, 30, 5)
                lookback = st.slider("Lookback Period (Days):", 1, 30, 5)

                try:
                    if return_model_type == "Linear Regression":
                        model, mse = train_return_predictor_linear(returns, lookback=lookback, forecast_horizon=forecast_horizon)
                    elif return_model_type == "Random Forest":
                        model, mse = train_return_predictor(returns, lookback=lookback, forecast_horizon=forecast_horizon)
                    elif return_model_type == "LSTM":
                        model, mse = train_return_predictor_lstm(returns, lookback=lookback, forecast_horizon=forecast_horizon)

                    st.write(f"Model trained with Mean Squared Error (MSE): {mse:.4f}")

                    # Predict returns for the next period
                    if return_model_type == "LSTM":
                        # For LSTM, we need to reshape the input differently
                        latest_data = returns.iloc[-lookback:].values.reshape(1, lookback, returns.shape[1])
                        predicted_returns = model.predict(latest_data).flatten()
                    else:
                        # For other models
                        latest_data = returns.iloc[-lookback:].values.flatten().reshape(1, -1)
                        predicted_returns = model.predict(latest_data).flatten()

                    # Display predictions
                    st.write("### Predicted Returns for Each Asset")
                    predicted_df = pd.DataFrame(
                        {"Asset": returns.columns, "Predicted Return": predicted_returns}
                    )
                    st.dataframe(predicted_df)

                    # Create line graph for all stocks
                    st.write("### Line Graph of Predicted Returns")
                    
                    # Allow user to adjust number of periods to forecast
                    actual_periods = st.slider("Number of periods to visualize:", 1, min(forecast_horizon, 30), min(5, forecast_horizon))
                    
                    try:
                        # Initialize progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        all_predictions = []
                        
                        for period in range(actual_periods):
                            # Update progress
                            progress = int((period + 1) / actual_periods * 100)
                            progress_bar.progress(progress)
                            status_text.text(f"Generating prediction {period + 1}/{actual_periods}...")
                            
                            try:
                                if return_model_type == "LSTM":
                                    if period == 0:
                                        # First prediction uses actual data
                                        pred_input = returns.iloc[-lookback:].values.reshape(1, lookback, returns.shape[1])
                                    else:
                                        # Subsequent predictions use rolling window with previous predictions
                                        # Limit recursive predictions to avoid error amplification
                                        if period < 10:  # Only use previous predictions for a limited number of steps
                                            new_data = np.vstack([pred_input[0, 1:, :], [prev_pred]])
                                            pred_input = new_data.reshape(1, lookback, returns.shape[1])
                                        else:
                                            # For longer horizons, add small random noise to the last prediction
                                            # to avoid unrealistic projections
                                            noise = np.random.normal(0, 0.001, size=prev_pred.shape)
                                            prev_pred = prev_pred + noise
                                    
                                    prev_pred = model.predict(pred_input).flatten()
                                    
                                    # Limit extreme predictions
                                    prev_pred = np.clip(prev_pred, -0.1, 0.1)  # Limit to reasonable daily returns
                                    
                                    all_predictions.append(prev_pred)
                                else:
                                    if period == 0:
                                        # First prediction uses actual data
                                        pred_input = returns.iloc[-lookback:].values.flatten().reshape(1, -1)
                                    else:
                                        # Create a rolling window for prediction with safeguards
                                        if period < 10:  # Only use previous predictions for a limited number of steps
                                            window_size = len(returns.columns)
                                            new_data = np.append(pred_input[0, window_size:], prev_pred)
                                            pred_input = new_data.reshape(1, -1)
                                        else:
                                            # For longer horizons, add small random noise to the last prediction
                                            noise = np.random.normal(0, 0.001, size=prev_pred.shape)
                                            prev_pred = prev_pred + noise
                                    
                                    prev_pred = model.predict(pred_input).flatten()
                                    
                                    # Limit extreme predictions
                                    prev_pred = np.clip(prev_pred, -0.1, 0.1)  # Limit to reasonable daily returns
                                    
                                    all_predictions.append(prev_pred)
                            except Exception as e:
                                st.warning(f"Error in prediction for period {period+1}: {str(e)}")
                                # If we encounter an error, use the last successful prediction or zeros
                                if all_predictions:
                                    all_predictions.append(all_predictions[-1])  # Use last prediction
                                else:
                                    all_predictions.append(np.zeros(len(returns.columns)))  # Use zeros
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Check if we have any predictions
                        if not all_predictions:
                            raise ValueError("Failed to generate predictions for any period")
                        
                        # Create a DataFrame for plotting
                        prediction_df = pd.DataFrame(all_predictions, columns=returns.columns)
                        
                        # Calculate cumulative returns for better visualization
                        cumulative_option = st.checkbox("Show cumulative returns", value=False)
                        
                        if cumulative_option:
                            for col in prediction_df.columns:
                                prediction_df[col] = (1 + prediction_df[col]).cumprod() - 1
                            y_axis_title = "Cumulative Predicted Return"
                        else:
                            y_axis_title = "Predicted Return per Period"
                        
                        # Plot line graph for predictions
                        fig_line = go.Figure()
                        
                        for asset in returns.columns:
                            fig_line.add_trace(go.Scatter(
                                y=prediction_df[asset],
                                x=list(range(1, len(prediction_df) + 1)),
                                mode='lines+markers',
                                name=asset
                            ))
                        
                        fig_line.update_layout(
                            title=f"Predicted Returns Over Next {len(prediction_df)} Periods",
                            xaxis_title="Future Period",
                            yaxis_title=y_axis_title,
                            yaxis=dict(tickformat='.2%'),
                            template="plotly_white",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            height=500
                        )
                        
                        # Add zero reference line
                        fig_line.add_hline(
                            y=0, 
                            line_dash="dash", 
                            line_color="gray",
                            annotation_text="Zero Return",
                            annotation_position="bottom right"
                        )
                        
                        st.plotly_chart(fig_line, use_container_width=True)
                        
                        # Add download button for prediction data
                        csv = prediction_df.to_csv(index=False)
                        st.download_button(
                            label="Download Prediction Data as CSV",
                            data=csv,
                            file_name="stock_return_predictions.csv",
                            mime="text/csv",
                        )
                        
                        # Add note about the nature of predictions
                        st.info("""
                        **Note:** Multi-period forecasts become less reliable as the forecast horizon extends. 
                        The predictions shown here should be used as general directional guidance rather than precise forecasts.
                        """)
                        
                    except Exception as e:
                        st.error(f"Error generating multi-period predictions: {str(e)}")
                        st.info("Displaying single-period prediction only.")

                    # Plot bar chart for single-period prediction
                    fig_bar = go.Figure()
                    fig_bar.add_trace(go.Bar(x=predicted_df["Asset"], y=predicted_df["Predicted Return"], name="Predicted Returns"))
                    fig_bar.update_layout(
                        title="Predicted Returns for Next Period",
                        xaxis_title="Asset",
                        yaxis_title="Predicted Return",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_bar)

                    st.markdown("#### Predicted Return Interpretation:")
                    st.write(
                    """
                    - **Predicted returns** refer to the model's best estimate of how each asset is expected to perform during the forecast period (e.g., the next period, which could be one day, a week, or a month). These predictions are generated using historical return data and statistical or machine learning models, and they provide a glimpse into the potential future performance of each asset.
                    
                    - **Positive predicted returns** suggest that the asset is expected to increase in value during the forecast horizon. For example, if the model predicts a positive return for a stock, it indicates that the stock's price is expected to go up. Positive predicted returns are generally viewed as favorable for investors looking for capital appreciation or gains.
                    
                    - **Negative predicted returns**, on the other hand, suggest that the asset is expected to decline in value over the forecast period. This indicates potential losses and could signal that the asset might be less attractive for investment in the near term. Investors might consider reducing their exposure to such assets or looking for hedging opportunities.
                    
                    - **Magnitude of predicted returns**: The size of the predicted return gives investors an idea of how strongly the model expects the asset to move. A larger positive or negative predicted return implies a larger expected change in the asset's price, while a smaller return indicates a more stable, less volatile price movement.
                        - For example, a predicted return of +10% indicates a substantial upward movement, while +1% indicates only a modest increase. Similarly, -10% signals a substantial decrease, while -1% indicates a small decline.
                        
                    - **How to use predicted returns**: Investors can use the predicted returns to inform their portfolio decisions. If a certain asset is predicted to have a high positive return, an investor might choose to allocate more capital to it, betting on future gains. Conversely, if the predicted return is negative or lower than other assets, the investor might choose to reduce their exposure to that asset or avoid it entirely.
                        - For example, if Asset A is predicted to have a high positive return and Asset B has a negative predicted return, an investor might decide to increase the weight of Asset A in their portfolio and reduce or eliminate their holdings in Asset B.
                        
                    - **Forecast horizon and volatility**: It's crucial to consider the **forecast horizon** (the time frame for which the returns are predicted) alongside the predicted returns. For instance, a short-term prediction may show large returns, but those returns could be very volatile, while a long-term prediction might show more stable but moderate returns.
                    
                    - **Volatility and risk**: Predicted returns should not be viewed in isolation. Volatility predictions (which indicate the risk or price fluctuations of the asset) should also be taken into account. An asset with a high predicted return but also high forecasted volatility might carry substantial risk, while an asset with a moderate predicted return and low volatility might offer more predictable and safer returns.
                        - For example, an investor might be willing to take on high volatility if they believe the predicted return justifies the potential risk, while others may prefer more stable returns with lower risk, even if the returns are not as high.
                        
                    - **Strategic investment decisions**: By combining predicted returns with other information like volatility and the investor's risk tolerance, the model's return forecasts can serve as a valuable tool for portfolio management. The goal is to balance risk and reward, allocating capital to assets that align with the investor's objectives.
                    """
                    )
                except Exception as e:
                    st.error(f"An error occurred during return prediction: {e}")

        
        with tab4:
            st.subheader("Anomaly Detection")
            anomalies = detect_anomalies(returns)
            anomaly_df = pd.DataFrame({'Asset': returns.columns, 'Anomaly': anomalies})
            st.dataframe(anomaly_df)

            anomaly_fig = go.Figure()
            anomaly_fig.add_trace(go.Bar(x=anomaly_df['Asset'], y=anomaly_df['Anomaly'], name='Anomalies'))
            st.plotly_chart(anomaly_fig)

            st.markdown("""
            ### What is Anomaly Detection?
            Anomaly detection identifies unusual behavior or outliers in data, often indicative of significant events or data issues.

            ### How is Anomaly Detection Helpful?
            - **Risk Monitoring**: Flags unusual returns that may signify major market events or errors.
            - **Fraud Detection**: Identifies suspicious activity in trading or portfolio data.
            - **Portfolio Review**: Helps refine strategies by understanding outlier behavior.

            ### Interpretation:
            - Anomalies flagged as `-1` indicate potential issues or significant deviations in returns.
            - Regular review of anomalies can provide early warnings of market instability.
            """)
    else:
        st.info("Please select at least one stock in the sidebar to begin analysis.")



if __name__ == "__main__":
    main()
