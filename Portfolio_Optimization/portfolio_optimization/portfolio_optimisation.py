import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
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


def optimize_portfolio(returns, risk_free_rate):
    """
    Optimizes portfolio weights using mean-variance optimization to maximize the Sharpe ratio.
    
    :param returns: Pandas DataFrame of asset returns.
    :param risk_free_rate: Risk-free rate for Sharpe ratio calculation.
    :return: Optimal asset weights.
    """
    mu = returns.mean() * 252  # Annualized expected returns
    cov_matrix = returns.cov() * 252  # Annualized covariance matrix
    num_assets = len(mu)

    def sharpe_ratio(weights):
        portfolio_return = np.dot(weights, mu)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(portfolio_return - risk_free_rate) / portfolio_vol  # Negative for minimization

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
    bounds = tuple((0, 1) for _ in range(num_assets))  # No short-selling
    initial_weights = np.ones(num_assets) / num_assets  # Equal allocation start

    result = minimize(sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x , sharpe_ratio(result.x)

'''# Optimize portfolio weights using mean-variance optimization
def optimize_portfolio(returns):
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

    num_assets = len(returns.columns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = [1 / num_assets] * num_assets

    result = minimize(portfolio_volatility, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x '''

# Calculate Value at Risk (VaR) and Conditional VaR (CVaR)
def calculate_var_cvar(returns, weights, confidence_level=0.95):
    portfolio_returns = returns.dot(weights)
    VaR = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
    CVaR = portfolio_returns[portfolio_returns <= VaR].mean()
    return VaR, CVaR, portfolio_returns

# Monte Carlo simulation for portfolio optimization
def monte_carlo_simulation(returns, num_portfolios=5000, risk_free_rate=0.03):
    num_assets = len(returns.columns)
    portfolio_metrics = []
    for _ in range(num_portfolios):
        weights = np.random.dirichlet(np.ones(num_assets), size=1).flatten()
        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        portfolio_metrics.append((weights, portfolio_return, portfolio_volatility, sharpe_ratio))

    portfolio_df = pd.DataFrame(portfolio_metrics, columns=['Weights', 'Return', 'Volatility', 'Sharpe'])
    optimal_portfolio = portfolio_df.iloc[portfolio_df['Sharpe'].idxmax()]
    min_vol_portfolio = portfolio_df.iloc[portfolio_df['Volatility'].idxmin()]
    return portfolio_df, optimal_portfolio, min_vol_portfolio


def plot_efficient_frontier(returns, risk_free_rate=0.03, num_portfolios=5000):
    mean_returns = returns.mean() * 252  # annualized mean returns
    cov_matrix = returns.cov() * 252  # annualized covariance matrix
    num_portfolios = 10000  # Number of random portfolios
    results = np.zeros((4, num_portfolios), dtype=object)  # Store results (Returns, Volatility, Sharpe Ratio, Weights)

    for i in range(num_portfolios):
        # Random portfolio weights
        weights = np.random.random(len(returns.columns))
        weights /= np.sum(weights)
        
        # Portfolio returns and volatility
        portfolio_return = np.sum(weights * mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Store results
        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_volatility  # Sharpe Ratio
        results[3, i] = weights  # Store weights as a list/array

    # Extract the portfolio with the maximum Sharpe ratio and the minimum volatility
    max_sharpe_idx = np.argmax(results[2])
    min_vol_idx = np.argmin(results[1])

    # Plot the Efficient Frontier
    frontier_fig = go.Figure()

    # Efficient Frontier
    frontier_fig.add_trace(go.Scatter(
        x=results[1],  # Volatility
        y=results[0],  # Return
        mode='markers',
        marker=dict(color=results[2], colorscale='Viridis', colorbar=dict(title='Sharpe Ratio')),
        name='Random Portfolios',
        text=[f"Weights: {np.round(results[3][i], 2)}" for i in range(num_portfolios)],  # Fixed index access
        hovertemplate='<b>Volatility:</b> %{x:.2%}<br><b>Return:</b> %{y:.2%}<br>%{text}'
    ))

    # Plot the Maximum Sharpe Ratio Portfolio
    frontier_fig.add_trace(go.Scatter(
        x=[results[1, max_sharpe_idx]],
        y=[results[0, max_sharpe_idx]],
        mode='markers',
        marker=dict(color='red', size=12),
        name='Maximum Sharpe Ratio Portfolio'
    ))

    # Plot the Minimum Volatility Portfolio
    frontier_fig.add_trace(go.Scatter(
        x=[results[1, min_vol_idx]],
        y=[results[0, min_vol_idx]],
        mode='markers',
        marker=dict(color='blue', size=12),
        name='Minimum Volatility Portfolio'
    ))

    frontier_fig.update_layout(
        title='Efficient Frontier',
        xaxis_title='Volatility (Risk)',
        yaxis_title='Return',
        template='plotly_white',
        showlegend=True
    )
    return frontier_fig

def asset_clustering(returns, num_clusters=3):
    # Calculate mean returns and volatilities for each asset
    mean_returns = returns.mean() * 252  # Annualized returns
    volatilities = returns.std() * np.sqrt(252)  # Annualized volatilities
    
    # Create a DataFrame of returns and volatilities
    asset_metrics = pd.DataFrame({
        'Return': mean_returns,
        'Volatility': volatilities
    })
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    asset_metrics['Cluster'] = kmeans.fit_predict(asset_metrics[['Return', 'Volatility']])
    
    return asset_metrics, kmeans

# Asset clustering and plotting
def cluster_assets(data, n_clusters=3):
    returns = calculate_returns(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(returns.T)
    clustered_data = pd.DataFrame({'Asset': data.columns, 'Cluster': clusters})

    # Plot clusters
    fig = go.Figure()
    for cluster in range(n_clusters):
        cluster_assets = clustered_data[clustered_data['Cluster'] == cluster]['Asset']
        cluster_returns = returns[cluster_assets]
        avg_return = cluster_returns.mean(axis=1)

        fig.add_trace(go.Scatter(
            x=avg_return.index,
            y=avg_return,
            mode='lines',
            name=f'Cluster {cluster}'
        ))

    fig.update_layout(
        title='Asset Clustering',
        xaxis_title='Date',
        yaxis_title='Average Returns',
        template='plotly_white'
    )

    return clustered_data, fig

# Correlation matrix plot
def plot_correlation_matrix(returns):
    corr_matrix = returns.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='Viridis',
        colorbar=dict(title="Correlation"),
    ))
    fig.update_layout(
        title="Correlation Matrix of Asset Returns",
        xaxis_title="Assets",
        yaxis_title="Assets",
        template="plotly_white"
    )
    return fig


def main():
    st.title("Portfolio Optimization and Risk Analysis")
    st.write("This application helps analyze and optimize a portfolio using Monte Carlo simulation, Mean-Variance optimization, and clustering techniques.")

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

    # Hyperparameters for optimization methods
    num_simulations = st.sidebar.number_input("Number of Monte Carlo Simulations:", min_value=1000, max_value=10000, value=5000, step=500)
    risk_tolerance = st.sidebar.number_input("Risk Tolerance (for Mean-Variance Optimization):", value=0.5, step=0.1)

    # User selection for method
    method = st.sidebar.selectbox("Select Optimization Method", ["Monte Carlo Simulation", "Mean-Variance Optimization"])

    # Process inputs
    if tickers:
        data = fetch_data(tickers, start_date, end_date)
        returns = calculate_returns(data)

        # Display historical data
        st.write("Historical Adjusted Close Prices:")
        st.line_chart(data)

        # Display correlation matrix
        st.subheader("Correlation Matrix")
        correlation_fig = plot_correlation_matrix(returns)
        st.plotly_chart(correlation_fig)
        st.write("""
        **Analysis of Correlation Matrix:**  
        - This heatmap shows the relationships between the returns of different assets.  
        - Positive values (closer to 1) indicate that two assets tend to move in the same direction.  
        - Negative values (closer to -1) indicate that two assets move in opposite directions.  
        - A diversified portfolio typically includes assets with low or negative correlations to minimize overall risk.
        """)
        
        if st.sidebar.button("Run Analysis"):
            # Tabs for different outputs
            # tabs = st.tabs(["Portfolio Optimization", "Risk Metrics", "Asset Clustering"])
            tabs = st.tabs(["Portfolio Optimization"])

            # Tab 1: Portfolio Optimization
            with tabs[0]:
                st.subheader("Optimized Portfolio")
                if method == "Monte Carlo Simulation":
                    portfolio_df, optimal_portfolio, min_vol_portfolio = monte_carlo_simulation(returns, risk_free_rate=risk_free_rate)
                    weights = optimal_portfolio['Weights']
                    
                    st.write("### Optimal Portfolio Weights")
                    st.dataframe(pd.DataFrame({'Ticker': tickers, 'Stock': selected_stock_names, 'Weight': weights}))
                    st.write("**Expected Return:**", f"{optimal_portfolio['Return']:.2%}")
                    st.write("**Volatility (Risk):**", f"{optimal_portfolio['Volatility']:.2%}")
                    st.write("**Sharpe Ratio:**", f"{optimal_portfolio['Sharpe']:.2f}")

                    st.write("### Minimum Volatility Portfolio")
                    st.dataframe(pd.DataFrame({'Ticker': tickers, 'Stock': selected_stock_names, "Weights": min_vol_portfolio["Weights"]}))
                    st.write("**Expected Return:**", f"{min_vol_portfolio['Return']:.2%}")
                    st.write("**Volatility (Risk):**", f"{min_vol_portfolio['Volatility']:.2%}")
                    st.write("**Sharpe Ratio:**", f"{min_vol_portfolio['Sharpe']:.2f}")

                    # Plot Efficient Frontier
                    frontier_fig = plot_efficient_frontier(returns)
                    st.plotly_chart(frontier_fig)
                    st.write("""
            **Analysis:**  
            - The Maximum Sharpe Ratio portfolio provides the best trade-off between risk and return.  
              - It maximizes the excess return per unit of risk, making it a balanced choice for growth-oriented investors.  
              - This portfolio often includes assets with strong historical performance and diversification benefits.  

            - The Minimum Volatility portfolio is ideal for risk-averse investors but may have lower returns.  
              - It aims to reduce portfolio fluctuations, offering stability during volatile market conditions.  
              - This portfolio typically emphasizes low-risk assets like bonds or defensive equities.  

            - The Efficient Frontier highlights the portfolios that offer the best possible return for a given level of risk.  
              - Portfolios along the Efficient Frontier are optimized for the highest return at each risk level.  
              - Moving up the frontier increases returns but also entails higher risk.  

            - Diversification plays a critical role in achieving these portfolios.  
              - Combining uncorrelated assets reduces overall portfolio risk.  
              - Proper asset allocation ensures resilience against market downturns.  

            - Consideration of individual risk tolerance and investment horizon is essential.  
              - Aggressive investors may favor portfolios closer to the Maximum Sharpe Ratio.  
              - Conservative investors may lean towards the Minimum Volatility portfolio.  

            - Regular portfolio rebalancing ensures alignment with investment goals.  
              - Market changes can shift the risk-return profile, necessitating adjustments.  
              - Rebalancing helps maintain the desired level of diversification and risk.  
            """)


                elif method == "Mean-Variance Optimization":
                    weights = optimize_portfolio(returns)
                    st.write("### Optimal Portfolio Weights")
                    st.dataframe(pd.DataFrame({'Ticker': tickers, 'Stock': selected_stock_names, 'Weight': weights}))

                    # Plot Efficient Frontier
                    frontier_fig = plot_efficient_frontier(returns)
                    st.plotly_chart(frontier_fig)
                    st.write("""
            **Analysis:**  
            - The Maximum Sharpe Ratio portfolio provides the best trade-off between risk and return.  
              - It maximizes the excess return per unit of risk, making it a balanced choice for growth-oriented investors.  
              - This portfolio often includes assets with strong historical performance and diversification benefits.  

            - The Minimum Volatility portfolio is ideal for risk-averse investors but may have lower returns.  
              - It aims to reduce portfolio fluctuations, offering stability during volatile market conditions.  
              - This portfolio typically emphasizes low-risk assets like bonds or defensive equities.  

            - The Efficient Frontier highlights the portfolios that offer the best possible return for a given level of risk.  
              - Portfolios along the Efficient Frontier are optimized for the highest return at each risk level.  
              - Moving up the frontier increases returns but also entails higher risk.  

            - Diversification plays a critical role in achieving these portfolios.  
              - Combining uncorrelated assets reduces overall portfolio risk.  
              - Proper asset allocation ensures resilience against market downturns.  

            - Consideration of individual risk tolerance and investment horizon is essential.  
              - Aggressive investors may favor portfolios closer to the Maximum Sharpe Ratio.  
              - Conservative investors may lean towards the Minimum Volatility portfolio.  

            - Regular portfolio rebalancing ensures alignment with investment goals.  
              - Market changes can shift the risk-return profile, necessitating adjustments.  
              - Rebalancing helps maintain the desired level of diversification and risk.  
            """)
    else:
        st.info("Please select at least one stock in the sidebar to begin analysis.")


if __name__ == "__main__":
    main()
