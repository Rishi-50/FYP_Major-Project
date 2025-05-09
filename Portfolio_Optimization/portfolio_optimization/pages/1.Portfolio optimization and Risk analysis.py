import streamlit as st 
from portfolio_optimisation import *
import sys
import os
# Import the Nifty 50 stocks
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nifty50_stocks import NIFTY_50_STOCKS

def main():
    st.title("Portfolio Optimization and Risk Analysis")
    st.write("This application helps analyze and optimize a portfolio using Monte Carlo simulation.")

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
            tabs = st.tabs(["Portfolio Optimization"])

            # Tab 1: Portfolio Optimization
            with tabs[0]:
                st.subheader("Optimized Portfolio")
                
                # Monte Carlo Simulation with fixed 5000 simulations
                portfolio_df, optimal_portfolio, min_vol_portfolio = monte_carlo_simulation(returns, num_portfolios=5000, risk_free_rate=risk_free_rate)
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
    else:
        st.info("Please select at least one stock in the sidebar to begin analysis.")

if __name__ == "__main__":
    main()
