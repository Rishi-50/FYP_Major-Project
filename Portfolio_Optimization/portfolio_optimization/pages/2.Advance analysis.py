# TODO: UNCOMMENT THE FULL CODE TO RUN

import streamlit as st 
import sys
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adavanced_risk_analysis import *
from nifty50_stocks import NIFTY_50_STOCKS

def main():
    st.title("Advanced Portfolio Analysis")
    st.write("Explore volatility forecasting and return prediction methods for portfolio analysis.")

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
    risk_free_rate = st.sidebar.number_input("Risk-Free Rate (in decimal form):", value=0.04, step=0.01)

    if tickers:
        data = fetch_data(tickers, start_date, end_date)
        returns = calculate_returns(data)

        st.subheader("Volatility and Return Prediction")

        # Option to choose between Volatility and Return Prediction
        prediction_type = st.radio("Select Prediction Type:", options=["Volatility Forecasting", "Return Prediction"], index=0)

        if prediction_type == "Volatility Forecasting":
            # Volatility Forecasting
            st.markdown("### Volatility Forecasting")
            
            # Initialize containers for results
            forecast_results = {}
            failed_assets = []

            with st.spinner('Calculating volatility forecasts using GARCH model...'):
                for asset in tickers:
                    try:
                        forecast, metrics = forecast_volatility(data, asset, forecast_horizon=10, model_type="GARCH")
                        
                        # Ensure we have valid forecast values
                        if len(forecast) > 0:
                            forecast_results[asset] = {
                                "Forecasted Volatility": forecast[0],
                                "MSE": metrics['MSE'],
                                "RMSE": metrics['RMSE'],
                                "MAE": metrics['MAE']
                            }
                        else:
                            failed_assets.append((asset, "No valid forecast generated"))
                    except Exception as e:
                        failed_assets.append((asset, str(e)))
                        continue

            if forecast_results:
                # Display forecasted volatility
                st.subheader("Forecasted Volatility")
                try:
                    volatility_df = pd.DataFrame({
                        'Asset': list(forecast_results.keys()),
                        'Forecasted Volatility (%)': [results['Forecasted Volatility'] for results in forecast_results.values()]
                    }).set_index('Asset')
                    
                    # Format and display the dataframe
                    st.dataframe(volatility_df.style.format({'Forecasted Volatility (%)': '{:.2f}'}))

                    # Display evaluation metrics
                    st.subheader("GARCH Model Evaluation Metrics")
                    metrics_df = pd.DataFrame({
                        'Asset': list(forecast_results.keys()),
                        'MSE': [results['MSE'] for results in forecast_results.values()],
                        'RMSE': [results['RMSE'] for results in forecast_results.values()],
                        'MAE': [results['MAE'] for results in forecast_results.values()]
                    }).set_index('Asset')
                    
                    st.dataframe(metrics_df.style.format({
                        'MSE': '{:.6f}',
                        'RMSE': '{:.6f}',
                        'MAE': '{:.6f}'
                    }))

                    # Create a bar chart for volatility comparison
                    if len(volatility_df) > 0:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=volatility_df.index,
                            y=volatility_df['Forecasted Volatility (%)'],
                            name='Forecasted Volatility',
                            marker_color='lightblue'
                        ))
                        fig.update_layout(
                            title="GARCH Model: Forecasted Volatility Comparison",
                            xaxis_title="Assets",
                            yaxis_title="Forecasted Volatility (%)",
                            template="plotly_white"
                        )
                        st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"Error displaying results: {str(e)}")

            # Display any failures
            if failed_assets:
                st.warning("Some assets could not be processed:")
                for asset, error in failed_assets:
                    st.error(f"{asset}: {error}")
            
            if not forecast_results and not failed_assets:
                st.warning("No forecasts could be generated. Please check your data and parameters.")

            st.markdown("""
            #### Interpretation of GARCH Model Metrics:
            - **MSE (Mean Squared Error)**: Measures the average squared difference between predicted and actual volatility. Lower values indicate better predictions.
            - **RMSE (Root Mean Squared Error)**: Square root of MSE, provides error metric in the same unit as volatility. Lower values are better.
            - **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual volatility. Less sensitive to outliers than MSE.
            """)

            # Interpretation of Volatility Forecast
            st.markdown("#### GARCH Model Volatility Forecast Interpretation:")
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
            # Return Prediction using LSTM
            st.markdown("### LSTM Return Prediction")

            # Forecast horizon option for prediction
            forecast_horizon = st.slider("Forecast Horizon (Days):", 1, 30, 1)

            try:
                # Prepare latest data for prediction
                if len(returns) < 7:
                    st.error(f"Insufficient data: need at least 7 days of returns.")
                    return

                latest_data = returns.iloc[-7:].values
                latest_data = latest_data.reshape(1, 7, returns.shape[1])

                # Train LSTM model
                model, mse = train_return_predictor_lstm(returns, lookback=7, forecast_horizon=forecast_horizon)
                
                # Display evaluation metrics
                st.write("### LSTM Model Evaluation Metrics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Mean Squared Error (MSE)", f"{mse:.6f}")
                
                with col2:
                    rmse = np.sqrt(mse)
                    st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.6f}")
                
                with col3:
                    # Calculate MAE using the latest prediction
                    predicted = model.predict(latest_data).flatten()
                    actual = returns.iloc[-1].values
                    mae = np.mean(np.abs(predicted - actual))
                    st.metric("Mean Absolute Error (MAE)", f"{mae:.6f}")
                
                st.write("#### Interpretation of LSTM Metrics:")
                st.markdown("""
                - **MSE**: Measures the average squared difference between predicted and actual returns. Lower values indicate better predictions.
                - **RMSE**: Square root of MSE, provides error metric in the same unit as the returns. Lower values are better.
                - **MAE**: Average absolute difference between predicted and actual returns. Less sensitive to outliers than MSE.
                """)

                # Make predictions for the next period
                predicted_returns = model.predict(latest_data).flatten()
                
                # Convert the predicted returns to percentages
                predicted_returns_percent = predicted_returns * 100

                # Format the predicted returns as strings with '%' symbol
                predicted_returns_percent_str = [f"{value:.2f}%" for value in predicted_returns_percent]

                # Display predictions
                st.write("### LSTM Predicted Returns for Each Asset")
                predicted_df = pd.DataFrame({
                    "Asset": returns.columns,
                    "Predicted Return": predicted_returns_percent_str
                })
                st.dataframe(predicted_df)

                # Plot predictions
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=predicted_df["Asset"],
                    y=predicted_returns_percent,
                    name="Predicted Returns",
                    marker_color=['red' if x < 0 else 'green' for x in predicted_returns_percent]
                ))
                fig.update_layout(
                    title="LSTM Model: Predicted Returns for Next Period",
                    xaxis_title="Asset",
                    yaxis_title="Predicted Return (%)",
                    template="plotly_white",
                    showlegend=True
                )
                st.plotly_chart(fig)

                # Add confidence intervals to the predictions
                st.subheader("LSTM Prediction Confidence")
                confidence_df = pd.DataFrame({
                    "Asset": returns.columns,
                    "Predicted Return (%)": [f"{x:.2f}%" for x in predicted_returns_percent],
                    "Confidence Level": ["High" if abs(x) > 2*rmse else "Medium" if abs(x) > rmse else "Low" for x in predicted_returns_percent]
                })
                st.dataframe(confidence_df)

                st.markdown("#### LSTM Model Return Prediction Interpretation:")
                st.write(
                f"""
                The LSTM (Long Short-Term Memory) model has generated predictions for each asset's returns. Here's how to interpret the results:
                
                - **Predicted returns** show the model's estimate of how each asset may perform in the next {forecast_horizon} day(s), based on the past 7 days of data.
                
                - **Positive predicted returns** (shown in green) suggest potential price increases.
                - **Negative predicted returns** (shown in red) suggest potential price decreases.
                
                - **Confidence Levels**:
                    - High: The predicted return is more than twice the model's typical error (RMSE)
                    - Medium: The predicted return is between one and two times the model's typical error
                    - Low: The predicted return is less than the model's typical error
                
                - **Model Performance**:
                    - MSE: {mse:.6f}
                    - RMSE: {rmse:.6f}
                    - MAE: {mae:.6f}
                
                **Note**: These predictions are based on historical patterns and should be used as one of many tools in making investment decisions. Always consider other factors such as market conditions, company fundamentals, and your investment goals.
                """
                )
            except Exception as e:
                st.error(f"An error occurred during LSTM return prediction: {str(e)}")
                st.write("Please try adjusting the forecast horizon.")
    else:
        st.info("Please select at least one stock in the sidebar to begin analysis.")

if __name__ == "__main__":
    main()