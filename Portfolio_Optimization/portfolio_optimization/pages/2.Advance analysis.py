# TODO: UNCOMMENT THE FULL CODE TO RUN

import streamlit as st 
import sys
import os
import numpy as np

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adavanced_risk_analysis import *

# Main app
def main():
    st.title("Advanced Portfolio Optimization and Risk Analysis")
    st.write("Explore PCA, time-series forecasting, risk parity, and anomaly detection methods for portfolio analysis.")

    # Sidebar inputs
    st.sidebar.header("User Input")
    tickers = st.sidebar.text_input("Enter asset tickers (comma-separated):", 
                                     "TCS.NS,ITC.NS,RELIANCE.NS,HDFCBANK.NS,INFY.NS")
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))
    risk_free_rate = st.sidebar.number_input("Risk-Free Rate (in decimal form):", value=0.04, step=0.01)

    tickers = tickers.split(',')
    data = fetch_data(tickers, start_date, end_date)
    returns = calculate_returns(data)

     # Tabs for different functionalities
    # tab1,
    tab2, tab3, tab4 = st.tabs([
        "Risk Parity", "Volatility and Return Prediction", "Anomaly Detection"
    ])



    # with tab1:
    #     st.subheader("PCA Analysis")
    #     pca_data, explained_variance = apply_pca(returns)
    #     st.dataframe(pca_data)
    #     pca_fig = plot_pca_results(pca_data, explained_variance)
    #     st.plotly_chart(pca_fig)

    #     st.markdown("""
    #     ### What is PCA?
    #     Principal Component Analysis (PCA) reduces the dimensionality of data while retaining most of its variance. 
    #     It transforms correlated variables into a smaller number of uncorrelated variables called principal components.

    #     ### How is PCA Helpful?
    #     - **Understanding Drivers of Variance**: PCA identifies the main factors driving portfolio returns.
    #     - **Risk Management**: By focusing on major components, you can better understand where risks are concentrated.
    #     - **Data Simplification**: Reduces the complexity of large datasets, making analysis more manageable.

    #     ### Interpretation:
    #     - PC1 explains the majority of variance in the portfolio, showing where most of the movement in asset prices originates.
    #     - Lower components like PC2 and PC3 explain residual variance, which may relate to secondary factors or noise.

    #     ### **Example Use Case**:
    #     1. **Financial Data Analysis**:
    #     - In financial markets, PCA can be applied to analyze correlations between assets.
    #     - For example, returns of multiple stocks can be reduced to a few principal components, where PC1 may capture the overall market movement, and subsequent components may reflect sector-specific trends.

    #     2. **Customer Segmentation**:
    #     - In marketing, PCA helps reduce customer behavior data into fewer dimensions, enabling clustering and segmentation analysis.

    #     ---

    #     ### **Applications of PCA**:
    #     - **Exploratory Data Analysis (EDA)**: Identify patterns and clusters in datasets.
    #     - **Preprocessing**: Reduce dimensionality before applying machine learning algorithms.
    #     - **Visualization**: Plot high-dimensional data in 2D or 3D space for easier interpretation.
    #     - **Feature Engineering**: Extract meaningful features for modeling.

    #     ---

    #     ### **Summary of PCA Analysis**:
    #     - In this analysis:
    #     - The data is transformed into principal components, with **Principal Component 1 (PC1)** and **Principal Component 2 (PC2)** capturing the most variance.
    #     - The **explained variance** metric indicates how much information is preserved in the reduced dimensions.
    #     - The first two components explain the majority of the variance, allowing for a simplified yet informative representation of the dataset.
    #     """)

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
            forecast_horizon = st.slider("Forecast Horizon (Days):", 5, 30, 10)
            
            # Initialize containers for results
            forecast_results = {}
            failed_assets = []

            with st.spinner('Calculating volatility forecasts using GARCH model...'):
                for asset in tickers:
                    try:
                        forecast, metrics = forecast_volatility(data, asset, forecast_horizon=forecast_horizon, model_type="GARCH")
                        
                        # Ensure we have valid forecast values
                        if len(forecast) > 0:
                            forecast_results[asset] = {
                                'Forecasted_Volatility': forecast[0],  # First forecast point
                                'MSE': metrics['MSE'],
                                'RMSE': metrics['RMSE'],
                                'MAE': metrics['MAE'],
                                'Directional_Accuracy': metrics['Directional_Accuracy']
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
                        'Forecasted Volatility (%)': [results['Forecasted_Volatility'] for results in forecast_results.values()]
                    }).set_index('Asset')
                    
                    # Format and display the dataframe
                    st.dataframe(volatility_df.style.format({'Forecasted Volatility (%)': '{:.2f}'}))

                    # Display evaluation metrics
                    st.subheader("GARCH Model Evaluation Metrics")
                    metrics_df = pd.DataFrame({
                        'Asset': list(forecast_results.keys()),
                        'MSE': [results['MSE'] for results in forecast_results.values()],
                        'RMSE': [results['RMSE'] for results in forecast_results.values()],
                        'MAE': [results['MAE'] for results in forecast_results.values()],
                        'Directional Accuracy (%)': [results['Directional_Accuracy'] * 100 for results in forecast_results.values()]
                    }).set_index('Asset')
                    
                    st.dataframe(metrics_df.style.format({
                        'MSE': '{:.6f}',
                        'RMSE': '{:.6f}',
                        'MAE': '{:.6f}',
                        'Directional Accuracy (%)': '{:.2f}'
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
            - **Directional Accuracy**: Percentage of times the model correctly predicts whether volatility will increase or decrease. Higher values indicate better directional prediction.
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

            # Forecast horizon and lookback options for prediction
            forecast_horizon = st.slider("Forecast Horizon (Days):", 1, 30, 1)
            lookback = st.slider("Lookback Period (Days):", 1, 30, 5)

            try:
                # Prepare latest data for prediction
                if len(returns) < lookback:
                    st.error(f"Insufficient data: need at least {lookback} days of returns.")
                    return

                latest_data = returns.iloc[-lookback:].values
                latest_data = latest_data.reshape(1, lookback, returns.shape[1])

                # Train LSTM model
                model, mse = train_return_predictor_lstm(returns, lookback=lookback, forecast_horizon=forecast_horizon)
                
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
                
                - **Predicted returns** show the model's estimate of how each asset may perform in the next {forecast_horizon} day(s), based on the past {lookback} days of data.
                
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
                st.write("Please try adjusting the lookback period or forecast horizon.")

        
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



if __name__ == "__main__":
    main()