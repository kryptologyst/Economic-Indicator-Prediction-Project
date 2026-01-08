"""
Streamlit demo application for Economic Indicator Prediction.

This application provides an interactive interface for exploring economic forecasting
models, visualizing predictions, and running backtests.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import load_config
from data import EconomicDataLoader, DataPreprocessor
from features import FeatureEngineer
from models import ModelFactory
from evaluation import ModelEvaluator, FinancialEvaluator
from risk import RiskManager


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Economic Indicator Prediction",
        page_icon="📈",
        layout="wide"
    )
    
    # Add disclaimer
    st.warning("""
    **DISCLAIMER: This application is for RESEARCH AND EDUCATIONAL PURPOSES ONLY.**
    
    - This is NOT investment advice
    - Predictions may be inaccurate and should not be used for financial decisions
    - Backtests are hypothetical and do not guarantee future performance
    - Past performance does not indicate future results
    - Always consult with qualified financial professionals before making investment decisions
    """)
    
    st.title("📈 Economic Indicator Prediction")
    st.markdown("Advanced machine learning models for economic forecasting")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["linear", "tree", "neural_network", "arima", "ensemble"],
        index=4
    )
    
    # Data configuration
    st.sidebar.subheader("Data Configuration")
    use_real_data = st.sidebar.checkbox("Use Real Data (FRED)", value=False)
    prediction_horizon = st.sidebar.slider("Prediction Horizon (days)", 1, 30, 7)
    
    # Feature configuration
    st.sidebar.subheader("Features")
    include_technical = st.sidebar.checkbox("Technical Indicators", value=True)
    include_economic = st.sidebar.checkbox("Economic Features", value=True)
    include_time = st.sidebar.checkbox("Time Features", value=True)
    
    # Risk configuration
    st.sidebar.subheader("Risk Management")
    calculate_var = st.sidebar.checkbox("Calculate VaR", value=True)
    run_stress_tests = st.sidebar.checkbox("Run Stress Tests", value=True)
    
    # Load configuration
    try:
        config = load_config("configs/default.yaml")
        
        # Update config based on UI selections
        config.model.type = model_type
        config.features.technical_indicators.enabled = include_technical
        config.features.economic_features.enabled = include_economic
        config.features.time_features.enabled = include_time
        config.risk.calculate_var = calculate_var
        config.risk.run_stress_tests = run_stress_tests
        
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        return
    
    # Load data
    with st.spinner("Loading data..."):
        try:
            data_loader = EconomicDataLoader("data")
            
            if use_real_data:
                # Try to load real data
                try:
                    data = data_loader.load_fred_data(
                        ["GDPC1", "CPIAUCSL", "UNRATE", "FEDFUNDS"],
                        start_date="2000-01-01"
                    )
                    st.success("✅ Real economic data loaded from FRED")
                except Exception as e:
                    st.warning(f"Could not load real data: {e}. Using synthetic data.")
                    data = data_loader._generate_synthetic_economic_data(4)
            else:
                data = data_loader._generate_synthetic_economic_data(4)
            
            # Display data info
            st.subheader("📊 Data Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Data Points", len(data))
            with col2:
                st.metric("Features", len(data.columns))
            with col3:
                st.metric("Date Range", f"{data.index[0].date()} to {data.index[-1].date()}")
            
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(data.head(10))
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return
    
    # Feature engineering
    with st.spinner("Creating features..."):
        try:
            feature_engineer = FeatureEngineer(config.features)
            features_data = feature_engineer.create_features(data)
            
            st.subheader("🔧 Feature Engineering")
            st.write(f"Created {features_data.shape[1]} features from {data.shape[1]} original columns")
            
        except Exception as e:
            st.error(f"Error in feature engineering: {e}")
            return
    
    # Model training
    with st.spinner("Training model..."):
        try:
            # Prepare target variable
            target_col = "GDP_Growth"
            if target_col not in features_data.columns:
                target_col = features_data.columns[0]
            
            features_data = features_data.dropna(subset=[target_col])
            target = features_data[target_col].shift(-1)
            features = features_data.drop(columns=[target_col])
            
            valid_idx = ~target.isna()
            features = features[valid_idx]
            target = target[valid_idx]
            
            # Train-test split
            split_idx = int(len(features) * 0.8)
            X_train = features.iloc[:split_idx]
            y_train = target.iloc[:split_idx]
            X_test = features.iloc[split_idx:]
            y_test = target.iloc[split_idx:]
            
            # Scale features
            preprocessor = DataPreprocessor("standard")
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)
            
            # Train model
            model = ModelFactory.create_model(config.model)
            model.fit(X_train_scaled, y_train)
            
            st.success(f"✅ {model_type.title()} model trained successfully")
            
        except Exception as e:
            st.error(f"Error training model: {e}")
            return
    
    # Predictions
    st.subheader("🔮 Predictions")
    
    try:
        predictions = model.predict(X_test_scaled)
        
        # Create prediction plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=y_test.index,
            y=y_test.values,
            mode='lines',
            name='Actual',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=y_test.index,
            y=predictions,
            mode='lines',
            name='Predicted',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title="Actual vs Predicted Values",
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction metrics
        col1, col2, col3, col4 = st.columns(4)
        
        mse = np.mean((y_test.values - predictions) ** 2)
        mae = np.mean(np.abs(y_test.values - predictions))
        rmse = np.sqrt(mse)
        r2 = 1 - np.sum((y_test.values - predictions) ** 2) / np.sum((y_test.values - np.mean(y_test.values)) ** 2)
        
        with col1:
            st.metric("RMSE", f"{rmse:.4f}")
        with col2:
            st.metric("MAE", f"{mae:.4f}")
        with col3:
            st.metric("R²", f"{r2:.4f}")
        with col4:
            hit_rate = np.mean(np.sign(np.diff(y_test.values)) == np.sign(np.diff(predictions)))
            st.metric("Hit Rate", f"{hit_rate:.2%}")
        
    except Exception as e:
        st.error(f"Error making predictions: {e}")
    
    # Feature importance
    if hasattr(model, 'get_feature_importance'):
        try:
            feature_importance = model.get_feature_importance()
            if feature_importance is not None:
                st.subheader("📊 Feature Importance")
                
                # Get top 20 features
                top_features = feature_importance.nlargest(20)
                
                fig = px.bar(
                    x=top_features.values,
                    y=top_features.index,
                    orientation='h',
                    title="Top 20 Most Important Features"
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.warning(f"Could not display feature importance: {e}")
    
    # Backtesting
    st.subheader("📈 Backtesting")
    
    try:
        # Calculate returns
        returns = y_test.pct_change().dropna()
        
        # Simple buy and hold strategy
        portfolio_value = 10000 * (1 + returns).cumprod()
        
        # Create backtest plot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Portfolio Value", "Returns"),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=portfolio_value.index, y=portfolio_value.values, name="Portfolio Value"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=returns.index, y=returns.values, name="Returns"),
            row=2, col=1
        )
        
        fig.update_layout(height=600, title="Backtest Results")
        st.plotly_chart(fig, use_container_width=True)
        
        # Backtest metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1
        annualized_return = returns.mean() * 252
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        with col1:
            st.metric("Total Return", f"{total_return:.2%}")
        with col2:
            st.metric("Annualized Return", f"{annualized_return:.2%}")
        with col3:
            st.metric("Volatility", f"{volatility:.2%}")
        with col4:
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    
    except Exception as e:
        st.error(f"Error in backtesting: {e}")
    
    # Risk analysis
    if calculate_var:
        st.subheader("⚠️ Risk Analysis")
        
        try:
            # Calculate VaR
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("95% VaR", f"{var_95:.2%}")
            with col2:
                st.metric("99% VaR", f"{var_99:.2%}")
            
            # VaR plot
            fig = go.Figure()
            fig.add_histogram(x=returns.values, name="Returns Distribution")
            fig.add_vline(x=var_95, line_dash="dash", line_color="red", annotation_text="95% VaR")
            fig.add_vline(x=var_99, line_dash="dash", line_color="darkred", annotation_text="99% VaR")
            
            fig.update_layout(
                title="Returns Distribution with VaR",
                xaxis_title="Returns",
                yaxis_title="Frequency"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error in risk analysis: {e}")
    
    # Stress testing
    if run_stress_tests:
        st.subheader("🧪 Stress Testing")
        
        try:
            # Simple stress test scenarios
            scenarios = {
                "Baseline": predictions,
                "Recession (-20%)": predictions * 0.8,
                "High Growth (+20%)": predictions * 1.2,
                "Volatility Shock": predictions + np.random.normal(0, 0.1, len(predictions))
            }
            
            # Create stress test plot
            fig = go.Figure()
            
            colors = ['blue', 'red', 'green', 'orange']
            for i, (scenario, values) in enumerate(scenarios.items()):
                fig.add_trace(go.Scatter(
                    x=y_test.index,
                    y=values,
                    mode='lines',
                    name=scenario,
                    line=dict(color=colors[i])
                ))
            
            fig.update_layout(
                title="Stress Test Scenarios",
                xaxis_title="Date",
                yaxis_title="Predicted Value"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Stress test metrics
            st.subheader("Stress Test Results")
            
            stress_results = []
            for scenario, values in scenarios.items():
                mse = np.mean((y_test.values - values) ** 2)
                mae = np.mean(np.abs(y_test.values - values))
                stress_results.append({
                    "Scenario": scenario,
                    "MSE": mse,
                    "MAE": mae
                })
            
            stress_df = pd.DataFrame(stress_results)
            st.dataframe(stress_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error in stress testing: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Note**: This is a demonstration application for educational purposes. 
    The models and predictions shown here are not intended for actual investment decisions.
    """)


if __name__ == "__main__":
    main()
