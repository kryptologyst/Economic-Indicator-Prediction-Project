# Economic Indicator Prediction Project

## DISCLAIMER

**IMPORTANT: This project is for RESEARCH AND EDUCATIONAL PURPOSES ONLY.**

- This is NOT investment advice
- Predictions may be inaccurate and should not be used for financial decisions
- Backtests are hypothetical and do not guarantee future performance
- Past performance does not indicate future results
- Always consult with qualified financial professionals before making investment decisions

## Overview

This project implements advanced machine learning models for predicting economic indicators such as GDP growth, inflation, unemployment rates, and other macroeconomic variables. The system includes:

- Multiple forecasting models (ARIMA, VAR, tree-based, neural networks)
- Comprehensive evaluation with both ML and financial metrics
- Risk management and uncertainty quantification
- Interactive demo for exploration and visualization
- Production-ready code structure with proper documentation

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the demo:**
   ```bash
   streamlit run demo/app.py
   ```

3. **Train models:**
   ```bash
   python scripts/train.py --config configs/default.yaml
   ```

4. **Run backtests:**
   ```bash
   python scripts/backtest.py --config configs/backtest.yaml
   ```

## Project Structure

```
├── src/                    # Core source code
│   ├── data/              # Data loading and preprocessing
│   ├── features/          # Feature engineering
│   ├── models/            # ML models
│   ├── evaluation/        # Evaluation metrics and backtesting
│   ├── risk/              # Risk management
│   └── utils/             # Utilities
├── data/                  # Data storage
├── configs/               # Configuration files
├── scripts/               # Training and evaluation scripts
├── notebooks/             # Jupyter notebooks for exploration
├── tests/                 # Unit tests
├── assets/                # Generated plots and results
├── demo/                  # Streamlit demo application
└── docs/                  # Documentation
```

## Dataset Schema

The project expects economic data in the following format:

- **market_data.csv**: OHLCV data for financial instruments
- **fundamentals.csv**: Economic indicators (GDP, inflation, unemployment, etc.)
- **labels.csv**: Target variables for prediction

## Models

- **Linear Regression**: Baseline model
- **ARIMA/ARIMAX**: Time series forecasting
- **VAR**: Vector autoregression for multiple indicators
- **XGBoost/LightGBM**: Tree-based ensemble methods
- **Neural Networks**: Deep learning for complex patterns
- **Ensemble**: Combination of multiple models

## Evaluation Metrics

### ML Metrics
- RMSE, MAE, SMAPE, MASE for forecasting
- R² for regression quality

### Financial Metrics
- Information Ratio
- Sharpe Ratio
- Maximum Drawdown
- Calmar Ratio
- Hit Rate

## Configuration

Models and experiments are configured via YAML files in the `configs/` directory. Key parameters include:

- Data sources and preprocessing
- Model hyperparameters
- Evaluation settings
- Risk management rules

## Contributing

1. Follow the code style (black + ruff)
2. Add type hints and docstrings
3. Write tests for new functionality
4. Update documentation

## License

This project is for educational purposes only. See LICENSE file for details.# Economic-Indicator-Prediction-Project
