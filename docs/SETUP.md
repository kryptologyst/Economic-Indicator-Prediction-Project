# Economic Indicator Prediction - Setup Guide

## Prerequisites

- Python 3.10 or higher
- pip package manager
- Git (optional, for version control)

## Installation

1. **Clone or download the project:**
   ```bash
   git clone <repository-url>
   cd 0506_Economic_Indicator_Prediction
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install pre-commit hooks (optional):**
   ```bash
   pre-commit install
   ```

## Quick Start

1. **Run the interactive demo:**
   ```bash
   streamlit run demo/app.py
   ```
   Open your browser to `http://localhost:8501`

2. **Train a model:**
   ```bash
   python scripts/train.py --config configs/default.yaml
   ```

3. **Run backtests:**
   ```bash
   python scripts/backtest.py --config configs/backtest.yaml
   ```

4. **Explore with Jupyter:**
   ```bash
   jupyter notebook notebooks/example_analysis.ipynb
   ```

## Configuration

The project uses YAML configuration files in the `configs/` directory:

- `default.yaml`: Main configuration for training and evaluation
- `backtest.yaml`: Backtesting-specific configuration

You can modify these files to:
- Change model parameters
- Adjust feature engineering settings
- Configure risk management rules
- Set evaluation metrics

## Data Sources

The project can work with:

1. **Real Economic Data (FRED):**
   - Requires FRED API key
   - Set `use_real_data: true` in demo
   - Configure API key in `configs/default.yaml`

2. **Synthetic Data (Default):**
   - Generated for demonstration purposes
   - Realistic economic time series patterns
   - No external dependencies

## Troubleshooting

### Common Issues

1. **Import Errors:**
   ```bash
   # Make sure you're in the project directory
   cd 0506_Economic_Indicator_Prediction
   
   # Check Python path
   python -c "import sys; print(sys.path)"
   ```

2. **Missing Dependencies:**
   ```bash
   # Reinstall requirements
   pip install -r requirements.txt --force-reinstall
   ```

3. **CUDA/MPS Issues:**
   - The project automatically falls back to CPU if CUDA/MPS is not available
   - Check device detection: `python -c "from src.utils import get_device; print(get_device())"`

4. **Data Loading Issues:**
   - Ensure data directory exists: `mkdir -p data/raw data/processed`
   - Check file permissions

### Performance Tips

1. **For faster training:**
   - Reduce dataset size in configuration
   - Use simpler models (linear regression)
   - Disable expensive features (technical indicators)

2. **For better accuracy:**
   - Increase training data size
   - Use ensemble models
   - Enable all feature engineering options

## Development

### Code Style

The project uses:
- **Black** for code formatting
- **Ruff** for linting
- **Type hints** for better code documentation

Run formatting:
```bash
black .
ruff check .
```

### Testing

Run tests:
```bash
pytest tests/ -v
```

### Adding New Features

1. Create new modules in `src/`
2. Add type hints and docstrings
3. Write unit tests
4. Update configuration files
5. Update documentation

## Support

For questions or issues:
1. Check the troubleshooting section
2. Review the example notebook
3. Check the demo application
4. Create an issue in the repository

## Disclaimer

**This software is for RESEARCH AND EDUCATIONAL PURPOSES ONLY.**

- This is NOT investment advice
- Predictions may be inaccurate
- Do not use for actual financial decisions
- Always consult qualified professionals
