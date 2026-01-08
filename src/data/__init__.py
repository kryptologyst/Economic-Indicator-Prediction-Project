"""
Data loading and preprocessing utilities for economic indicators.

This module handles loading economic data from various sources, preprocessing,
and creating datasets suitable for machine learning models.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


class EconomicDataLoader:
    """Loader for economic indicator data."""
    
    def __init__(self, data_dir: Union[str, Path]):
        """Initialize data loader.
        
        Args:
            data_dir: Directory containing data files.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_fred_data(
        self,
        series_ids: List[str],
        start_date: str = "2000-01-01",
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load data from FRED (Federal Reserve Economic Data).
        
        Args:
            series_ids: List of FRED series IDs.
            start_date: Start date for data.
            end_date: End date for data (defaults to today).
            
        Returns:
            DataFrame with economic indicators.
        """
        try:
            import fredapi
            fred = fredapi.Fred(api_key="demo")  # Use demo key for now
            
            data = {}
            for series_id in series_ids:
                try:
                    series_data = fred.get_series(
                        series_id, start=start_date, end=end_date
                    )
                    data[series_id] = series_data
                    logger.info(f"Loaded {series_id}: {len(series_data)} observations")
                except Exception as e:
                    logger.warning(f"Failed to load {series_id}: {e}")
            
            df = pd.DataFrame(data)
            df.index.name = "date"
            return df
            
        except ImportError:
            logger.warning("fredapi not available, generating synthetic data")
            return self._generate_synthetic_economic_data(len(series_ids))
    
    def load_yahoo_finance_data(
        self,
        symbols: List[str],
        start_date: str = "2000-01-01",
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load financial data from Yahoo Finance.
        
        Args:
            symbols: List of ticker symbols.
            start_date: Start date for data.
            end_date: End date for data.
            
        Returns:
            DataFrame with OHLCV data.
        """
        data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date)
                if not hist.empty:
                    data[symbol] = hist["Close"]
                    logger.info(f"Loaded {symbol}: {len(hist)} observations")
            except Exception as e:
                logger.warning(f"Failed to load {symbol}: {e}")
        
        df = pd.DataFrame(data)
        df.index.name = "date"
        return df
    
    def _generate_synthetic_economic_data(
        self, n_series: int, n_periods: int = 252
    ) -> pd.DataFrame:
        """Generate synthetic economic data for demonstration.
        
        Args:
            n_series: Number of economic series to generate.
            n_periods: Number of time periods.
            
        Returns:
            DataFrame with synthetic economic data.
        """
        np.random.seed(42)
        
        # Generate realistic economic time series
        dates = pd.date_range("2000-01-01", periods=n_periods, freq="D")
        
        series_names = [
            "GDP_Growth", "Inflation", "Interest_Rate", "Unemployment",
            "Consumer_Confidence", "Industrial_Production", "Retail_Sales"
        ]
        
        data = {}
        for i in range(min(n_series, len(series_names))):
            name = series_names[i]
            
            # Generate trend + seasonality + noise
            trend = np.linspace(0, 0.5, n_periods)
            seasonal = 0.3 * np.sin(2 * np.pi * np.arange(n_periods) / 252)
            noise = np.random.normal(0, 0.1, n_periods)
            
            # Add some autocorrelation
            series = np.zeros(n_periods)
            series[0] = trend[0] + seasonal[0] + noise[0]
            
            for t in range(1, n_periods):
                series[t] = (
                    0.7 * series[t-1] + 
                    0.3 * (trend[t] + seasonal[t] + noise[t])
                )
            
            data[name] = series
        
        df = pd.DataFrame(data, index=dates)
        df.index.name = "date"
        return df
    
    def load_csv_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load data from CSV file.
        
        Args:
            file_path: Path to CSV file.
            
        Returns:
            DataFrame with loaded data.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded data from {file_path}: {df.shape}")
        return df
    
    def save_data(self, df: pd.DataFrame, file_path: Union[str, Path]) -> None:
        """Save DataFrame to CSV file.
        
        Args:
            df: DataFrame to save.
            file_path: Output file path.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(file_path)
        logger.info(f"Saved data to {file_path}: {df.shape}")


class DataPreprocessor:
    """Data preprocessing utilities."""
    
    def __init__(self, scaler_type: str = "standard"):
        """Initialize preprocessor.
        
        Args:
            scaler_type: Type of scaler ("standard", "minmax", "robust").
        """
        scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
        }
        
        self.scaler = scalers.get(scaler_type, StandardScaler())
        self.is_fitted = False
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler and transform data.
        
        Args:
            data: Input data.
            
        Returns:
            Scaled data.
        """
        scaled_data = self.scaler.fit_transform(data)
        self.is_fitted = True
        
        return pd.DataFrame(
            scaled_data,
            index=data.index,
            columns=data.columns
        )
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted scaler.
        
        Args:
            data: Input data.
            
        Returns:
            Scaled data.
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        scaled_data = self.scaler.transform(data)
        
        return pd.DataFrame(
            scaled_data,
            index=data.index,
            columns=data.columns
        )
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform scaled data.
        
        Args:
            data: Scaled data.
            
        Returns:
            Original scale data.
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse transform")
        
        original_data = self.scaler.inverse_transform(data)
        
        return pd.DataFrame(
            original_data,
            index=data.index,
            columns=data.columns
        )
    
    def handle_missing_values(
        self,
        data: pd.DataFrame,
        method: str = "forward_fill",
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Handle missing values in data.
        
        Args:
            data: Input data.
            method: Method for handling missing values.
            limit: Maximum number of consecutive periods to fill.
            
        Returns:
            Data with missing values handled.
        """
        if method == "forward_fill":
            return data.fillna(method="ffill", limit=limit)
        elif method == "backward_fill":
            return data.fillna(method="bfill", limit=limit)
        elif method == "interpolate":
            return data.interpolate(limit=limit)
        elif method == "drop":
            return data.dropna()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def create_lagged_features(
        self,
        data: pd.DataFrame,
        lags: List[int],
        target_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Create lagged features for time series modeling.
        
        Args:
            data: Input data.
            lags: List of lag periods.
            target_cols: Columns to create lags for (defaults to all).
            
        Returns:
            Data with lagged features.
        """
        if target_cols is None:
            target_cols = data.columns.tolist()
        
        lagged_data = data.copy()
        
        for col in target_cols:
            for lag in lags:
                lagged_data[f"{col}_lag_{lag}"] = data[col].shift(lag)
        
        return lagged_data
    
    def create_rolling_features(
        self,
        data: pd.DataFrame,
        windows: List[int],
        functions: List[str] = ["mean", "std"],
        target_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Create rolling window features.
        
        Args:
            data: Input data.
            windows: List of window sizes.
            functions: List of aggregation functions.
            target_cols: Columns to create features for.
            
        Returns:
            Data with rolling features.
        """
        if target_cols is None:
            target_cols = data.columns.tolist()
        
        rolling_data = data.copy()
        
        for col in target_cols:
            for window in windows:
                for func in functions:
                    rolling_data[f"{col}_{func}_{window}"] = (
                        data[col].rolling(window=window).agg(func)
                    )
        
        return rolling_data


def create_time_series_splits(
    data: pd.DataFrame,
    n_splits: int = 5,
    test_size: float = 0.2,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Create time series splits for cross-validation.
    
    Args:
        data: Input data.
        n_splits: Number of splits.
        test_size: Proportion of data for testing.
        
    Returns:
        List of (train, test) splits.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    
    for train_idx, test_idx in tscv.split(data):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        splits.append((train_data, test_data))
    
    return splits
