"""
Feature engineering for economic indicator prediction.

This module provides feature engineering utilities including technical indicators,
economic features, and time-based features for economic forecasting models.
"""

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Technical indicators for economic time series."""
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average.
        
        Args:
            data: Input time series.
            window: Window size.
            
        Returns:
            SMA values.
        """
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int, alpha: Optional[float] = None) -> pd.Series:
        """Exponential Moving Average.
        
        Args:
            data: Input time series.
            window: Window size.
            alpha: Smoothing factor (if None, uses 2/(window+1)).
            
        Returns:
            EMA values.
        """
        if alpha is None:
            alpha = 2.0 / (window + 1)
        return data.ewm(alpha=alpha).mean()
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index.
        
        Args:
            data: Input time series.
            window: Window size.
            
        Returns:
            RSI values.
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD (Moving Average Convergence Divergence).
        
        Args:
            data: Input time series.
            fast: Fast EMA window.
            slow: Slow EMA window.
            signal: Signal line EMA window.
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram).
        """
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands.
        
        Args:
            data: Input time series.
            window: Window size.
            num_std: Number of standard deviations.
            
        Returns:
            Tuple of (Upper band, Middle band, Lower band).
        """
        middle = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        
        return upper, middle, lower
    
    @staticmethod
    def stochastic(data: pd.Series, window: int = 14) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator.
        
        Args:
            data: Input time series.
            window: Window size.
            
        Returns:
            Tuple of (%K, %D).
        """
        low_min = data.rolling(window=window).min()
        high_max = data.rolling(window=window).max()
        
        k_percent = 100 * ((data - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=3).mean()
        
        return k_percent, d_percent


class EconomicFeatures:
    """Economic-specific feature engineering."""
    
    @staticmethod
    def growth_rate(data: pd.Series, periods: int = 1) -> pd.Series:
        """Calculate growth rate.
        
        Args:
            data: Input time series.
            periods: Number of periods for growth calculation.
            
        Returns:
            Growth rate values.
        """
        return data.pct_change(periods=periods)
    
    @staticmethod
    def year_over_year(data: pd.Series) -> pd.Series:
        """Calculate year-over-year change.
        
        Args:
            data: Input time series.
            
        Returns:
            YoY change values.
        """
        return data.pct_change(periods=252)  # Assuming daily data
    
    @staticmethod
    def seasonal_decomposition(data: pd.Series, model: str = "additive") -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Seasonal decomposition.
        
        Args:
            data: Input time series.
            model: Decomposition model ("additive" or "multiplicative").
            
        Returns:
            Tuple of (trend, seasonal, residual).
        """
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        decomposition = seasonal_decompose(data, model=model, period=252)
        
        return decomposition.trend, decomposition.seasonal, decomposition.resid
    
    @staticmethod
    def volatility(data: pd.Series, window: int = 30) -> pd.Series:
        """Calculate rolling volatility.
        
        Args:
            data: Input time series.
            window: Window size.
            
        Returns:
            Volatility values.
        """
        returns = data.pct_change()
        return returns.rolling(window=window).std() * np.sqrt(252)
    
    @staticmethod
    def skewness(data: pd.Series, window: int = 30) -> pd.Series:
        """Calculate rolling skewness.
        
        Args:
            data: Input time series.
            window: Window size.
            
        Returns:
            Skewness values.
        """
        returns = data.pct_change()
        return returns.rolling(window=window).skew()
    
    @staticmethod
    def kurtosis(data: pd.Series, window: int = 30) -> pd.Series:
        """Calculate rolling kurtosis.
        
        Args:
            data: Input time series.
            window: Window size.
            
        Returns:
            Kurtosis values.
        """
        returns = data.pct_change()
        return returns.rolling(window=window).kurt()


class TimeFeatures:
    """Time-based feature engineering."""
    
    @staticmethod
    def create_time_features(data: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features.
        
        Args:
            data: Input DataFrame with datetime index.
            
        Returns:
            DataFrame with time features.
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")
        
        features = data.copy()
        
        # Basic time features
        features["year"] = data.index.year
        features["month"] = data.index.month
        features["day"] = data.index.day
        features["dayofweek"] = data.index.dayofweek
        features["dayofyear"] = data.index.dayofyear
        features["quarter"] = data.index.quarter
        features["week"] = data.index.isocalendar().week
        
        # Cyclical encoding
        features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
        features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)
        features["day_sin"] = np.sin(2 * np.pi * features["day"] / 31)
        features["day_cos"] = np.cos(2 * np.pi * features["day"] / 31)
        features["dayofweek_sin"] = np.sin(2 * np.pi * features["dayofweek"] / 7)
        features["dayofweek_cos"] = np.cos(2 * np.pi * features["dayofweek"] / 7)
        
        # Business day features
        features["is_business_day"] = data.index.isin(pd.bdate_range(data.index[0], data.index[-1]))
        features["is_month_end"] = data.index.is_month_end
        features["is_quarter_end"] = data.index.is_quarter_end
        features["is_year_end"] = data.index.is_year_end
        
        return features
    
    @staticmethod
    def create_lag_features(
        data: pd.DataFrame,
        columns: List[str],
        lags: List[int],
    ) -> pd.DataFrame:
        """Create lagged features.
        
        Args:
            data: Input DataFrame.
            columns: Columns to create lags for.
            lags: List of lag periods.
            
        Returns:
            DataFrame with lagged features.
        """
        features = data.copy()
        
        for col in columns:
            for lag in lags:
                features[f"{col}_lag_{lag}"] = data[col].shift(lag)
        
        return features
    
    @staticmethod
    def create_lead_features(
        data: pd.DataFrame,
        columns: List[str],
        leads: List[int],
    ) -> pd.DataFrame:
        """Create lead features (future values).
        
        Args:
            data: Input DataFrame.
            columns: Columns to create leads for.
            leads: List of lead periods.
            
        Returns:
            DataFrame with lead features.
        """
        features = data.copy()
        
        for col in columns:
            for lead in leads:
                features[f"{col}_lead_{lead}"] = data[col].shift(-lead)
        
        return features


class FeatureSelector:
    """Feature selection utilities."""
    
    @staticmethod
    def correlation_filter(
        data: pd.DataFrame,
        threshold: float = 0.95,
        target_col: Optional[str] = None,
    ) -> List[str]:
        """Filter highly correlated features.
        
        Args:
            data: Input DataFrame.
            threshold: Correlation threshold.
            target_col: Target column (optional).
            
        Returns:
            List of selected feature names.
        """
        corr_matrix = data.corr().abs()
        
        # Create a mask for highly correlated pairs
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        
        # Keep target column if specified
        if target_col and target_col in to_drop:
            to_drop.remove(target_col)
        
        # Return features to keep
        return [col for col in data.columns if col not in to_drop]
    
    @staticmethod
    def variance_filter(data: pd.DataFrame, threshold: float = 0.01) -> List[str]:
        """Filter low variance features.
        
        Args:
            data: Input DataFrame.
            threshold: Variance threshold.
            
        Returns:
            List of selected feature names.
        """
        variances = data.var()
        return variances[variances > threshold].index.tolist()
    
    @staticmethod
    def mutual_information_filter(
        data: pd.DataFrame,
        target: pd.Series,
        k: int = 10,
    ) -> List[str]:
        """Filter features using mutual information.
        
        Args:
            data: Input DataFrame.
            target: Target variable.
            k: Number of top features to select.
            
        Returns:
            List of selected feature names.
        """
        from sklearn.feature_selection import mutual_info_regression
        
        # Calculate mutual information
        mi_scores = mutual_info_regression(data, target, random_state=42)
        
        # Get top k features
        feature_scores = pd.Series(mi_scores, index=data.columns)
        top_features = feature_scores.nlargest(k).index.tolist()
        
        return top_features


class FeatureEngineer:
    """Main feature engineering class."""
    
    def __init__(self, config: dict):
        """Initialize feature engineer.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.technical_indicators = TechnicalIndicators()
        self.economic_features = EconomicFeatures()
        self.time_features = TimeFeatures()
        self.feature_selector = FeatureSelector()
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create all features based on configuration.
        
        Args:
            data: Input DataFrame.
            
        Returns:
            DataFrame with engineered features.
        """
        features = data.copy()
        
        # Technical indicators
        if self.config.get("technical_indicators", {}).get("enabled", False):
            features = self._add_technical_indicators(features)
        
        # Economic features
        if self.config.get("economic_features", {}).get("enabled", False):
            features = self._add_economic_features(features)
        
        # Time features
        if self.config.get("time_features", {}).get("enabled", False):
            features = self._add_time_features(features)
        
        # Lag features
        if self.config.get("lag_features", {}).get("enabled", False):
            features = self._add_lag_features(features)
        
        # Feature selection
        if self.config.get("feature_selection", {}).get("enabled", False):
            features = self._apply_feature_selection(features)
        
        return features
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators."""
        features = data.copy()
        
        for col in data.columns:
            if col in ["date", "year", "month", "day"]:
                continue
            
            # SMA
            if self.config["technical_indicators"].get("sma", False):
                windows = self.config["technical_indicators"]["sma"]["windows"]
                for window in windows:
                    features[f"{col}_sma_{window}"] = self.technical_indicators.sma(data[col], window)
            
            # EMA
            if self.config["technical_indicators"].get("ema", False):
                windows = self.config["technical_indicators"]["ema"]["windows"]
                for window in windows:
                    features[f"{col}_ema_{window}"] = self.technical_indicators.ema(data[col], window)
            
            # RSI
            if self.config["technical_indicators"].get("rsi", False):
                window = self.config["technical_indicators"]["rsi"]["window"]
                features[f"{col}_rsi_{window}"] = self.technical_indicators.rsi(data[col], window)
        
        return features
    
    def _add_economic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add economic features."""
        features = data.copy()
        
        for col in data.columns:
            if col in ["date", "year", "month", "day"]:
                continue
            
            # Growth rates
            if self.config["economic_features"].get("growth_rates", False):
                periods = self.config["economic_features"]["growth_rates"]["periods"]
                for period in periods:
                    features[f"{col}_growth_{period}"] = self.economic_features.growth_rate(data[col], period)
            
            # Volatility
            if self.config["economic_features"].get("volatility", False):
                window = self.config["economic_features"]["volatility"]["window"]
                features[f"{col}_volatility_{window}"] = self.economic_features.volatility(data[col], window)
        
        return features
    
    def _add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add time features."""
        return self.time_features.create_time_features(data)
    
    def _add_lag_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add lag features."""
        columns = self.config["lag_features"]["columns"]
        lags = self.config["lag_features"]["lags"]
        
        return self.time_features.create_lag_features(data, columns, lags)
    
    def _apply_feature_selection(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply feature selection."""
        # Correlation filter
        if self.config["feature_selection"].get("correlation_filter", False):
            threshold = self.config["feature_selection"]["correlation_filter"]["threshold"]
            selected_features = self.feature_selector.correlation_filter(data, threshold)
            data = data[selected_features]
        
        # Variance filter
        if self.config["feature_selection"].get("variance_filter", False):
            threshold = self.config["feature_selection"]["variance_filter"]["threshold"]
            selected_features = self.feature_selector.variance_filter(data, threshold)
            data = data[selected_features]
        
        return data
