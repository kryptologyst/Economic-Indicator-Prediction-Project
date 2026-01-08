"""
Evaluation metrics and backtesting for economic indicator prediction.

This module provides comprehensive evaluation metrics including both ML metrics
and financial metrics, as well as backtesting capabilities for economic forecasting.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)

logger = logging.getLogger(__name__)


class MLEvaluator:
    """Machine learning evaluation metrics."""
    
    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: List[str] = None
    ) -> Dict[str, float]:
        """Calculate ML evaluation metrics.
        
        Args:
            y_true: True values.
            y_pred: Predicted values.
            metrics: List of metrics to calculate.
            
        Returns:
            Dictionary of metric scores.
        """
        if metrics is None:
            metrics = ["mse", "mae", "rmse", "mape", "r2", "smape"]
        
        results = {}
        
        if "mse" in metrics:
            results["mse"] = mean_squared_error(y_true, y_pred)
        
        if "mae" in metrics:
            results["mae"] = mean_absolute_error(y_true, y_pred)
        
        if "rmse" in metrics:
            results["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
        
        if "mape" in metrics:
            results["mape"] = mean_absolute_percentage_error(y_true, y_pred)
        
        if "r2" in metrics:
            results["r2"] = r2_score(y_true, y_pred)
        
        if "smape" in metrics:
            results["smape"] = MLEvaluator._smape(y_true, y_pred)
        
        if "mase" in metrics:
            results["mase"] = MLEvaluator._mase(y_true, y_pred)
        
        return results
    
    @staticmethod
    def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error."""
        return np.mean(
            2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))
        ) * 100
    
    @staticmethod
    def _mase(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Scaled Error."""
        # Use naive forecast as baseline
        naive_forecast = np.roll(y_true, 1)
        naive_forecast[0] = y_true[0]
        
        mae_naive = mean_absolute_error(y_true[1:], naive_forecast[1:])
        mae_model = mean_absolute_error(y_true, y_pred)
        
        return mae_model / mae_naive if mae_naive != 0 else np.inf


class FinancialEvaluator:
    """Financial evaluation metrics for economic forecasting."""
    
    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        """Calculate returns from price series.
        
        Args:
            prices: Price series.
            
        Returns:
            Returns series.
        """
        return prices.pct_change().dropna()
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio.
        
        Args:
            returns: Returns series.
            risk_free_rate: Risk-free rate (annual).
            
        Returns:
            Sharpe ratio.
        """
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio.
        
        Args:
            returns: Returns series.
            risk_free_rate: Risk-free rate (annual).
            
        Returns:
            Sortino ratio.
        """
        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf
        
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
        return np.sqrt(252) * excess_returns.mean() / downside_deviation
    
    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> float:
        """Calculate maximum drawdown.
        
        Args:
            returns: Returns series.
            
        Returns:
            Maximum drawdown.
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return drawdown.min()
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series) -> float:
        """Calculate Calmar ratio.
        
        Args:
            returns: Returns series.
            
        Returns:
            Calmar ratio.
        """
        annual_return = returns.mean() * 252
        max_dd = abs(FinancialEvaluator.calculate_max_drawdown(returns))
        
        return annual_return / max_dd if max_dd != 0 else np.inf
    
    @staticmethod
    def calculate_information_ratio(
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """Calculate information ratio.
        
        Args:
            returns: Strategy returns.
            benchmark_returns: Benchmark returns.
            
        Returns:
            Information ratio.
        """
        excess_returns = returns - benchmark_returns
        tracking_error = excess_returns.std()
        
        return excess_returns.mean() / tracking_error if tracking_error != 0 else 0
    
    @staticmethod
    def calculate_hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate directional accuracy (hit rate).
        
        Args:
            y_true: True values.
            y_pred: Predicted values.
            
        Returns:
            Hit rate.
        """
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        
        return np.mean(true_direction == pred_direction)
    
    @staticmethod
    def calculate_volatility(returns: pd.Series) -> float:
        """Calculate annualized volatility.
        
        Args:
            returns: Returns series.
            
        Returns:
            Annualized volatility.
        """
        return returns.std() * np.sqrt(252)
    
    @staticmethod
    def calculate_skewness(returns: pd.Series) -> float:
        """Calculate skewness of returns.
        
        Args:
            returns: Returns series.
            
        Returns:
            Skewness.
        """
        return returns.skew()
    
    @staticmethod
    def calculate_kurtosis(returns: pd.Series) -> float:
        """Calculate kurtosis of returns.
        
        Args:
            returns: Returns series.
            
        Returns:
            Kurtosis.
        """
        return returns.kurtosis()


class Backtester:
    """Backtesting framework for economic forecasting strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize backtester.
        
        Args:
            config: Backtesting configuration.
        """
        self.config = config
        self.results = {}
    
    def run_backtest(
        self,
        predictions: pd.Series,
        actual_values: pd.Series,
        strategy_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run backtest on predictions.
        
        Args:
            predictions: Model predictions.
            actual_values: Actual values.
            strategy_config: Strategy configuration.
            
        Returns:
            Backtest results.
        """
        strategy_type = strategy_config.get("type", "buy_and_hold")
        
        if strategy_type == "buy_and_hold":
            return self._buy_and_hold_strategy(predictions, actual_values, strategy_config)
        elif strategy_type == "momentum":
            return self._momentum_strategy(predictions, actual_values, strategy_config)
        elif strategy_type == "mean_reversion":
            return self._mean_reversion_strategy(predictions, actual_values, strategy_config)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    def _buy_and_hold_strategy(
        self,
        predictions: pd.Series,
        actual_values: pd.Series,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Buy and hold strategy."""
        # Simple strategy: buy at the beginning, hold throughout
        initial_value = config.get("initial_capital", 10000)
        
        # Calculate returns
        returns = actual_values.pct_change().dropna()
        
        # Calculate portfolio value
        portfolio_value = initial_value * (1 + returns).cumprod()
        
        # Calculate metrics
        metrics = self._calculate_strategy_metrics(returns, portfolio_value)
        
        return {
            "strategy": "buy_and_hold",
            "portfolio_value": portfolio_value,
            "returns": returns,
            "metrics": metrics
        }
    
    def _momentum_strategy(
        self,
        predictions: pd.Series,
        actual_values: pd.Series,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Momentum strategy based on predictions."""
        initial_value = config.get("initial_capital", 10000)
        threshold = config.get("threshold", 0.02)
        
        # Generate signals based on predictions
        signals = np.where(predictions > threshold, 1, 0)
        signals = np.where(predictions < -threshold, -1, signals)
        
        # Calculate returns
        actual_returns = actual_values.pct_change().dropna()
        
        # Apply signals to returns
        strategy_returns = signals[1:] * actual_returns
        
        # Calculate portfolio value
        portfolio_value = initial_value * (1 + strategy_returns).cumprod()
        
        # Calculate metrics
        metrics = self._calculate_strategy_metrics(strategy_returns, portfolio_value)
        
        return {
            "strategy": "momentum",
            "portfolio_value": portfolio_value,
            "returns": strategy_returns,
            "signals": signals,
            "metrics": metrics
        }
    
    def _mean_reversion_strategy(
        self,
        predictions: pd.Series,
        actual_values: pd.Series,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Mean reversion strategy based on predictions."""
        initial_value = config.get("initial_capital", 10000)
        threshold = config.get("threshold", 0.02)
        
        # Generate signals (opposite of momentum)
        signals = np.where(predictions > threshold, -1, 0)
        signals = np.where(predictions < -threshold, 1, signals)
        
        # Calculate returns
        actual_returns = actual_values.pct_change().dropna()
        
        # Apply signals to returns
        strategy_returns = signals[1:] * actual_returns
        
        # Calculate portfolio value
        portfolio_value = initial_value * (1 + strategy_returns).cumprod()
        
        # Calculate metrics
        metrics = self._calculate_strategy_metrics(strategy_returns, portfolio_value)
        
        return {
            "strategy": "mean_reversion",
            "portfolio_value": portfolio_value,
            "returns": strategy_returns,
            "signals": signals,
            "metrics": metrics
        }
    
    def _calculate_strategy_metrics(
        self,
        returns: pd.Series,
        portfolio_value: pd.Series
    ) -> Dict[str, float]:
        """Calculate strategy performance metrics."""
        metrics = {}
        
        # Basic metrics
        metrics["total_return"] = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1
        metrics["annualized_return"] = returns.mean() * 252
        metrics["volatility"] = FinancialEvaluator.calculate_volatility(returns)
        metrics["sharpe_ratio"] = FinancialEvaluator.calculate_sharpe_ratio(returns)
        metrics["sortino_ratio"] = FinancialEvaluator.calculate_sortino_ratio(returns)
        metrics["max_drawdown"] = FinancialEvaluator.calculate_max_drawdown(returns)
        metrics["calmar_ratio"] = FinancialEvaluator.calculate_calmar_ratio(returns)
        metrics["skewness"] = FinancialEvaluator.calculate_skewness(returns)
        metrics["kurtosis"] = FinancialEvaluator.calculate_kurtosis(returns)
        
        # Hit rate
        metrics["hit_rate"] = np.mean(returns > 0)
        
        return metrics


class CrossValidator:
    """Cross-validation for time series data."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize cross-validator.
        
        Args:
            config: Cross-validation configuration.
        """
        self.config = config
    
    def time_series_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model,
        n_splits: int = 5,
        test_size: float = 0.2
    ) -> Dict[str, List[float]]:
        """Perform time series cross-validation.
        
        Args:
            X: Feature matrix.
            y: Target variable.
            model: Model to evaluate.
            n_splits: Number of CV splits.
            test_size: Test set size.
            
        Returns:
            Dictionary of CV scores.
        """
        from sklearn.model_selection import TimeSeriesSplit
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        scores = {
            "mse": [],
            "mae": [],
            "rmse": [],
            "r2": []
        }
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            scores["mse"].append(mse)
            scores["mae"].append(mae)
            scores["rmse"].append(rmse)
            scores["r2"].append(r2)
        
        return scores


class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize model evaluator.
        
        Args:
            config: Evaluation configuration.
        """
        self.config = config
        self.ml_evaluator = MLEvaluator()
        self.financial_evaluator = FinancialEvaluator()
        self.backtester = Backtester(config.get("backtest", {}))
        self.cross_validator = CrossValidator(config.get("cv", {}))
    
    def evaluate_model(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        run_cv: bool = True,
        run_backtest: bool = True
    ) -> Dict[str, Any]:
        """Comprehensive model evaluation.
        
        Args:
            model: Trained model.
            X_train: Training features.
            y_train: Training targets.
            X_test: Test features.
            y_test: Test targets.
            run_cv: Whether to run cross-validation.
            run_backtest: Whether to run backtest.
            
        Returns:
            Comprehensive evaluation results.
        """
        results = {}
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # ML metrics
        ml_metrics = self.ml_evaluator.calculate_metrics(y_test.values, y_pred)
        results["ml_metrics"] = ml_metrics
        
        # Cross-validation
        if run_cv:
            cv_scores = self.cross_validator.time_series_cv(X_train, y_train, model)
            results["cv_scores"] = cv_scores
        
        # Backtesting
        if run_backtest:
            predictions_series = pd.Series(y_pred, index=y_test.index)
            backtest_results = self.backtester.run_backtest(
                predictions_series,
                y_test,
                self.config.get("strategy", {"type": "buy_and_hold"})
            )
            results["backtest"] = backtest_results
        
        # Feature importance
        if hasattr(model, 'get_feature_importance'):
            feature_importance = model.get_feature_importance()
            if feature_importance is not None:
                results["feature_importance"] = feature_importance.to_dict()
        
        return results
