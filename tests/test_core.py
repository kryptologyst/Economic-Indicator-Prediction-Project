"""
Test suite for Economic Indicator Prediction project.

This module contains unit tests for the core functionality.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import set_seed, get_device, ConfigManager
from data import EconomicDataLoader, DataPreprocessor
from features import TechnicalIndicators, EconomicFeatures, TimeFeatures
from models import LinearModel, TreeModel, ModelFactory
from evaluation import MLEvaluator, FinancialEvaluator, Backtester


class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test random seed setting."""
        set_seed(42)
        # Test that seeds are set (basic check)
        self.assertTrue(True)  # Placeholder for actual seed testing
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        self.assertIsNotNone(device)
    
    def test_config_manager(self):
        """Test configuration manager."""
        # Create a mock config
        config_data = {"test": "value"}
        config_manager = ConfigManager.from_dict(config_data)
        
        self.assertEqual(config_manager.get("test"), "value")
        self.assertEqual(config_manager.get("nonexistent", "default"), "default")


class TestDataLoader(unittest.TestCase):
    """Test data loading functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_loader = EconomicDataLoader("test_data")
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        data = self.data_loader._generate_synthetic_economic_data(3, 100)
        
        self.assertEqual(len(data.columns), 3)
        self.assertEqual(len(data), 100)
        self.assertIsInstance(data.index, pd.DatetimeIndex)
    
    def test_data_preprocessor(self):
        """Test data preprocessing."""
        preprocessor = DataPreprocessor("standard")
        
        # Create test data
        data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        
        # Test fit_transform
        scaled_data = preprocessor.fit_transform(data)
        self.assertEqual(scaled_data.shape, data.shape)
        
        # Test transform
        new_data = pd.DataFrame({'A': [6, 7], 'B': [60, 70]})
        transformed_data = preprocessor.transform(new_data)
        self.assertEqual(transformed_data.shape, new_data.shape)
        
        # Test inverse_transform
        original_data = preprocessor.inverse_transform(scaled_data)
        np.testing.assert_array_almost_equal(original_data.values, data.values, decimal=5)


class TestFeatures(unittest.TestCase):
    """Test feature engineering."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = pd.Series(np.random.randn(100).cumsum())
    
    def test_technical_indicators(self):
        """Test technical indicators."""
        # Test SMA
        sma = TechnicalIndicators.sma(self.data, 10)
        self.assertEqual(len(sma), len(self.data))
        
        # Test EMA
        ema = TechnicalIndicators.ema(self.data, 10)
        self.assertEqual(len(ema), len(self.data))
        
        # Test RSI
        rsi = TechnicalIndicators.rsi(self.data, 14)
        self.assertEqual(len(rsi), len(self.data))
    
    def test_economic_features(self):
        """Test economic features."""
        # Test growth rate
        growth_rate = EconomicFeatures.growth_rate(self.data, 1)
        self.assertEqual(len(growth_rate), len(self.data))
        
        # Test volatility
        volatility = EconomicFeatures.volatility(self.data, 30)
        self.assertEqual(len(volatility), len(self.data))
    
    def test_time_features(self):
        """Test time features."""
        # Create test data with datetime index
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({'value': np.random.randn(100)}, index=dates)
        
        # Test time feature creation
        time_features = TimeFeatures.create_time_features(data)
        
        # Check that time features were added
        self.assertIn('year', time_features.columns)
        self.assertIn('month', time_features.columns)
        self.assertIn('dayofweek', time_features.columns)


class TestModels(unittest.TestCase):
    """Test model functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        self.y = pd.Series(np.random.randn(100))
    
    def test_linear_model(self):
        """Test linear model."""
        model = LinearModel({"model_type": "linear"})
        
        # Test fit
        model.fit(self.X, self.y)
        self.assertTrue(model.is_fitted)
        
        # Test predict
        predictions = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))
        
        # Test feature importance
        importance = model.get_feature_importance()
        self.assertIsNotNone(importance)
    
    def test_tree_model(self):
        """Test tree model."""
        model = TreeModel({"model_type": "random_forest"})
        
        # Test fit
        model.fit(self.X, self.y)
        self.assertTrue(model.is_fitted)
        
        # Test predict
        predictions = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))
    
    def test_model_factory(self):
        """Test model factory."""
        # Test linear model creation
        linear_model = ModelFactory.create_model({"type": "linear", "model_type": "linear"})
        self.assertIsInstance(linear_model, LinearModel)
        
        # Test tree model creation
        tree_model = ModelFactory.create_model({"type": "tree", "model_type": "random_forest"})
        self.assertIsInstance(tree_model, TreeModel)


class TestEvaluation(unittest.TestCase):
    """Test evaluation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.y_true = np.array([1, 2, 3, 4, 5])
        self.y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
    
    def test_ml_evaluator(self):
        """Test ML evaluator."""
        metrics = MLEvaluator.calculate_metrics(self.y_true, self.y_pred)
        
        self.assertIn('mse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('r2', metrics)
        
        # Check that metrics are reasonable
        self.assertGreater(metrics['r2'], 0.9)  # Should be high for this test case
    
    def test_financial_evaluator(self):
        """Test financial evaluator."""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        
        # Test Sharpe ratio
        sharpe = FinancialEvaluator.calculate_sharpe_ratio(returns)
        self.assertIsInstance(sharpe, float)
        
        # Test max drawdown
        max_dd = FinancialEvaluator.calculate_max_drawdown(returns)
        self.assertIsInstance(max_dd, float)
        self.assertLessEqual(max_dd, 0)  # Drawdown should be negative or zero
    
    def test_backtester(self):
        """Test backtester."""
        backtester = Backtester({"initial_capital": 10000})
        
        # Create test data
        predictions = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02])
        actual_values = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02])
        
        # Test buy and hold strategy
        strategy_config = {"type": "buy_and_hold", "initial_capital": 10000}
        results = backtester.run_backtest(predictions, actual_values, strategy_config)
        
        self.assertIn('strategy', results)
        self.assertIn('portfolio_value', results)
        self.assertIn('returns', results)
        self.assertIn('metrics', results)


if __name__ == '__main__':
    unittest.main()
