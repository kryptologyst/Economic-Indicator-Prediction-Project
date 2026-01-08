"""
Machine learning models for economic indicator prediction.

This module implements various forecasting models including traditional time series
models, tree-based methods, and neural networks for economic forecasting.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Base class for all forecasting models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize model.
        
        Args:
            config: Model configuration.
        """
        self.config = config
        self.model = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model.
        
        Args:
            X: Feature matrix.
            y: Target variable.
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Predictions.
        """
        pass
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance if available.
        
        Returns:
            Feature importance scores.
        """
        return None


class LinearModel(BaseModel):
    """Linear regression models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize linear model.
        
        Args:
            config: Model configuration.
        """
        super().__init__(config)
        
        model_type = config.get("model_type", "linear")
        
        if model_type == "linear":
            self.model = LinearRegression()
        elif model_type == "ridge":
            alpha = config.get("alpha", 1.0)
            self.model = Ridge(alpha=alpha)
        elif model_type == "lasso":
            alpha = config.get("alpha", 1.0)
            self.model = Lasso(alpha=alpha)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the linear model."""
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info(f"Fitted {self.config['model_type']} model")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance (coefficients)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        return pd.Series(
            self.model.coef_,
            index=self.model.feature_names_in_ if hasattr(self.model, 'feature_names_in_') else range(len(self.model.coef_))
        )


class TreeModel(BaseModel):
    """Tree-based models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize tree model.
        
        Args:
            config: Model configuration.
        """
        super().__init__(config)
        
        model_type = config.get("model_type", "random_forest")
        
        if model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=config.get("n_estimators", 100),
                max_depth=config.get("max_depth", None),
                min_samples_split=config.get("min_samples_split", 2),
                min_samples_leaf=config.get("min_samples_leaf", 1),
                random_state=config.get("random_state", 42)
            )
        elif model_type == "xgboost":
            self.model = xgb.XGBRegressor(
                n_estimators=config.get("n_estimators", 100),
                max_depth=config.get("max_depth", 6),
                learning_rate=config.get("learning_rate", 0.1),
                random_state=config.get("random_state", 42)
            )
        elif model_type == "lightgbm":
            self.model = lgb.LGBMRegressor(
                n_estimators=config.get("n_estimators", 100),
                max_depth=config.get("max_depth", 6),
                learning_rate=config.get("learning_rate", 0.1),
                random_state=config.get("random_state", 42),
                verbose=-1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the tree model."""
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info(f"Fitted {self.config['model_type']} model")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            return pd.Series(
                self.model.feature_importances_,
                index=self.model.feature_names_in_ if hasattr(self.model, 'feature_names_in_') else range(len(self.model.feature_importances_))
            )
        return None


class LSTMModel(nn.Module):
    """LSTM neural network for time series forecasting."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
    ):
        """Initialize LSTM model.
        
        Args:
            input_size: Number of input features.
            hidden_size: Hidden layer size.
            num_layers: Number of LSTM layers.
            dropout: Dropout rate.
            output_size: Output size.
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        out = out[:, -1, :]
        
        # Apply dropout and fully connected layer
        out = self.dropout(out)
        out = self.fc(out)
        
        return out


class NeuralNetworkModel(BaseModel):
    """Neural network model wrapper."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize neural network model.
        
        Args:
            config: Model configuration.
        """
        super().__init__(config)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        
        # Training parameters
        self.epochs = config.get("epochs", 100)
        self.batch_size = config.get("batch_size", 32)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.patience = config.get("patience", 10)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the neural network."""
        # Convert to tensors
        X_tensor = torch.FloatTensor(X.values)
        y_tensor = torch.FloatTensor(y.values).unsqueeze(1)
        
        # Create model
        self.model = LSTMModel(
            input_size=X.shape[1],
            hidden_size=self.config.get("hidden_size", 64),
            num_layers=self.config.get("num_layers", 2),
            dropout=self.config.get("dropout", 0.2),
            output_size=1
        ).to(self.device)
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )
        
        # Move data to device
        X_tensor = X_tensor.to(self.device)
        y_tensor = y_tensor.to(self.device)
        
        # Training loop
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Forward pass
            outputs = self.model(X_tensor.unsqueeze(1))
            loss = self.criterion(outputs, y_tensor)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        self.is_fitted = True
        logger.info("Fitted LSTM model")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.values).to(self.device)
            predictions = self.model(X_tensor.unsqueeze(1))
            return predictions.cpu().numpy().flatten()


class ARIMAModel(BaseModel):
    """ARIMA model for time series forecasting."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize ARIMA model.
        
        Args:
            config: Model configuration.
        """
        super().__init__(config)
        
        try:
            from pmdarima import auto_arima
            self.auto_arima = auto_arima
        except ImportError:
            logger.warning("pmdarima not available, using basic ARIMA")
            self.auto_arima = None
        
        self.model = None
        self.order = config.get("order", (1, 1, 1))
        self.seasonal_order = config.get("seasonal_order", (0, 0, 0, 0))
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the ARIMA model."""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            
            if self.auto_arima and self.config.get("auto_arima", False):
                # Use auto_arima for automatic parameter selection
                self.model = self.auto_arima(
                    y,
                    start_p=1, start_q=1,
                    max_p=5, max_q=5,
                    seasonal=False,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore'
                )
            else:
                # Use manual parameters
                self.model = ARIMA(y, order=self.order)
                self.model = self.model.fit()
            
            self.is_fitted = True
            logger.info("Fitted ARIMA model")
            
        except ImportError:
            logger.error("statsmodels not available for ARIMA")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if hasattr(self.model, 'predict'):
            # For auto_arima
            predictions = self.model.predict(n_periods=len(X))
        else:
            # For statsmodels ARIMA
            predictions = self.model.forecast(steps=len(X))
        
        return predictions.values if hasattr(predictions, 'values') else predictions


class ModelEnsemble(BaseModel):
    """Ensemble of multiple models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize ensemble model.
        
        Args:
            config: Model configuration.
        """
        super().__init__(config)
        
        self.models = []
        self.weights = config.get("weights", None)
        
        # Create individual models
        for model_config in config.get("models", []):
            model_type = model_config.get("type")
            
            if model_type == "linear":
                model = LinearModel(model_config)
            elif model_type == "tree":
                model = TreeModel(model_config)
            elif model_type == "neural_network":
                model = NeuralNetworkModel(model_config)
            elif model_type == "arima":
                model = ARIMAModel(model_config)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            self.models.append(model)
        
        # Set equal weights if not specified
        if self.weights is None:
            self.weights = [1.0 / len(self.models)] * len(self.models)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit all models in the ensemble."""
        for model in self.models:
            model.fit(X, y)
        
        self.is_fitted = True
        logger.info(f"Fitted ensemble with {len(self.models)} models")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted average
        ensemble_pred = np.zeros(len(X))
        for i, (pred, weight) in enumerate(zip(predictions, self.weights)):
            ensemble_pred += weight * pred
        
        return ensemble_pred
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get ensemble feature importance."""
        importances = []
        
        for model in self.models:
            imp = model.get_feature_importance()
            if imp is not None:
                importances.append(imp)
        
        if not importances:
            return None
        
        # Average feature importance across models
        avg_importance = pd.concat(importances, axis=1).mean(axis=1)
        return avg_importance


class ModelFactory:
    """Factory for creating models."""
    
    @staticmethod
    def create_model(config: Dict[str, Any]) -> BaseModel:
        """Create a model based on configuration.
        
        Args:
            config: Model configuration.
            
        Returns:
            Model instance.
        """
        model_type = config.get("type")
        
        if model_type == "linear":
            return LinearModel(config)
        elif model_type == "tree":
            return TreeModel(config)
        elif model_type == "neural_network":
            return NeuralNetworkModel(config)
        elif model_type == "arima":
            return ARIMAModel(config)
        elif model_type == "ensemble":
            return ModelEnsemble(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


def evaluate_model(
    model: BaseModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metrics: List[str] = ["mse", "mae", "rmse"]
) -> Dict[str, float]:
    """Evaluate model performance.
    
    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test targets.
        metrics: List of metrics to calculate.
        
    Returns:
        Dictionary of metric scores.
    """
    predictions = model.predict(X_test)
    
    results = {}
    
    if "mse" in metrics:
        results["mse"] = mean_squared_error(y_test, predictions)
    
    if "mae" in metrics:
        results["mae"] = mean_absolute_error(y_test, predictions)
    
    if "rmse" in metrics:
        results["rmse"] = np.sqrt(mean_squared_error(y_test, predictions))
    
    if "r2" in metrics:
        from sklearn.metrics import r2_score
        results["r2"] = r2_score(y_test, predictions)
    
    return results
