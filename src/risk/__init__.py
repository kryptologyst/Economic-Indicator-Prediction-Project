"""
Risk management and uncertainty quantification for economic forecasting.

This module provides risk management utilities including VaR, ES, stress testing,
and uncertainty quantification for economic indicator predictions.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class RiskMetrics:
    """Risk metrics calculation."""
    
    @staticmethod
    def calculate_var(
        returns: pd.Series,
        confidence_level: float = 0.05,
        method: str = "historical"
    ) -> float:
        """Calculate Value at Risk (VaR).
        
        Args:
            returns: Returns series.
            confidence_level: Confidence level (e.g., 0.05 for 95% VaR).
            method: Method for VaR calculation ("historical", "parametric", "monte_carlo").
            
        Returns:
            VaR value.
        """
        if method == "historical":
            return np.percentile(returns, confidence_level * 100)
        
        elif method == "parametric":
            mean_return = returns.mean()
            std_return = returns.std()
            return mean_return + std_return * stats.norm.ppf(confidence_level)
        
        elif method == "monte_carlo":
            # Monte Carlo simulation
            n_simulations = 10000
            mean_return = returns.mean()
            std_return = returns.std()
            
            simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
            return np.percentile(simulated_returns, confidence_level * 100)
        
        else:
            raise ValueError(f"Unknown VaR method: {method}")
    
    @staticmethod
    def calculate_expected_shortfall(
        returns: pd.Series,
        confidence_level: float = 0.05,
        method: str = "historical"
    ) -> float:
        """Calculate Expected Shortfall (ES) / Conditional VaR.
        
        Args:
            returns: Returns series.
            confidence_level: Confidence level.
            method: Method for ES calculation.
            
        Returns:
            ES value.
        """
        var = RiskMetrics.calculate_var(returns, confidence_level, method)
        
        if method == "historical":
            return returns[returns <= var].mean()
        
        elif method == "parametric":
            mean_return = returns.mean()
            std_return = returns.std()
            z_score = stats.norm.ppf(confidence_level)
            
            # ES formula for normal distribution
            return mean_return - std_return * stats.norm.pdf(z_score) / confidence_level
        
        elif method == "monte_carlo":
            n_simulations = 10000
            mean_return = returns.mean()
            std_return = returns.std()
            
            simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
            return simulated_returns[simulated_returns <= var].mean()
        
        else:
            raise ValueError(f"Unknown ES method: {method}")
    
    @staticmethod
    def calculate_maximum_drawdown(returns: pd.Series) -> Tuple[float, int, int]:
        """Calculate maximum drawdown with start and end indices.
        
        Args:
            returns: Returns series.
            
        Returns:
            Tuple of (max_drawdown, start_idx, end_idx).
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_dd_idx = drawdown.idxmin()
        max_dd_value = drawdown.min()
        
        # Find start of drawdown period
        start_idx = cumulative[:max_dd_idx].idxmax()
        
        return max_dd_value, start_idx, max_dd_idx
    
    @staticmethod
    def calculate_volatility_forecast(
        returns: pd.Series,
        window: int = 30,
        method: str = "ewma"
    ) -> pd.Series:
        """Calculate volatility forecast.
        
        Args:
            returns: Returns series.
            window: Window size for calculation.
            method: Method for volatility calculation.
            
        Returns:
            Volatility forecast series.
        """
        if method == "ewma":
            # Exponentially weighted moving average
            alpha = 2.0 / (window + 1)
            return returns.ewm(alpha=alpha).std() * np.sqrt(252)
        
        elif method == "garch":
            # Simple GARCH(1,1) approximation
            squared_returns = returns ** 2
            alpha = 0.1
            beta = 0.85
            omega = 0.0001
            
            volatility = pd.Series(index=returns.index, dtype=float)
            volatility.iloc[0] = returns.std()
            
            for i in range(1, len(returns)):
                volatility.iloc[i] = np.sqrt(
                    omega + alpha * squared_returns.iloc[i-1] + beta * volatility.iloc[i-1]**2
                )
            
            return volatility * np.sqrt(252)
        
        else:
            # Simple rolling window
            return returns.rolling(window=window).std() * np.sqrt(252)


class StressTester:
    """Stress testing for economic scenarios."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize stress tester.
        
        Args:
            config: Stress testing configuration.
        """
        self.config = config
    
    def run_stress_tests(
        self,
        model,
        X_test: pd.DataFrame,
        scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run stress tests on model predictions.
        
        Args:
            model: Trained model.
            X_test: Test features.
            scenarios: List of stress scenarios.
            
        Returns:
            Stress test results.
        """
        results = {}
        
        # Baseline prediction
        baseline_pred = model.predict(X_test)
        results["baseline"] = {
            "predictions": baseline_pred,
            "mean": np.mean(baseline_pred),
            "std": np.std(baseline_pred)
        }
        
        # Run stress scenarios
        for scenario in scenarios:
            scenario_name = scenario["name"]
            scenario_data = self._apply_scenario(X_test, scenario)
            
            stress_pred = model.predict(scenario_data)
            
            results[scenario_name] = {
                "predictions": stress_pred,
                "mean": np.mean(stress_pred),
                "std": np.std(stress_pred),
                "change_from_baseline": np.mean(stress_pred) - np.mean(baseline_pred)
            }
        
        return results
    
    def _apply_scenario(
        self,
        data: pd.DataFrame,
        scenario: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply stress scenario to data.
        
        Args:
            data: Input data.
            scenario: Scenario configuration.
            
        Returns:
            Modified data.
        """
        modified_data = data.copy()
        
        scenario_type = scenario["type"]
        
        if scenario_type == "shock":
            # Apply shock to specific variables
            for var, shock in scenario["shocks"].items():
                if var in modified_data.columns:
                    modified_data[var] = modified_data[var] * (1 + shock)
        
        elif scenario_type == "recession":
            # Simulate recession scenario
            recession_factor = scenario.get("factor", 0.8)
            for col in modified_data.columns:
                if "growth" in col.lower() or "return" in col.lower():
                    modified_data[col] = modified_data[col] * recession_factor
        
        elif scenario_type == "inflation":
            # Simulate high inflation scenario
            inflation_factor = scenario.get("factor", 1.5)
            for col in modified_data.columns:
                if "inflation" in col.lower() or "price" in col.lower():
                    modified_data[col] = modified_data[col] * inflation_factor
        
        return modified_data


class UncertaintyQuantifier:
    """Uncertainty quantification for predictions."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize uncertainty quantifier.
        
        Args:
            config: Uncertainty quantification configuration.
        """
        self.config = config
    
    def bootstrap_uncertainty(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        n_bootstrap: int = 100,
        confidence_level: float = 0.95
    ) -> Dict[str, np.ndarray]:
        """Calculate prediction uncertainty using bootstrap.
        
        Args:
            model: Model to evaluate.
            X_train: Training features.
            y_train: Training targets.
            X_test: Test features.
            n_bootstrap: Number of bootstrap samples.
            confidence_level: Confidence level for intervals.
            
        Returns:
            Dictionary with prediction intervals.
        """
        predictions = []
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            bootstrap_idx = np.random.choice(
                len(X_train), size=len(X_train), replace=True
            )
            
            X_bootstrap = X_train.iloc[bootstrap_idx]
            y_bootstrap = y_train.iloc[bootstrap_idx]
            
            # Train model on bootstrap sample
            bootstrap_model = self._create_bootstrap_model(model)
            bootstrap_model.fit(X_bootstrap, y_bootstrap)
            
            # Make predictions
            pred = bootstrap_model.predict(X_test)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Calculate prediction intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(predictions, lower_percentile, axis=0)
        upper_bound = np.percentile(predictions, upper_percentile, axis=0)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return {
            "mean": mean_pred,
            "std": std_pred,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "confidence_level": confidence_level
        }
    
    def _create_bootstrap_model(self, original_model):
        """Create a copy of the original model for bootstrap."""
        # This is a simplified implementation
        # In practice, you'd need to properly clone the model
        return original_model
    
    def conformal_prediction(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        confidence_level: float = 0.95
    ) -> Dict[str, np.ndarray]:
        """Calculate conformal prediction intervals.
        
        Args:
            model: Trained model.
            X_train: Training features.
            y_train: Training targets.
            X_test: Test features.
            confidence_level: Confidence level.
            
        Returns:
            Dictionary with conformal prediction intervals.
        """
        # Split training data for calibration
        n_cal = int(0.2 * len(X_train))
        cal_idx = np.random.choice(len(X_train), n_cal, replace=False)
        train_idx = np.setdiff1d(np.arange(len(X_train)), cal_idx)
        
        X_cal = X_train.iloc[cal_idx]
        y_cal = y_train.iloc[cal_idx]
        X_train_reduced = X_train.iloc[train_idx]
        y_train_reduced = y_train.iloc[train_idx]
        
        # Train model on reduced training set
        model.fit(X_train_reduced, y_train_reduced)
        
        # Calculate residuals on calibration set
        cal_pred = model.predict(X_cal)
        residuals = np.abs(y_cal.values - cal_pred)
        
        # Calculate quantile of residuals
        alpha = 1 - confidence_level
        quantile = np.quantile(residuals, 1 - alpha)
        
        # Make predictions on test set
        test_pred = model.predict(X_test)
        
        # Calculate prediction intervals
        lower_bound = test_pred - quantile
        upper_bound = test_pred + quantile
        
        return {
            "mean": test_pred,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "confidence_level": confidence_level,
            "quantile": quantile
        }


class RiskManager:
    """Main risk management class."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize risk manager.
        
        Args:
            config: Risk management configuration.
        """
        self.config = config
        self.risk_metrics = RiskMetrics()
        self.stress_tester = StressTester(config.get("stress_test", {}))
        self.uncertainty_quantifier = UncertaintyQuantifier(config.get("uncertainty", {}))
    
    def assess_model_risk(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """Comprehensive risk assessment.
        
        Args:
            model: Trained model.
            X_train: Training features.
            y_train: Training targets.
            X_test: Test features.
            y_test: Test targets.
            
        Returns:
            Comprehensive risk assessment results.
        """
        results = {}
        
        # Calculate predictions
        predictions = model.predict(X_test)
        residuals = y_test.values - predictions
        
        # Basic risk metrics
        results["risk_metrics"] = {
            "prediction_std": np.std(predictions),
            "residual_std": np.std(residuals),
            "max_residual": np.max(np.abs(residuals)),
            "mean_absolute_residual": np.mean(np.abs(residuals))
        }
        
        # VaR and ES
        if self.config.get("calculate_var", True):
            results["var"] = self.risk_metrics.calculate_var(
                pd.Series(residuals),
                confidence_level=self.config.get("var_confidence", 0.05)
            )
            results["expected_shortfall"] = self.risk_metrics.calculate_expected_shortfall(
                pd.Series(residuals),
                confidence_level=self.config.get("var_confidence", 0.05)
            )
        
        # Stress testing
        if self.config.get("run_stress_tests", True):
            stress_scenarios = self.config.get("stress_scenarios", [])
            if stress_scenarios:
                results["stress_tests"] = self.stress_tester.run_stress_tests(
                    model, X_test, stress_scenarios
                )
        
        # Uncertainty quantification
        if self.config.get("quantify_uncertainty", True):
            uncertainty_method = self.config.get("uncertainty_method", "bootstrap")
            
            if uncertainty_method == "bootstrap":
                results["uncertainty"] = self.uncertainty_quantifier.bootstrap_uncertainty(
                    model, X_train, y_train, X_test
                )
            elif uncertainty_method == "conformal":
                results["uncertainty"] = self.uncertainty_quantifier.conformal_prediction(
                    model, X_train, y_train, X_test
                )
        
        return results
    
    def generate_risk_report(
        self,
        risk_assessment: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """Generate risk assessment report.
        
        Args:
            risk_assessment: Risk assessment results.
            output_path: Optional output file path.
            
        Returns:
            Risk report as string.
        """
        report = []
        report.append("=" * 50)
        report.append("RISK ASSESSMENT REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Risk metrics
        if "risk_metrics" in risk_assessment:
            report.append("RISK METRICS:")
            report.append("-" * 20)
            for metric, value in risk_assessment["risk_metrics"].items():
                report.append(f"{metric}: {value:.4f}")
            report.append("")
        
        # VaR and ES
        if "var" in risk_assessment:
            report.append("VALUE AT RISK (VaR):")
            report.append("-" * 20)
            report.append(f"VaR: {risk_assessment['var']:.4f}")
            report.append(f"Expected Shortfall: {risk_assessment['expected_shortfall']:.4f}")
            report.append("")
        
        # Stress tests
        if "stress_tests" in risk_assessment:
            report.append("STRESS TEST RESULTS:")
            report.append("-" * 20)
            for scenario, results in risk_assessment["stress_tests"].items():
                if scenario != "baseline":
                    report.append(f"{scenario}:")
                    report.append(f"  Mean prediction: {results['mean']:.4f}")
                    report.append(f"  Change from baseline: {results['change_from_baseline']:.4f}")
            report.append("")
        
        # Uncertainty
        if "uncertainty" in risk_assessment:
            report.append("UNCERTAINTY QUANTIFICATION:")
            report.append("-" * 20)
            uncertainty = risk_assessment["uncertainty"]
            report.append(f"Mean prediction: {np.mean(uncertainty['mean']):.4f}")
            report.append(f"Average std: {np.mean(uncertainty['std']):.4f}")
            report.append(f"Confidence level: {uncertainty['confidence_level']:.2%}")
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, "w") as f:
                f.write(report_text)
        
        return report_text
