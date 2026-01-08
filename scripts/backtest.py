#!/usr/bin/env python3
"""
Backtesting script for Economic Indicator Prediction models.

This script runs comprehensive backtests on trained models and evaluates
their performance across different strategies and market conditions.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import set_seed, setup_logging, load_config
from data import EconomicDataLoader, DataPreprocessor
from features import FeatureEngineer
from models import ModelFactory
from evaluation import Backtester, FinancialEvaluator


def main():
    """Main backtesting function."""
    parser = argparse.ArgumentParser(description="Run backtests on Economic Indicator Prediction models")
    parser.add_argument("--config", type=str, default="configs/backtest.yaml", help="Backtest configuration file")
    parser.add_argument("--model-path", type=str, help="Path to trained model")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="assets/results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level)
    
    logger.info("Starting Economic Indicator Prediction backtesting")
    logger.info(f"Configuration: {args.config}")
    
    try:
        # Load data
        logger.info("Loading data...")
        data_loader = EconomicDataLoader(args.data_dir)
        
        # Load economic data
        data = data_loader._generate_synthetic_economic_data(4)
        
        # Create features
        feature_engineer = FeatureEngineer({
            "technical_indicators": {"enabled": True},
            "economic_features": {"enabled": True},
            "time_features": {"enabled": True}
        })
        features_data = feature_engineer.create_features(data)
        
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
        
        # Scale features
        preprocessor = DataPreprocessor("standard")
        features_scaled = preprocessor.fit_transform(features)
        
        # Load or train model
        if args.model_path and Path(args.model_path).exists():
            import joblib
            model = joblib.load(args.model_path)
            logger.info(f"Loaded model from {args.model_path}")
        else:
            # Train a simple model for demonstration
            from models import LinearModel
            model = LinearModel({"model_type": "linear"})
            model.fit(features_scaled, target)
            logger.info("Trained new model for backtesting")
        
        # Make predictions
        predictions = model.predict(features_scaled)
        predictions_series = pd.Series(predictions, index=target.index)
        
        # Initialize backtester
        backtester = Backtester(config.backtest)
        
        # Run backtests for different strategies
        logger.info("Running backtests...")
        
        results = {}
        for strategy_config in config.backtest.strategies:
            strategy_name = strategy_config.name
            logger.info(f"Running {strategy_name} strategy...")
            
            strategy_results = backtester.run_backtest(
                predictions_series,
                target,
                strategy_config
            )
            
            results[strategy_name] = strategy_results
            
            # Log strategy metrics
            metrics = strategy_results["metrics"]
            logger.info(f"{strategy_name} Results:")
            logger.info(f"  Total Return: {metrics['total_return']:.2%}")
            logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
            logger.info(f"  Hit Rate: {metrics['hit_rate']:.2%}")
        
        # Create comparison plot
        logger.info("Creating comparison plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Backtest Results Comparison", fontsize=16)
        
        # Portfolio value comparison
        ax1 = axes[0, 0]
        for strategy_name, strategy_results in results.items():
            portfolio_value = strategy_results["portfolio_value"]
            ax1.plot(portfolio_value.index, portfolio_value.values, 
                    label=strategy_name, linewidth=2)
        ax1.set_title("Portfolio Value Over Time")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Returns comparison
        ax2 = axes[0, 1]
        for strategy_name, strategy_results in results.items():
            returns = strategy_results["returns"]
            ax2.plot(returns.index, returns.values, 
                    label=strategy_name, alpha=0.7)
        ax2.set_title("Returns Over Time")
        ax2.set_ylabel("Returns")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Metrics comparison
        ax3 = axes[1, 0]
        metrics_df = pd.DataFrame({
            strategy: results[strategy]["metrics"] 
            for strategy in results.keys()
        }).T
        
        # Select key metrics for comparison
        key_metrics = ["total_return", "sharpe_ratio", "max_drawdown", "hit_rate"]
        metrics_subset = metrics_df[key_metrics]
        
        # Normalize metrics for comparison (except hit_rate)
        normalized_metrics = metrics_subset.copy()
        for col in ["total_return", "sharpe_ratio", "max_drawdown"]:
            normalized_metrics[col] = (metrics_subset[col] - metrics_subset[col].min()) / \
                                    (metrics_subset[col].max() - metrics_subset[col].min())
        
        sns.heatmap(normalized_metrics.T, annot=True, cmap="RdYlBu_r", 
                   ax=ax3, cbar_kws={'label': 'Normalized Score'})
        ax3.set_title("Normalized Metrics Comparison")
        
        # Drawdown comparison
        ax4 = axes[1, 1]
        for strategy_name, strategy_results in results.items():
            portfolio_value = strategy_results["portfolio_value"]
            cumulative = portfolio_value / portfolio_value.iloc[0]
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            
            ax4.fill_between(drawdown.index, drawdown.values, 0, 
                           alpha=0.3, label=strategy_name)
        ax4.set_title("Drawdown Comparison")
        ax4.set_ylabel("Drawdown")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plots
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_dir / "backtest_comparison.png", dpi=300, bbox_inches="tight")
        logger.info(f"Comparison plot saved to {output_dir / 'backtest_comparison.png'}")
        
        # Save detailed results
        detailed_results = {}
        for strategy_name, strategy_results in results.items():
            detailed_results[strategy_name] = {
                "metrics": strategy_results["metrics"],
                "portfolio_value": strategy_results["portfolio_value"].to_dict(),
                "returns": strategy_results["returns"].to_dict()
            }
        
        import json
        with open(output_dir / "backtest_results.json", "w") as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        logger.info(f"Detailed results saved to {output_dir / 'backtest_results.json'}")
        
        # Create summary table
        summary_df = pd.DataFrame({
            strategy: results[strategy]["metrics"] 
            for strategy in results.keys()
        }).T
        
        logger.info("\nBacktest Summary:")
        logger.info("=" * 50)
        logger.info(summary_df.round(4).to_string())
        
        # Save summary
        summary_df.to_csv(output_dir / "backtest_summary.csv")
        logger.info(f"Summary saved to {output_dir / 'backtest_summary.csv'}")
        
        logger.info("Backtesting completed successfully!")
        
    except Exception as e:
        logger.error(f"Backtesting failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
