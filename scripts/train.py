#!/usr/bin/env python3
"""
Training script for Economic Indicator Prediction models.

This script trains various models for economic forecasting and evaluates their performance.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from omegaconf import DictConfig

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import set_seed, setup_logging, load_config, create_project_dirs
from data import EconomicDataLoader, DataPreprocessor
from features import FeatureEngineer
from models import ModelFactory, evaluate_model
from evaluation import ModelEvaluator
from risk import RiskManager


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Economic Indicator Prediction models")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Configuration file")
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
    log_level = "DEBUG" if args.verbose else config.logging.level
    logger = setup_logging(log_level, config.logging.log_file)
    
    logger.info("Starting Economic Indicator Prediction training")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Random seed: {args.seed}")
    
    # Create project directories
    create_project_dirs(Path.cwd())
    
    try:
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        data_loader = EconomicDataLoader(args.data_dir)
        
        # Load economic data
        if config.data.sources.fred.enabled:
            fred_data = data_loader.load_fred_data(
                config.data.sources.fred.series_ids,
                start_date="2000-01-01"
            )
            logger.info(f"Loaded FRED data: {fred_data.shape}")
        else:
            # Generate synthetic data if FRED is not available
            fred_data = data_loader._generate_synthetic_economic_data(
                len(config.data.sources.fred.series_ids)
            )
            logger.info(f"Generated synthetic economic data: {fred_data.shape}")
        
        # Load financial data
        if config.data.sources.yahoo_finance.enabled:
            yahoo_data = data_loader.load_yahoo_finance_data(
                config.data.sources.yahoo_finance.symbols,
                start_date="2000-01-01"
            )
            logger.info(f"Loaded Yahoo Finance data: {yahoo_data.shape}")
        else:
            yahoo_data = pd.DataFrame()
        
        # Combine data
        if not yahoo_data.empty:
            # Align data by date
            combined_data = pd.concat([fred_data, yahoo_data], axis=1, join="inner")
        else:
            combined_data = fred_data
        
        logger.info(f"Combined data shape: {combined_data.shape}")
        
        # Handle missing values
        preprocessor = DataPreprocessor(config.data.preprocessing.scaling)
        combined_data = preprocessor.handle_missing_values(
            combined_data,
            method=config.data.preprocessing.handle_missing
        )
        
        # Create features
        logger.info("Creating features...")
        feature_engineer = FeatureEngineer(config.features)
        features_data = feature_engineer.create_features(combined_data)
        
        logger.info(f"Features created: {features_data.shape}")
        
        # Prepare target variable (GDP growth as example)
        target_col = "GDP_Growth"
        if target_col not in features_data.columns:
            # Use first column as target if GDP_Growth not available
            target_col = features_data.columns[0]
            logger.warning(f"Target column 'GDP_Growth' not found, using '{target_col}'")
        
        # Remove rows with missing target
        features_data = features_data.dropna(subset=[target_col])
        
        # Create lagged target for prediction
        target = features_data[target_col].shift(-1)  # Predict next period
        features = features_data.drop(columns=[target_col])
        
        # Remove rows with missing target
        valid_idx = ~target.isna()
        features = features[valid_idx]
        target = target[valid_idx]
        
        logger.info(f"Final dataset shape: {features.shape}, target shape: {target.shape}")
        
        # Train-test split
        test_size = config.training.test_size
        split_idx = int(len(features) * (1 - test_size))
        
        X_train = features.iloc[:split_idx]
        y_train = target.iloc[:split_idx]
        X_test = features.iloc[split_idx:]
        y_test = target.iloc[split_idx:]
        
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Scale features
        X_train_scaled = preprocessor.fit_transform(X_train)
        X_test_scaled = preprocessor.transform(X_test)
        
        # Create and train model
        logger.info("Creating and training model...")
        model = ModelFactory.create_model(config.model)
        
        logger.info(f"Training {config.model.type} model...")
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        logger.info("Evaluating model...")
        evaluator = ModelEvaluator(config.evaluation)
        
        evaluation_results = evaluator.evaluate_model(
            model,
            X_train_scaled,
            y_train,
            X_test_scaled,
            y_test,
            run_cv=True,
            run_backtest=True
        )
        
        # Print results
        logger.info("Evaluation Results:")
        logger.info("=" * 50)
        
        # ML Metrics
        ml_metrics = evaluation_results["ml_metrics"]
        logger.info("ML Metrics:")
        for metric, value in ml_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Cross-validation results
        if "cv_scores" in evaluation_results:
            cv_scores = evaluation_results["cv_scores"]
            logger.info("\nCross-Validation Scores:")
            for metric, scores in cv_scores.items():
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                logger.info(f"  {metric}: {mean_score:.4f} (+/- {std_score:.4f})")
        
        # Backtest results
        if "backtest" in evaluation_results:
            backtest_results = evaluation_results["backtest"]
            logger.info("\nBacktest Results:")
            metrics = backtest_results["metrics"]
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
        
        # Risk assessment
        logger.info("\nRisk Assessment:")
        risk_manager = RiskManager(config.risk)
        risk_assessment = risk_manager.assess_model_risk(
            model, X_train_scaled, y_train, X_test_scaled, y_test
        )
        
        risk_report = risk_manager.generate_risk_report(risk_assessment)
        logger.info(risk_report)
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save predictions
        if config.output.save_predictions:
            predictions = model.predict(X_test_scaled)
            pred_df = pd.DataFrame({
                "actual": y_test.values,
                "predicted": predictions,
                "date": y_test.index
            })
            pred_df.to_csv(output_dir / "predictions.csv", index=False)
            logger.info(f"Predictions saved to {output_dir / 'predictions.csv'}")
        
        # Save model
        if config.output.save_model:
            import joblib
            joblib.dump(model, output_dir / "model.joblib")
            logger.info(f"Model saved to {output_dir / 'model.joblib'}")
        
        # Save evaluation results
        import json
        with open(output_dir / "evaluation_results.json", "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            results_to_save = {}
            for key, value in evaluation_results.items():
                if isinstance(value, dict):
                    results_to_save[key] = {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in value.items()
                    }
                else:
                    results_to_save[key] = value.tolist() if isinstance(value, np.ndarray) else value
            
            json.dump(results_to_save, f, indent=2)
        
        logger.info(f"Evaluation results saved to {output_dir / 'evaluation_results.json'}")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
