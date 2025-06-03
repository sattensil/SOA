#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced main module for the Mine Safety Injury Rate Prediction model.
This script orchestrates the entire enhanced pipeline with command-line arguments.
"""

import os
import logging
import argparse
from typing import Dict, Any

# Import modules
from .data_loader import load_and_process_data
from .enhanced_feature_engineering import engineer_enhanced_features
from .enhanced_model_training import train_enhanced_xgboost_model, save_enhanced_model, plot_enhanced_feature_importance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Run the enhanced Mine Safety Injury Rate Prediction pipeline')
    
    # Data arguments
    parser.add_argument('--test-size', type=float, default=0.25,
                        help='Proportion of data to use for testing (default: 0.25)')
    parser.add_argument('--random-state', type=int, default=1234,
                        help='Random state for reproducibility (default: 1234)')
    
    # XGBoost arguments
    parser.add_argument('--n-estimators', type=int, default=100,
                        help='Number of trees in the model (default: 100)')
    parser.add_argument('--max-depth', type=int, default=5,
                        help='Maximum tree depth (default: 5)')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                        help='Learning rate (default: 0.1)')
    parser.add_argument('--skip-data', action='store_true',
                        help='Skip data loading and processing (default: False)')
    parser.add_argument('--skip-features', action='store_true',
                        help='Skip feature engineering (default: False)')
    
    return parser.parse_args()


def run_enhanced_pipeline(args) -> Dict[str, float]:
    """
    Run the enhanced pipeline with the given arguments.
    
    Args:
        args (argparse.Namespace): Command-line arguments
    """
    logger.info("Starting enhanced Mine Safety Injury Rate Prediction pipeline")
    
    # Step 1: Load and process data (if not skipped)
    if not args.skip_data:
        logger.info("Step 1: Loading and processing data")
        data = load_and_process_data(save=True)
        logger.info(f"Data loaded and processed: {data.shape}")
    else:
        logger.info("Skipping data loading and processing")
    
    # Step 2: Engineer enhanced features (if not skipped)
    if not args.skip_features:
        logger.info("Step 2: Engineering enhanced features")
        train_data, test_data = engineer_enhanced_features(
            test_size=args.test_size,
            random_state=args.random_state
        )
        logger.info(f"Enhanced features engineered: {train_data['X'].shape}, {test_data['X'].shape}")
    else:
        logger.info("Skipping feature engineering")
        # Load saved features
        import joblib
        from enhanced_feature_engineering import TRAIN_DATA_PATH, TEST_DATA_PATH
        
        train_data = joblib.load(TRAIN_DATA_PATH)
        test_data = joblib.load(TEST_DATA_PATH)
        logger.info(f"Loaded saved features: {train_data['X'].shape}, {test_data['X'].shape}")
    
    # Step 3: Train enhanced model
    logger.info("Step 3: Training enhanced XGBoost model")
    
    # Define XGBoost parameters
    params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'objective': 'count:poisson',
        'tree_method': 'hist',
        'eval_metric': 'rmse',  # Add eval_metric to constructor in XGBoost 3.0+
        'random_state': args.random_state,
        'n_jobs': -1
    }
    
    # Train model
    model, metrics = train_enhanced_xgboost_model(train_data, test_data, params)
    
    # --- MLflow logging and model registration ---
    try:
        from .mlflow_utils import log_model_training
    except ImportError:
        from scripts.mlflow_utils import log_model_training
    run_id = log_model_training(model, params, metrics, train_data['feature_names'], train_data, test_data)
    # --- End MLflow logging ---
    
    # Step 4: Save model and plot feature importance
    logger.info("Step 4: Saving model and plotting feature importance")
    save_enhanced_model(model, train_data['feature_names'])
    plot_enhanced_feature_importance(model, train_data['feature_names'])
    
    # Print metrics
    logger.info("\nEnhanced Model Evaluation Metrics:")
    logger.info(f"Train Log-Likelihood: {metrics['train_ll']:.2f}")
    logger.info(f"Test Log-Likelihood: {metrics['test_ll']:.2f}")
    logger.info(f"Train RMSE: {metrics['train_rmse']:.4f}")
    logger.info(f"Test RMSE: {metrics['test_rmse']:.4f}")
    logger.info(f"Train MAE: {metrics['train_mae']:.4f}")
    logger.info(f"Test MAE: {metrics['test_mae']:.4f}")
    
    logger.info("Enhanced pipeline completed successfully")
    
    return metrics


def main():
    """
    Main function to run the enhanced pipeline.
    """
    args = parse_args()
    run_enhanced_pipeline(args)


if __name__ == "__main__":
    main()
