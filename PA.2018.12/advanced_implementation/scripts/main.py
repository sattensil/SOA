#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for the Mine Safety Injury Rate Prediction model.
This script orchestrates the entire process from data loading to model training.
"""

import os
import logging
import argparse
import time
from typing import Dict, Any

# Import modules
from scripts.data_loader import load_and_process_data
from scripts.feature_engineering import engineer_features
from scripts.model_training import train_xgboost_model, save_model, plot_feature_importance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Mine Safety Injury Rate Prediction')
    
    parser.add_argument('--skip-data-loading', action='store_true',
                       help='Skip data loading and use existing processed data')
    
    parser.add_argument('--skip-feature-engineering', action='store_true',
                       help='Skip feature engineering and use existing features')
    
    parser.add_argument('--n-estimators', type=int, default=100,
                       help='Number of estimators for XGBoost')
    
    parser.add_argument('--max-depth', type=int, default=5,
                       help='Maximum depth of trees for XGBoost')
    
    parser.add_argument('--learning-rate', type=float, default=0.1,
                       help='Learning rate for XGBoost')
    
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility')
    
    return parser.parse_args()


def main():
    """
    Main function to run the entire pipeline.
    """
    # Parse arguments
    args = parse_arguments()
    
    start_time = time.time()
    logger.info("Starting Mine Safety Injury Rate Prediction pipeline")
    
    # Step 1: Data Loading
    if not args.skip_data_loading:
        logger.info("Step 1: Data Loading")
        processed_data = load_and_process_data(save=True)
        logger.info(f"Data loaded and processed: {processed_data.shape}")
    else:
        logger.info("Skipping data loading")
    
    # Step 2: Feature Engineering
    if not args.skip_feature_engineering:
        logger.info("Step 2: Feature Engineering")
        train_data, test_data = engineer_features()
        logger.info(f"Features engineered: {train_data['X'].shape}, {test_data['X'].shape}")
    else:
        logger.info("Skipping feature engineering")
        # Load engineered features if skipping
        # This would need to be implemented
    
    # Step 3: Model Training
    logger.info("Step 3: Model Training")
    
    # Define XGBoost parameters from arguments
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
    model, metrics = train_xgboost_model(train_data, test_data, params)
    
    # Save model
    save_model(model, train_data['feature_names'])
    
    # Plot feature importance
    plot_feature_importance(model, train_data['feature_names'])
    
    # Print metrics
    print("\nModel Evaluation Metrics:")
    print(f"Train Log-Likelihood: {metrics['train_ll']:.2f}")
    print(f"Test Log-Likelihood: {metrics['test_ll']:.2f}")
    print(f"Train RMSE: {metrics['train_rmse']:.4f}")
    print(f"Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"Train MAE: {metrics['train_mae']:.4f}")
    print(f"Test MAE: {metrics['test_mae']:.4f}")
    
    # Calculate and print elapsed time
    elapsed_time = time.time() - start_time
    logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
