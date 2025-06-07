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
import pandas as pd # For MLflow list_versions output

# Import MLflow utilities
try:
    from .mlflow_utils import (
        log_model_training, # Already used, but good to list all
        register_model_version,
        list_model_versions,
        set_active_model_version,
        start_mlflow_server
    )
except ImportError:
    from scripts.mlflow_utils import ( # Fallback for different execution contexts
        log_model_training,
        register_model_version,
        list_model_versions,
        set_active_model_version,
        start_mlflow_server
    )

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
    parser = argparse.ArgumentParser(description='Run the enhanced Mine Safety Injury Rate Prediction pipeline with MLflow integration')
    
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
    parser.add_argument('--optimize_threshold', action='store_true',
                        help='Optimize classification threshold for binary model')
    parser.add_argument('--two_stage', action='store_true',
                        help='Use two-stage modeling approach')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output (default: False)')

    # MLflow arguments
    parser.add_argument('--register', action='store_true',
                        help='Register the model in MLflow registry')
    parser.add_argument('--description', type=str, default='Enhanced XGBoost Model',
                        help='Description for the model version (default: Enhanced XGBoost Model)')
    parser.add_argument('--list-versions', action='store_true',
                        help='List all model versions from MLflow')
    parser.add_argument('--set-active', type=str,
                        help='Set the active model version in MLflow (provide version string)')
    parser.add_argument('--start-mlflow', action='store_true',
                        help='Start MLflow server and exit')
    
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
    model, metrics = train_enhanced_xgboost_model(train_data, test_data, params, optimize_threshold=args.optimize_threshold, two_stage=args.two_stage)
    
    # --- MLflow logging and model registration ---
    try:
        from .mlflow_utils import log_model_training
    except ImportError:
        from scripts.mlflow_utils import log_model_training
    run_id = log_model_training(model, params, metrics, train_data['feature_names'], train_data, test_data)
    # --- End MLflow logging ---

    # Register model if requested
    if args.register:
        try:
            model_version_details = register_model_version(run_id, args.description)
            logger.info(f"Registered model version: {model_version_details.version} (Run ID: {run_id}) with description: '{args.description}'")
        except Exception as e:
            logger.error(f"Failed to register model version for run_id {run_id}: {e}")
    
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
    Handles MLflow utility commands before pipeline execution.
    """
    args = parse_args()

    # Handle MLflow utility commands that exit early
    if args.start_mlflow:
        start_mlflow_server()
        print("MLflow server started on http://localhost:5000. Access it in your browser.")
        return

    if args.list_versions:
        try:
            versions = list_model_versions()
            if not versions:
                print("No model versions found in MLflow registry.")
                return
            print("\nMLflow Model Versions:")
            for v_info in versions:
                active_marker = "* (active)" if v_info.get("is_active") else ""
                print(f"  Version: {v_info.get('version')} {active_marker}")
                created_ts = v_info.get('creation_timestamp')
                print(f"    Created: {pd.to_datetime(created_ts, unit='ms') if created_ts else 'N/A'}")
                print(f"    Status: {v_info.get('status', 'N/A')}")
                # Assuming metrics might be nested or directly available
                metrics_dict = v_info.get('metrics', {}) if isinstance(v_info.get('metrics'), dict) else {}
                run_data_metrics = v_info.get('run_data', {}).get('metrics', {}) if isinstance(v_info.get('run_data'), dict) else {}
                final_metrics = {**metrics_dict, **run_data_metrics} # Merge them, run_data_metrics might be more up-to-date
                print(f"    Test RMSE: {final_metrics.get('test_rmse', 'N/A')}")
                print(f"    Test MAE: {final_metrics.get('test_mae', 'N/A')}")
                print(f"    Description: {v_info.get('description', '')}")
                print()
        except Exception as e:
            print(f"Error listing model versions: {e}")
        return

    if args.set_active:
        try:
            set_active_model_version(args.set_active)
            print(f"Successfully set active model version to: {args.set_active}")
        except Exception as e:
            print(f"Failed to set active model version {args.set_active}: {e}")
        return

    # If no early-exit MLflow commands, run the main pipeline
    run_enhanced_pipeline(args)
    
    # Update metrics database if a model was registered
    if args.register:
        try:
            print("Updating metrics database with latest model information...")
            # Import here to avoid circular imports
            from scripts.export_metrics_to_db import export_metrics_to_db
            export_metrics_to_db()
            print("Metrics database updated successfully")
        except Exception as e:
            print(f"Warning: Failed to update metrics database: {e}")


if __name__ == "__main__":
    main()
