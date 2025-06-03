#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model training module for the Mine Safety Injury Rate Prediction model.
This script handles training and evaluation of the XGBoost model.
Integrated with MLflow for experiment tracking and model versioning.
"""

import pandas as pd
import numpy as np
import os
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from typing import Dict, Any, Tuple, List, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from xgboost import XGBRegressor

# Import MLflow utilities
from scripts.mlflow_utils import (
    log_model_training,
    register_model_version,
    load_model_from_registry,
    get_active_model_version,
    list_model_versions,
    start_mlflow_server
)

# Import feature engineering
from scripts.feature_engineering import engineer_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define constants
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_model.json")
MODEL_JOBLIB_PATH = os.path.join(MODELS_DIR, "xgboost_model.joblib")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def ll_function(targets: np.ndarray, predicted_values: np.ndarray) -> float:
    """
    Calculate log-likelihood for Poisson model.
    
    Args:
        targets (np.ndarray): Actual target values
        predicted_values (np.ndarray): Predicted values
        
    Returns:
        float: Log-likelihood value
    """
    p_v_zero = np.where(predicted_values <= 0, 0, predicted_values)
    p_v_pos = np.where(predicted_values <= 0, 0.000001, predicted_values)
    return np.sum(targets * np.log(p_v_pos)) - np.sum(p_v_zero)


def train_xgboost_model(train_data: Dict[str, Any], test_data: Dict[str, Any], 
                       params: Dict[str, Any] = None) -> Tuple[XGBRegressor, Dict[str, float]]:
    """
    Train an XGBoost model for injury rate prediction.
    
    Args:
        train_data (Dict[str, Any]): Training data dictionary
        test_data (Dict[str, Any]): Test data dictionary
        params (Dict[str, Any], optional): XGBoost parameters
        
    Returns:
        Tuple[XGBRegressor, Dict[str, float]]: Trained model and evaluation metrics
    """
    logger.info("Training XGBoost model")
    
    # Default parameters if not provided
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'objective': 'count:poisson',
            'tree_method': 'hist',
            'eval_metric': 'rmse',  # Add eval_metric to constructor in XGBoost 3.0+
            'random_state': 42,
            'n_jobs': -1
        }
    
    logger.info(f"XGBoost parameters: {params}")
    
    # Create and train XGBoost model
    model = XGBRegressor(**params)
    
    # Fit model with sample weights
    # In XGBoost 3.0+, eval_metric is set in the constructor, not in fit()
    model.fit(
        train_data['X'],
        train_data['y'],
        sample_weight=train_data['weights'],
        eval_set=[(test_data['X'], test_data['y'])],
        verbose=True
    )
    
    # Make predictions
    train_pred = model.predict(train_data['X'])
    test_pred = model.predict(test_data['X'])
    
    # Weight predictions by employee hours
    train_pred_weighted = train_pred * train_data['weights']
    test_pred_weighted = test_pred * test_data['weights']
    
    # Calculate metrics
    metrics = {
        'train_ll': ll_function(train_data['y'], train_pred_weighted),
        'test_ll': ll_function(test_data['y'], test_pred_weighted),
        'train_rmse': np.sqrt(mean_squared_error(train_data['y'], train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(test_data['y'], test_pred)),
        'train_mae': mean_absolute_error(train_data['y'], train_pred),
        'test_mae': mean_absolute_error(test_data['y'], test_pred)
    }
    
    logger.info(f"Model metrics: {metrics}")
    
    return model, metrics


def save_model(model: XGBRegressor, feature_names: List[str], run_id: Optional[str] = None) -> None:
    """
    Save the trained model to disk.
    
    Args:
        model (XGBRegressor): Trained XGBoost model
        feature_names (List[str]): List of feature names
        run_id (str, optional): MLflow run ID associated with this model
    """
    # Create directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save model in XGBoost binary format
    logger.info(f"Saving model to {MODEL_PATH}")
    model.save_model(MODEL_PATH)
    
    # Also save with joblib for easier loading
    logger.info(f"Saving model with joblib to {MODEL_JOBLIB_PATH}")
    joblib.dump(model, MODEL_JOBLIB_PATH)
    
    # Save feature names
    feature_names_path = os.path.join(MODELS_DIR, "feature_names.joblib")
    joblib.dump(feature_names, feature_names_path)
    logger.info(f"Saved feature names to {feature_names_path}")
    
    # Save MLflow run ID if provided
    if run_id:
        run_id_path = os.path.join(MODELS_DIR, "mlflow_run_id.txt")
        with open(run_id_path, 'w') as f:
            f.write(run_id)
        logger.info(f"Saved MLflow run ID to {run_id_path}")


def plot_feature_importance(model: XGBRegressor, feature_names: List[str]) -> None:
    """
    Plot feature importance from the trained model.
    
    Args:
        model (XGBRegressor): Trained XGBoost model
        feature_names (List[str]): List of feature names
    """
    # Create directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Create DataFrame for plotting
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Plot top 20 features
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feat_imp.head(20))
    plt.title('XGBoost Feature Importance (Top 20)')
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(RESULTS_DIR, "feature_importance.png")
    plt.savefig(plot_path)
    logger.info(f"Saved feature importance plot to {plot_path}")
    
    # Save feature importance data
    feat_imp_path = os.path.join(RESULTS_DIR, "feature_importance.csv")
    feat_imp.to_csv(feat_imp_path, index=False)
    logger.info(f"Saved feature importance data to {feat_imp_path}")


def load_model(version: Optional[str] = None):
    """
    Load the saved XGBoost model.
    
    Args:
        version (str, optional): Model version to load from MLflow registry
                                 If None, loads from local file or active version
    
    Returns:
        XGBRegressor: The loaded model
    """
    # Try to load from MLflow registry if version is specified or MLflow is available
    try:
        if version is not None:
            logger.info(f"Loading model version {version} from MLflow registry")
            return load_model_from_registry(version)
        
        # Try to load active version from MLflow registry
        active_version = get_active_model_version()
        if active_version != "latest":
            logger.info(f"Loading active model version {active_version} from MLflow registry")
            return load_model_from_registry(active_version)
    except Exception as e:
        logger.warning(f"Could not load model from MLflow registry: {str(e)}")
        logger.info("Falling back to local model file")
    
    # Fall back to local file
    if os.path.exists(MODEL_JOBLIB_PATH):
        logger.info(f"Loading model from {MODEL_JOBLIB_PATH}")
        model = joblib.load(MODEL_JOBLIB_PATH)
        
        # Load feature names
        feature_names_path = os.path.join(MODELS_DIR, "feature_names.joblib")
        if os.path.exists(feature_names_path):
            feature_names = joblib.load(feature_names_path)
            return model, feature_names
        else:
            logger.warning(f"Feature names not found at {feature_names_path}")
            return model, None
    else:
        logger.error(f"Model not found at {MODEL_JOBLIB_PATH}")
        raise FileNotFoundError(f"Model not found at {MODEL_JOBLIB_PATH}")


def main():
    """
    Main function to train and evaluate the XGBoost model.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and evaluate the XGBoost model with MLflow tracking')
    parser.add_argument('--register', action='store_true', help='Register the model in MLflow registry')
    parser.add_argument('--description', type=str, default='', help='Description for the model version')
    parser.add_argument('--list-versions', action='store_true', help='List all model versions')
    parser.add_argument('--set-active', type=str, help='Set the active model version')
    parser.add_argument('--start-mlflow', action='store_true', help='Start MLflow server')
    args = parser.parse_args()
    
    # Start MLflow server if requested
    if args.start_mlflow:
        start_mlflow_server()
        print("MLflow server started on http://localhost:5000")
        return
    
    # List model versions if requested
    if args.list_versions:
        versions = list_model_versions()
        print("\nModel Versions:")
        for v in versions:
            active_marker = "* (active)" if v["is_active"] else ""
            print(f"Version {v['version']} {active_marker}")
            print(f"  Created: {pd.to_datetime(v['creation_timestamp'], unit='ms')}")
            print(f"  Status: {v['status']}")
            print(f"  Test RMSE: {v.get('test_rmse', 'N/A')}")
            print(f"  Test MAE: {v.get('test_mae', 'N/A')}")
            print(f"  Description: {v.get('description', '')}")
            print()
        return
    
    # Set active model version if requested
    if args.set_active:
        from scripts.mlflow_utils import set_active_model_version
        set_active_model_version(args.set_active)
        print(f"Set active model version to: {args.set_active}")
        return
    
    # Engineer features
    train_data, test_data = engineer_features()
    
    # Define XGBoost parameters
    params = {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'objective': 'count:poisson',
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Train model
    model, metrics = train_xgboost_model(train_data, test_data, params)
    
    # Log to MLflow
    run_id = log_model_training(
        model=model,
        params=params,
        metrics=metrics,
        feature_names=train_data['feature_names'],
        train_data=train_data,
        test_data=test_data
    )
    
    # Register model if requested
    if args.register:
        model_version = register_model_version(run_id, args.description)
        print(f"Registered model version: {model_version.version}")
    
    # Save model locally
    save_model(model, train_data['feature_names'], run_id)
    
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


if __name__ == "__main__":
    main()
