#!/usr/bin/env python3
"""
Script to create multiple model versions with different hyperparameters
and register them in MLflow.
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from typing import Dict, Any, List

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.data_loader import load_and_process_data
from scripts.enhanced_feature_engineering import engineer_enhanced_features
from scripts.mlflow_utils import setup_mlflow, register_model_version, set_active_model_version

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define model configurations to try
MODEL_CONFIGS = [
    {
        "name": "XGBoost - Default",
        "model_class": XGBRegressor,
        "params": {
            "objective": "reg:squarederror",
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": 42
        }
    },
    {
        "name": "XGBoost - More Trees",
        "model_class": XGBRegressor,
        "params": {
            "objective": "reg:squarederror",
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": 42
        }
    },
    {
        "name": "XGBoost - Deeper Trees",
        "model_class": XGBRegressor,
        "params": {
            "objective": "reg:squarederror",
            "n_estimators": 100,
            "max_depth": 7,
            "learning_rate": 0.1,
            "random_state": 42
        }
    },
    {
        "name": "XGBoost - Lower Learning Rate",
        "model_class": XGBRegressor,
        "params": {
            "objective": "reg:squarederror",
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.05,
            "random_state": 42
        }
    },
    {
        "name": "RandomForest",
        "model_class": RandomForestRegressor,
        "params": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        }
    },
    {
        "name": "GradientBoosting",
        "model_class": GradientBoostingRegressor,
        "params": {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": 42
        }
    }
]

def evaluate_model(model, X_train, y_train, X_test, y_test) -> Dict[str, float]:
    """
    Evaluate a model and return performance metrics.
    """
    model.fit(X_train, y_train)
    
    # Make predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(np.mean((train_preds - y_train) ** 2))
    test_rmse = np.sqrt(np.mean((test_preds - y_test) ** 2))
    train_mae = np.mean(np.abs(train_preds - y_train))
    test_mae = np.mean(np.abs(test_preds - y_test))
    
    # Calculate log-likelihood (simplified)
    test_ll = -np.sum((test_preds - y_test) ** 2) / (2 * len(y_test))
    
    return {
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "test_ll": test_ll
    }

def train_and_register_models():
    """
    Train and register multiple model versions with different configurations.
    """
    logger.info("Loading and processing data...")
    data = load_and_process_data()
    
    logger.info("Engineering enhanced features...")
    train_data, test_data = engineer_enhanced_features()
    
    # Extract features and target
    X_train = train_data['X']
    y_train = train_data['y']
    X_test = test_data['X']
    y_test = test_data['y']
    feature_names = train_data['feature_names']
    
    # Set up MLflow
    setup_mlflow()
    
    best_model = None
    best_metrics = None
    best_run_id = None
    best_version = None
    
    # Train and evaluate each model configuration
    for i, config in enumerate(MODEL_CONFIGS):
        logger.info(f"Training model {i+1}/{len(MODEL_CONFIGS)}: {config['name']}")
        
        # Create and train the model
        model = config["model_class"](**config["params"])
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
        
        logger.info(f"Model {config['name']} metrics:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
        
        # Start MLflow run
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            
            # Log parameters
            mlflow.log_params(config["params"])
            mlflow.log_param("model_type", config["name"])
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            if isinstance(model, XGBRegressor):
                mlflow.xgboost.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")
            
            # Register model version
            version = register_model_version(run_id, f"Model version for {config['name']}")
            
            logger.info(f"Registered model version {version.version} for {config['name']}")
            
            # Track best model
            if best_metrics is None or metrics["test_rmse"] < best_metrics["test_rmse"]:
                best_model = model
                best_metrics = metrics
                best_run_id = run_id
                best_version = version.version
    
    # Set the best model as active
    if best_version:
        logger.info(f"Setting model version {best_version} as active (best test RMSE: {best_metrics['test_rmse']:.4f})")
        set_active_model_version(best_version)
    
    return best_version, best_metrics

if __name__ == "__main__":
    best_version, best_metrics = train_and_register_models()
    
    logger.info("\n=== Best Model ===")
    logger.info(f"Version: {best_version}")
    logger.info(f"Test RMSE: {best_metrics['test_rmse']:.4f}")
    logger.info(f"Test MAE: {best_metrics['test_mae']:.4f}")
    logger.info(f"Test Log-Likelihood: {best_metrics['test_ll']:.4f}")
