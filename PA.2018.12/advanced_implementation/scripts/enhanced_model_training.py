#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced model training module for the Mine Safety Injury Rate Prediction model.
This script handles training and evaluation of the XGBoost model with enhanced features.
"""

import pandas as pd
import numpy as np
import os
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, List
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from xgboost import XGBRegressor

# Import enhanced feature engineering
from enhanced_feature_engineering import engineer_enhanced_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define constants
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
ENHANCED_MODEL_PATH = os.path.join(MODELS_DIR, "enhanced_xgboost_model.json")
ENHANCED_MODEL_JOBLIB_PATH = os.path.join(MODELS_DIR, "enhanced_xgboost_model.joblib")
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


def train_enhanced_xgboost_model(train_data: Dict[str, Any], test_data: Dict[str, Any], 
                       params: Dict[str, Any] = None) -> Tuple[XGBRegressor, Dict[str, float]]:
    """
    Train an XGBoost model with enhanced features for injury rate prediction.
    
    Args:
        train_data (Dict[str, Any]): Training data dictionary
        test_data (Dict[str, Any]): Test data dictionary
        params (Dict[str, Any], optional): XGBoost parameters
        
    Returns:
        Tuple[XGBRegressor, Dict[str, float]]: Trained model and evaluation metrics
    """
    logger.info("Training XGBoost model with enhanced features")
    
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


def save_enhanced_model(model: XGBRegressor, feature_names: List[str]) -> None:
    """
    Save the trained enhanced model to disk.
    
    Args:
        model (XGBRegressor): Trained XGBoost model
        feature_names (List[str]): List of feature names
    """
    # Create directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save model in XGBoost binary format
    logger.info(f"Saving enhanced model to {ENHANCED_MODEL_PATH}")
    model.save_model(ENHANCED_MODEL_PATH)
    
    # Also save with joblib for easier loading
    logger.info(f"Saving enhanced model with joblib to {ENHANCED_MODEL_JOBLIB_PATH}")
    joblib.dump(model, ENHANCED_MODEL_JOBLIB_PATH)
    
    # Save feature names
    feature_names_path = os.path.join(MODELS_DIR, "enhanced_feature_names.joblib")
    joblib.dump(feature_names, feature_names_path)
    logger.info(f"Saved enhanced feature names to {feature_names_path}")


def plot_enhanced_feature_importance(model: XGBRegressor, feature_names: List[str]) -> None:
    """
    Plot feature importance from the trained enhanced model.
    
    Args:
        model (XGBRegressor): Trained XGBoost model
        feature_names (List[str]): List of feature names
    """
    # Create directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Debug: Check lengths of feature_names and importance
    logger.info(f"Number of feature names: {len(feature_names)}")
    logger.info(f"Number of feature importances: {len(importance)}")
    
    # Handle mismatch in lengths
    if len(feature_names) != len(importance):
        logger.warning(f"Mismatch between feature names ({len(feature_names)}) and importances ({len(importance)})")
        # Use generic feature names if mismatch
        if len(importance) > len(feature_names):
            logger.warning("Using generic feature names for plotting")
            feature_names = [f"Feature_{i}" for i in range(len(importance))]
        else:
            # Truncate feature names if there are more names than importances
            logger.warning("Truncating feature names to match importances")
            feature_names = feature_names[:len(importance)]
    
    # Create DataFrame for plotting
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Plot top 20 features (or fewer if less than 20 features)
    plt.figure(figsize=(12, 8))
    top_n = min(20, len(feat_imp))
    sns.barplot(x='importance', y='feature', data=feat_imp.head(top_n))
    plt.title('XGBoost Feature Importance with Enhanced Features (Top Features)')
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(RESULTS_DIR, "enhanced_feature_importance.png")
    plt.savefig(plot_path)
    logger.info(f"Saved enhanced feature importance plot to {plot_path}")
    
    # Save feature importance data
    feat_imp_path = os.path.join(RESULTS_DIR, "enhanced_feature_importance.csv")
    feat_imp.to_csv(feat_imp_path, index=False)
    logger.info(f"Saved enhanced feature importance data to {feat_imp_path}")


def load_enhanced_model():
    """
    Load the saved enhanced XGBoost model.
    
    Returns:
        XGBRegressor: The loaded model
    """
    if os.path.exists(ENHANCED_MODEL_JOBLIB_PATH):
        logger.info(f"Loading enhanced model from {ENHANCED_MODEL_JOBLIB_PATH}")
        return joblib.load(ENHANCED_MODEL_JOBLIB_PATH)
    else:
        logger.error(f"Enhanced model not found at {ENHANCED_MODEL_JOBLIB_PATH}")
        raise FileNotFoundError(f"Enhanced model not found at {ENHANCED_MODEL_JOBLIB_PATH}")


def main():
    """
    Main function to train and evaluate the enhanced XGBoost model.
    """
    # Engineer enhanced features
    train_data, test_data = engineer_enhanced_features()
    
    # Define XGBoost parameters
    params = {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'objective': 'count:poisson',
        'tree_method': 'hist',
        'eval_metric': 'rmse',
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Train model
    model, metrics = train_enhanced_xgboost_model(train_data, test_data, params)
    
    # Save model
    save_enhanced_model(model, train_data['feature_names'])
    
    # Plot feature importance
    plot_enhanced_feature_importance(model, train_data['feature_names'])
    
    # Print metrics
    print("\nEnhanced Model Evaluation Metrics:")
    print(f"Train Log-Likelihood: {metrics['train_ll']:.2f}")
    print(f"Test Log-Likelihood: {metrics['test_ll']:.2f}")
    print(f"Train RMSE: {metrics['train_rmse']:.4f}")
    print(f"Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"Train MAE: {metrics['train_mae']:.4f}")
    print(f"Test MAE: {metrics['test_mae']:.4f}")


if __name__ == "__main__":
    main()
