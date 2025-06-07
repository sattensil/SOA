#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MLflow utilities for experiment tracking and model versioning.
"""

import os
import logging
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define constants
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'mlruns')
EXPERIMENT_NAME = "mine_safety_injury_rate_prediction"
MODEL_NAME = "mine-safety-injury-rate-model"
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_VERSION_FILE = os.path.join(MODELS_DIR, "active_model_version.txt")


def setup_mlflow():
    """
    Set up MLflow tracking.
    """
    # Set the tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Create the experiment if it doesn't exist
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(EXPERIMENT_NAME)
    
    # Set the experiment
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    logger.info(f"MLflow experiment: {EXPERIMENT_NAME}")


def log_model_training(
    model: XGBRegressor, 
    params: Dict[str, Any], 
    metrics: Dict[str, float], 
    feature_names: List[str],
    train_data: Dict[str, Any],
    test_data: Dict[str, Any]
) -> str:
    """
    Log model training to MLflow.
    
    Args:
        model: Trained XGBoost model
        params: Model parameters
        metrics: Evaluation metrics
        feature_names: Feature names
        train_data: Training data dictionary
        test_data: Test data dictionary
        
    Returns:
        run_id: MLflow run ID
    """
    setup_mlflow()
    
    # Start a new MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        
        # Log parameters
        for key, value in params.items():
            mlflow.log_param(key, value)
        
        # Log dataset info
        mlflow.log_param("train_samples", train_data['X'].shape[0])
        mlflow.log_param("test_samples", test_data['X'].shape[0])
        mlflow.log_param("features_count", len(feature_names))
        
        # Log metrics
        for key, value in metrics.items():
            # Skip lists, arrays, and other non-numeric values
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                mlflow.log_metric(key, value)
            elif isinstance(value, np.ndarray) and value.size == 1:
                # Handle numpy scalars
                mlflow.log_metric(key, float(value))
            else:
                # Log as a parameter instead for non-numeric values
                logger.debug(f"Skipping metric {key} with non-numeric value type {type(value)}")
                # Optionally log as string parameter if needed
                # mlflow.log_param(f"metric_{key}", str(value))
        
        # Log feature names
        feature_names_path = os.path.join(MODELS_DIR, "feature_names.joblib")
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(feature_names, feature_names_path)
        mlflow.log_artifact(feature_names_path, "artifacts")
        
        # Log model
        if isinstance(model, dict) and model.get('type') == 'two_stage':
            # For two-stage models, log the regression model (second stage)
            logger.info("Logging two-stage model - using regression model for MLflow")
            mlflow.xgboost.log_model(
                model['regression'], 
                "regression_model", 
                registered_model_name=MODEL_NAME
            )
            
            # Also save the classification model as an artifact
            clf_model_path = os.path.join(MODELS_DIR, "classification_model.joblib")
            joblib.dump(model['classification'], clf_model_path)
            mlflow.log_artifact(clf_model_path, "artifacts")
            
            # Log feature importance for regression model
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model['regression'].feature_importances_
            })
        else:
            # For single models
            mlflow.xgboost.log_model(
                model, 
                "model", 
                registered_model_name=MODEL_NAME
            )
            
            # Log feature importance
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save feature importance to CSV
        importance_path = os.path.join(MODELS_DIR, "feature_importance.csv")
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path, "artifacts")
        
        logger.info(f"Logged model training to MLflow run: {run_id}")
        
    return run_id


def register_model_version(run_id: str, version_description: str = "") -> ModelVersion:
    """
    Register a model version in the MLflow Model Registry.
    
    Args:
        run_id: MLflow run ID
        version_description: Description for the model version
        
    Returns:
        model_version: Registered model version
    """
    client = MlflowClient()
    
    # Get the model URI from the run
    model_uri = f"runs:/{run_id}/model"
    
    # Register the model
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
        tags={"description": version_description}
    )
    
    logger.info(f"Registered model version: {model_version.version}")
    
    # Set the active model version
    set_active_model_version(model_version.version)
    
    return model_version


def set_active_model_version(version: str) -> None:
    """
    Set the active model version.
    
    Args:
        version: Model version to set as active
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    with open(MODEL_VERSION_FILE, 'w') as f:
        f.write(str(version))
    
    logger.info(f"Set active model version to: {version}")


def get_active_model_version() -> str:
    """
    Get the active model version.
    
    Returns:
        version: Active model version
    """
    if not os.path.exists(MODEL_VERSION_FILE):
        # Default to latest version if no active version is set
        return "latest"
    
    with open(MODEL_VERSION_FILE, 'r') as f:
        version = f.read().strip()
    
    return version


def load_model_from_registry(version: Optional[str] = None) -> Tuple[XGBRegressor, List[str]]:
    """
    Load a model from the MLflow Model Registry.
    
    Args:
        version: Model version to load (None for active version)
        
    Returns:
        Tuple of (model, feature_names)
    """
    # If no version is specified, use the active version
    if version is None:
        version = get_active_model_version()
    
    # If version is "latest", get the latest version
    if version == "latest":
        client = MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME)
        if not versions:
            raise ValueError(f"No versions found for model: {MODEL_NAME}")
        version = versions[0].version
    
    # Load the model
    model_uri = f"models:/{MODEL_NAME}/{version}"
    logger.info(f"Loading model from registry: {model_uri}")
    
    try:
        model = mlflow.xgboost.load_model(model_uri)
        
        # Load feature names
        client = MlflowClient()
        run_id = client.get_model_version(MODEL_NAME, version).run_id
        
        # Get the artifact URI for the run
        artifact_uri = mlflow.get_run(run_id).info.artifact_uri
        feature_names_path = os.path.join(artifact_uri, "artifacts", "feature_names.joblib")
        
        # Strip the file:// prefix if present
        if feature_names_path.startswith("file://"):
            feature_names_path = feature_names_path[7:]
        
        feature_names = joblib.load(feature_names_path)
        
        return model, feature_names
    except Exception as e:
        logger.error(f"Error loading model from registry: {str(e)}")
        raise


def list_model_versions() -> List[Dict[str, Any]]:
    """
    List all model versions in the MLflow Model Registry.
    
    Returns:
        List of model versions with metadata
    """
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    
    active_version = get_active_model_version()
    
    result = []
    for version in versions:
        # Get run information
        run = mlflow.get_run(version.run_id)
        
        # Extract metrics
        metrics = run.data.metrics
        
        result.append({
            "version": version.version,
            "run_id": version.run_id,
            "creation_timestamp": version.creation_timestamp,
            "last_updated_timestamp": version.last_updated_timestamp,
            "description": getattr(version, 'description', ''),
            "status": getattr(version, 'current_stage', None),
            "is_active": str(version.version) == str(active_version),
            "test_rmse": metrics.get("test_rmse", None),
            "test_mae": metrics.get("test_mae", None),
            "test_ll": metrics.get("test_ll", None)
        })
    
    return result


def start_mlflow_server():
    """
    Start an MLflow tracking server.
    """
    import subprocess
    import sys
    
    # Create the mlruns directory if it doesn't exist
    os.makedirs(MLFLOW_TRACKING_URI, exist_ok=True)
    
    # Get the MLflow port from environment variable or use default
    mlflow_port = os.environ.get('MLFLOW_PORT', '5000')
    
    # Start the MLflow server
    cmd = [sys.executable, "-m", "mlflow", "ui", "--port", mlflow_port]
    subprocess.Popen(cmd)
    
    logger.info(f"Started MLflow server on port {mlflow_port}")
