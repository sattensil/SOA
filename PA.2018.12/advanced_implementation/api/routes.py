"""
API routes for the mine safety injury rate prediction service.
"""
import os
import sys
import logging
import traceback
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Path
from typing import Dict, List, Optional

# Import models
from .models import MineData, BatchMineData, PredictionResponse, BatchPredictionResponse, ModelInfo, HealthResponse

# Import MLflow utilities if available
try:
    from scripts.mlflow_utils import (
        list_model_versions, 
        set_active_model_version, 
        get_active_model_version,
        start_mlflow_server
    )
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Import utilities
from .utils import load_model_artifacts, preprocess_input_data, get_model_metadata, validate_input_data

# Try both import paths to support both local development and containerized environment
try:
    from scripts.data_loader import perform_basic_cleaning as clean_data
except ImportError:
    from data_loader import perform_basic_cleaning as clean_data

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Model dependency
def get_model(version: Optional[str] = None):
    """Load the model as a dependency."""
    model, preprocessor, feature_names = load_model_artifacts(version)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")
    return model, preprocessor, feature_names

@router.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Mine Safety Injury Rate Prediction API"}

# Model Version Management Endpoints
@router.get("/model-versions", tags=["model-management"])
async def list_versions():
    """List all available model versions."""
    if not MLFLOW_AVAILABLE:
        raise HTTPException(status_code=501, detail="MLflow integration not available")
    
    try:
        versions = list_model_versions()
        active_version = get_active_model_version()
        
        # Format the response
        result = {
            "versions": [
                {
                    "version": v["version"],
                    "created_at": pd.to_datetime(v["creation_timestamp"], unit="ms").isoformat(),
                    "status": v["status"],
                    "is_active": v["is_active"],
                    "metrics": {
                        "test_rmse": v.get("test_rmse"),
                        "test_mae": v.get("test_mae"),
                        "test_ll": v.get("test_ll")
                    },
                    "description": v.get("description", "")
                } for v in versions
            ],
            "active_version": active_version
        }
        
        return result
    except Exception as e:
        logging.error(f"Error listing model versions: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error listing model versions: {str(e)}")


@router.post("/model-versions/{version}/activate", tags=["model-management"])
async def activate_version(version: str = Path(..., description="Model version to activate")):
    """Set a model version as the active version."""
    if not MLFLOW_AVAILABLE:
        raise HTTPException(status_code=501, detail="MLflow integration not available")
    
    try:
        # Check if version exists
        versions = list_model_versions()
        version_exists = any(v["version"] == version for v in versions)
        
        if not version_exists:
            raise HTTPException(status_code=404, detail=f"Model version {version} not found")
        
        # Set active version
        set_active_model_version(version)
        
        return {"message": f"Model version {version} activated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error activating model version: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error activating model version: {str(e)}")


@router.get("/model-versions/active", tags=["model-management"])
async def get_active_version():
    """Get the currently active model version."""
    if not MLFLOW_AVAILABLE:
        raise HTTPException(status_code=501, detail="MLflow integration not available")
    
    try:
        active_version = get_active_model_version()
        return {"active_version": active_version}
    except Exception as e:
        logging.error(f"Error getting active model version: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error getting active model version: {str(e)}")

@router.get("/health", response_model=HealthResponse)
async def health_check(version: Optional[str] = Query(None, description="Model version to check health for")):
    """Health check endpoint."""
    model, preprocessor, feature_names = load_model_artifacts(version)
    
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "feature_count": len(feature_names) if feature_names else 0
    }

@router.post("/predict", response_model=PredictionResponse)
async def predict_single(
    mine_data: MineData, 
    version: Optional[str] = Query(None, description="Model version to use for prediction"),
    model_data=Depends(get_model)
):
    """Predict injury rate for a single mine."""
    # If version is specified, load that version specifically
    if version is not None:
        model, preprocessor, feature_names = await get_model(version)
    else:
        model, preprocessor, feature_names = model_data
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([mine_data.dict()])
        
        # Validate input data
        is_valid, error_message = validate_input_data(df)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_message)
        
        # Preprocess data
        X = preprocess_input_data(df, preprocessor, feature_names)
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Return response
        return {
            "mine_id": mine_data.MINE_ID,
            "predicted_injury_rate": float(prediction)
        }
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(
    batch_data: BatchMineData, 
    version: Optional[str] = Query(None, description="Model version to use for prediction"),
    model_data=Depends(get_model)
):
    """Predict injury rates for multiple mines."""
    # If version is specified, load that version specifically
    if version is not None:
        model, preprocessor, feature_names = await get_model(version)
    else:
        model, preprocessor, feature_names = model_data
    
    # Get model version for response
    model_version = version or (get_active_model_version() if MLFLOW_AVAILABLE else "latest")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([mine.dict() for mine in batch_data.mines])
        
        # Validate input data
        is_valid, error_message = validate_input_data(df)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_message)
        
        # Preprocess data
        X = preprocess_input_data(df, preprocessor, feature_names)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Create response
        prediction_list = [
            {"mine_id": mine_id, "predicted_injury_rate": float(pred)}
            for mine_id, pred in zip(df["MINE_ID"], predictions)
        ]
        
        # Calculate average predicted rate
        avg_rate = float(np.mean(predictions))
        
        # Return response
        return {
            "predictions": prediction_list,
            "model_version": f"enhanced_xgboost_v{model_version}",
            "average_predicted_rate": avg_rate
        }
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logging.error(f"Error in batch prediction: {str(e)}")
        logging.error(f"Data shape: {df.shape}")
        logging.error(f"Columns: {df.columns.tolist()}")
        logging.error(f"First few rows: {df.head().to_dict('records')}")
        logging.error(tb)
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}\nTraceback:\n{tb}")

@router.get("/model-info", response_model=ModelInfo)
async def get_model_info(version: Optional[str] = Query(None, description="Model version to get info for")):
    """Get information about the model."""
    return get_model_metadata(version)

@router.get("/model/features")
async def model_features(version: Optional[str] = Query(None, description="Model version to get features for")):
    """Get the features used by the model."""
    _, _, feature_names = load_model_artifacts(version)
    
    return {"features": feature_names}
