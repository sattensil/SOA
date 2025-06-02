"""
API routes for the mine safety injury rate prediction service.
"""
import logging
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Optional

from .models import MineData, BatchMineData, PredictionResponse, BatchPredictionResponse
from .utils import load_model_artifacts, preprocess_input_data, get_model_metadata, validate_input_data
from scripts.data_loader import perform_basic_cleaning as clean_data

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Global variables to store model artifacts
model = None
preprocessor = None
feature_names = None

def get_model():
    """
    Dependency to get the model, preprocessor, and feature names.
    """
    global model, preprocessor, feature_names
    
    if model is None or preprocessor is None or feature_names is None:
        model, preprocessor, feature_names = load_model_artifacts()
        
    if model is None or preprocessor is None or feature_names is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    return model, preprocessor, feature_names

@router.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Mine Safety Injury Rate Prediction API"}

@router.get("/health")
async def health_check(artifacts=Depends(get_model)):
    """Health check endpoint."""
    model, preprocessor, feature_names = artifacts
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "feature_count": len(feature_names) if feature_names else 0
    }

@router.post("/predict", response_model=PredictionResponse)
async def predict(mine_data: MineData, artifacts=Depends(get_model)):
    """Predict injury rate for a single mine."""
    model, preprocessor, feature_names = artifacts
    
    try:
        # Convert to DataFrame for preprocessing
        df = pd.DataFrame([mine_data.dict()])
        
        # Validate input data
        is_valid, error_message = validate_input_data(df)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_message)
        
        # Clean data - using perform_basic_cleaning imported as clean_data
        cleaned_data = clean_data(df.copy())
        
        if cleaned_data.empty:
            raise HTTPException(status_code=400, detail="Data cleaning removed all records")
        
        # Preprocess data
        X = preprocess_input_data(cleaned_data, preprocessor, feature_names)
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        return {
            "mine_id": mine_data.MINE_ID,
            "predicted_injury_rate": float(prediction)
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch_data: BatchMineData, artifacts=Depends(get_model)):
    """Predict injury rates for multiple mines."""
    model, preprocessor, feature_names = artifacts
    
    if len(batch_data.mines) == 0:
        raise HTTPException(status_code=400, detail="Empty batch")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([mine.dict() for mine in batch_data.mines])
        
        # Validate input data
        is_valid, error_message = validate_input_data(df)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_message)
        
        # Clean data - using perform_basic_cleaning imported as clean_data
        cleaned_data = clean_data(df.copy())
        
        if cleaned_data.empty:
            raise HTTPException(status_code=400, detail="Data cleaning removed all records")
        
        # Preprocess data
        X = preprocess_input_data(cleaned_data, preprocessor, feature_names)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Create response
        results = []
        for i, mine in enumerate(batch_data.mines):
            if i < len(predictions):
                results.append({
                    "mine_id": mine.MINE_ID,
                    "predicted_injury_rate": float(predictions[i])
                })
        
        return {
            "predictions": results,
            "model_version": "enhanced_xgboost_v1.0",
            "average_predicted_rate": float(np.mean(predictions))
        }
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@router.get("/model/info")
async def model_info():
    """Get information about the model."""
    try:
        return get_model_metadata()
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@router.get("/model/features")
async def model_features(artifacts=Depends(get_model)):
    """Get the features used by the model."""
    _, _, feature_names = artifacts
    
    return {
        "feature_count": len(feature_names),
        "features": feature_names
    }
