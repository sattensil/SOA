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

# Try both import paths to support both local development and containerized environment
try:
    from scripts.data_loader import perform_basic_cleaning as clean_data
except ImportError:
    from data_loader import perform_basic_cleaning as clean_data

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
    In the simplified approach, preprocessor may be None.
    """
    global model, preprocessor, feature_names
    
    if model is None or feature_names is None:
        model, preprocessor, feature_names = load_model_artifacts()
        
    if model is None or feature_names is None:
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
        logger.info(f"Input data: {df.columns.tolist()}")
        logger.info(f"Input data values: {df.iloc[0].to_dict()}")
        
        # Ensure CURRENT_STATUS is mapped to MINE_STATUS directly in the dataframe
        if 'CURRENT_STATUS' in df.columns:
            df['MINE_STATUS'] = df['CURRENT_STATUS']
            logger.info(f"Explicitly mapped CURRENT_STATUS to MINE_STATUS: {df['MINE_STATUS'].iloc[0]}")
        
        # Validate input data
        is_valid, error_message = validate_input_data(df)
        if not is_valid:
            logger.error(f"Validation error: {error_message}")
            raise HTTPException(status_code=400, detail=error_message)
        
        logger.info("Input data validation successful")
        logger.info(f"Data after validation: {df.columns.tolist()}")
        
        # Log all data values for debugging
        for col in df.columns:
            logger.info(f"Column {col} value: {df[col].iloc[0]}")
        
        # Clean data - using perform_basic_cleaning imported as clean_data
        try:
            # Ensure MINE_STATUS is present and has the correct value
            if 'MINE_STATUS' not in df.columns and 'CURRENT_STATUS' in df.columns:
                df['MINE_STATUS'] = df['CURRENT_STATUS']
                logger.info(f"Added MINE_STATUS from CURRENT_STATUS before cleaning: {df['MINE_STATUS'].iloc[0]}")
            
            # Ensure MINE_STATUS is uppercase ACTIVE for filtering
            if 'MINE_STATUS' in df.columns:
                df['MINE_STATUS'] = df['MINE_STATUS'].str.upper()
                logger.info(f"Converted MINE_STATUS to uppercase: {df['MINE_STATUS'].iloc[0]}")
            
            cleaned_data = clean_data(df.copy())
            logger.info(f"Cleaned data shape: {cleaned_data.shape}")
            if not cleaned_data.empty:
                logger.info(f"Cleaned data columns: {cleaned_data.columns.tolist()}")
                for col in cleaned_data.columns:
                    logger.info(f"Cleaned column {col} value: {cleaned_data[col].iloc[0]}")
            else:
                logger.error("Cleaned data is empty")
        except Exception as clean_error:
            logger.error(f"Data cleaning error: {str(clean_error)}")
            import traceback
            logger.error(f"Cleaning traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Data cleaning error: {str(clean_error)}")
        
        if cleaned_data.empty:
            logger.error("Data cleaning removed all records")
            raise HTTPException(status_code=400, detail="Data cleaning removed all records - make sure MINE_STATUS is 'ACTIVE' and HOURS_WORKED >= 5000")
        
        # Preprocess data
        try:
            X = preprocess_input_data(cleaned_data, preprocessor, feature_names)
            logger.info(f"Preprocessed data shape: {X.shape}")
        except Exception as preprocess_error:
            logger.error(f"Preprocessing error: {str(preprocess_error)}")
            import traceback
            logger.error(f"Preprocessing traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(preprocess_error)}")
        
        # Make prediction
        try:
            prediction = model.predict(X)[0]
            logger.info(f"Prediction successful: {float(prediction)}")
        except Exception as predict_error:
            logger.error(f"Prediction model error: {str(predict_error)}")
            import traceback
            logger.error(f"Prediction traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Prediction model error: {str(predict_error)}")
        
        return {
            "mine_id": mine_data.MINE_ID,
            "predicted_injury_rate": float(prediction)
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        import traceback
        logger.error(f"General error traceback: {traceback.format_exc()}")
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
        logger.info(f"Batch input data columns: {df.columns.tolist()}")
        
        # Log the first record for debugging
        if not df.empty:
            logger.info(f"First batch record: {df.iloc[0].to_dict()}")
        
        # Apply field mappings before validation
        field_mappings = {
            'CURRENT_STATUS': 'MINE_STATUS',
            'INJURIES_COUNT': 'NUM_INJURIES',
            'HOURS_WORKED': 'EMP_HRS_TOTAL'
        }
        
        # Apply field mappings
        for source, target in field_mappings.items():
            if source in df.columns and target not in df.columns:
                df[target] = df[source]
                logger.info(f"Explicitly mapped {source} to {target} for batch prediction")
        
        # Ensure MINE_STATUS is uppercase for filtering
        if 'MINE_STATUS' in df.columns:
            df['MINE_STATUS'] = df['MINE_STATUS'].astype(str).str.upper()
            logger.info(f"Converted MINE_STATUS to uppercase: {df['MINE_STATUS'].unique().tolist()}")
        elif 'CURRENT_STATUS' in df.columns:
            # Double-check mapping
            df['MINE_STATUS'] = df['CURRENT_STATUS'].astype(str).str.upper()
            logger.info(f"Created MINE_STATUS from CURRENT_STATUS: {df['MINE_STATUS'].unique().tolist()}")
        
        # Validate input data
        is_valid, error_message = validate_input_data(df)
        if not is_valid:
            logger.error(f"Batch validation error: {error_message}")
            raise HTTPException(status_code=400, detail=error_message)
        
        # Clean data - using perform_basic_cleaning imported as clean_data
        try:
            cleaned_data = clean_data(df.copy())
            logger.info(f"Cleaned batch data shape: {cleaned_data.shape}")
        except Exception as clean_error:
            logger.error(f"Batch data cleaning error: {str(clean_error)}")
            import traceback
            logger.error(f"Cleaning traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Batch data cleaning error: {str(clean_error)}")
        
        if cleaned_data.empty:
            logger.error("Batch data cleaning removed all records")
            raise HTTPException(status_code=400, detail="Data cleaning removed all records - make sure MINE_STATUS is 'ACTIVE' and HOURS_WORKED >= 5000")
        
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
        import traceback
        logger.error(f"Batch prediction traceback: {traceback.format_exc()}")
        
        # Log the DataFrame columns and data for debugging
        if 'df' in locals():
            logger.error(f"Batch input DataFrame columns: {df.columns.tolist()}")
            logger.error(f"Batch input DataFrame head: {df.head(1).to_dict('records')}")
            # Print all columns to help debug the MINE_STATUS issue
            for col in df.columns:
                logger.error(f"Column {col} values: {df[col].tolist()}")
        
        # Log the cleaned data if available
        if 'cleaned_data' in locals():
            logger.error(f"Cleaned DataFrame columns: {cleaned_data.columns.tolist()}")
            logger.error(f"Cleaned DataFrame head: {cleaned_data.head(1).to_dict('records')}")
        
        # Try to directly debug the MINE_STATUS issue
        if 'df' in locals() and 'MINE_STATUS' not in df.columns:
            logger.error("MINE_STATUS column is missing from the DataFrame!")
            if 'CURRENT_STATUS' in df.columns:
                logger.error(f"CURRENT_STATUS values: {df['CURRENT_STATUS'].tolist()}")
                
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")
        
        # Return a more detailed error response for debugging
        # raise HTTPException(
        #    status_code=500, 
        #    detail={
        #        "error": str(e),
        #        "columns": df.columns.tolist() if 'df' in locals() else [],
        #        "traceback": traceback.format_exc()
        #    }
        #)

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
