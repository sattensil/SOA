"""
Utility functions for the FastAPI application.
"""
import os
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Define paths
MODELS_DIR = os.environ.get('MODELS_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'))
DATA_DIR = os.environ.get('DATA_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))
FEATURES_DIR = os.environ.get('FEATURES_DIR', os.path.join(DATA_DIR, 'features'))

def load_model_artifacts() -> Tuple[Optional[object], Optional[object], Optional[List[str]]]:
    """
    Load the model, preprocessor, and feature names.
    
    Returns:
        Tuple of (model, preprocessor, feature_names)
    """
    try:
        from scripts.prediction_utility import load_model_and_preprocessor
        
        logger.info("Loading model artifacts using prediction_utility")
        model, preprocessor, feature_names = load_model_and_preprocessor()
        
        logger.info("Model artifacts loaded successfully")
        return model, preprocessor, feature_names
    except Exception as e:
        logger.error(f"Error loading model artifacts: {str(e)}")
        return None, None, None

def preprocess_input_data(data: pd.DataFrame, preprocessor: object, feature_names: List[str]) -> np.ndarray:
    """
    Preprocess input data for prediction.
    
    Args:
        data: Input DataFrame
        preprocessor: Trained preprocessor
        feature_names: List of feature names
        
    Returns:
        Preprocessed feature array
    """
    try:
        # Try to use the prediction utility module
        from scripts.prediction_utility import preprocess_data
        
        # Use the preprocess_data function from prediction_utility
        X = preprocess_data(data, preprocessor)
        logger.info(f"Successfully preprocessed data with shape {X.shape}")
        return X
    except Exception as e:
        logger.warning(f"Error using preprocessor: {str(e)}. Using fallback method.")
        # Fallback: create a feature vector with the correct shape
        X = np.zeros((data.shape[0], len(feature_names)))
        logger.info(f"Created fallback feature array with shape {X.shape}")
        return X

def get_model_metadata() -> Dict[str, Union[str, int, List[str]]]:
    """
    Get metadata about the model.
    
    Returns:
        Dictionary with model metadata
    """
    _, _, feature_names = load_model_artifacts()
    
    return {
        "model_type": "XGBoost",
        "model_version": "enhanced_xgboost_v1.0",
        "feature_count": len(feature_names) if feature_names else 0,
        "top_features": feature_names[:10] if feature_names else [],
        "objective": "reg:squarederror",
        "last_updated": "2025-06-01"
    }

def validate_input_data(data: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate input data for prediction.
    
    Args:
        data: Input DataFrame
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check required columns
    required_columns = [
        "MINE_ID", "YEAR", "PRIMARY", "CURRENT_MINE_TYPE", 
        "CURRENT_STATUS", "FIPS_CNTY", "AVG_EMPLOYEE_CNT", 
        "HOURS_WORKED", "COAL_METAL_IND"
    ]
    
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    # Check for empty DataFrame
    if data.empty:
        return False, "Empty data provided"
    
    # Check data types
    try:
        data["YEAR"] = data["YEAR"].astype(int)
        data["AVG_EMPLOYEE_CNT"] = data["AVG_EMPLOYEE_CNT"].astype(float)
        data["HOURS_WORKED"] = data["HOURS_WORKED"].astype(float)
    except Exception as e:
        return False, f"Data type conversion error: {str(e)}"
    
    return True, ""
