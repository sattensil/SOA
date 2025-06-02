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
        # Use the simplified prediction utility
        try:
            from simple_prediction import load_model
        except ImportError:
            # Try with absolute import if relative import fails
            from api.simple_prediction import load_model
        
        model, feature_names = load_model()
        # Return the model and feature names, with None for preprocessor
        # since we're not using a serialized preprocessor anymore
        return model, None, feature_names
    except Exception as e:
        logger.error(f"Error loading model artifacts: {str(e)}")
        return None, None, None

def preprocess_input_data(data: pd.DataFrame, preprocessor: object, feature_names: List[str]) -> np.ndarray:
    """
    Preprocess input data using the simplified preprocessing function.
    
    Args:
        data (pd.DataFrame): Input data to preprocess
        preprocessor (object): Not used in the simplified version
        feature_names (List[str]): List of feature names
        
    Returns:
        np.ndarray: Preprocessed features
    """
    try:
        # Use the simplified prediction utility
        try:
            from simple_prediction import preprocess_data
        except ImportError:
            # Try with absolute import if relative import fails
            from api.simple_prediction import preprocess_data
            
        X = preprocess_data(data, feature_names)
        return X
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        # Fallback to returning empty array with correct shape
        return np.zeros((data.shape[0], len(feature_names)))

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
    # Make a copy of the data to avoid modifying the original
    df = data.copy()
    
    # Handle field mappings for validation
    field_mappings = {
        'CURRENT_STATUS': 'MINE_STATUS',
        'INJURIES_COUNT': 'NUM_INJURIES',
        'HOURS_WORKED': 'EMP_HRS_TOTAL'
    }
    
    # Apply field mappings to the data
    for source, target in field_mappings.items():
        if source in df.columns and target not in df.columns:
            df[target] = df[source]
            logger.info(f"Mapped {source} to {target} during validation")
    
    # Check required columns - note we check after mapping to handle aliases
    required_columns = [
        "MINE_ID", "YEAR", "PRIMARY", "CURRENT_MINE_TYPE", 
        "FIPS_CNTY", "AVG_EMPLOYEE_CNT", "COAL_METAL_IND", "US_STATE"
    ]
    
    # Also require either CURRENT_STATUS or MINE_STATUS
    if "CURRENT_STATUS" not in df.columns and "MINE_STATUS" not in df.columns:
        return False, "Missing required field: CURRENT_STATUS or MINE_STATUS"
    
    # Also require either HOURS_WORKED or EMP_HRS_TOTAL
    if "HOURS_WORKED" not in df.columns and "EMP_HRS_TOTAL" not in df.columns:
        return False, "Missing required field: HOURS_WORKED or EMP_HRS_TOTAL"
    
    # Check for missing columns after mapping
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    # Check for empty DataFrame
    if df.empty:
        return False, "Empty data provided"
    
    # Check data types
    try:
        df["YEAR"] = df["YEAR"].astype(int)
        df["AVG_EMPLOYEE_CNT"] = df["AVG_EMPLOYEE_CNT"].astype(float)
        
        # Handle either HOURS_WORKED or EMP_HRS_TOTAL
        if "HOURS_WORKED" in df.columns:
            df["HOURS_WORKED"] = df["HOURS_WORKED"].astype(float)
        if "EMP_HRS_TOTAL" in df.columns:
            df["EMP_HRS_TOTAL"] = df["EMP_HRS_TOTAL"].astype(float)
            
    except Exception as e:
        return False, f"Data type conversion error: {str(e)}"
    
    return True, ""
