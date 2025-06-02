"""
Simplified prediction utility for the mine safety injury rate prediction API.
This module provides a direct implementation of preprocessing and prediction functions
without relying on serialized preprocessor objects that have dependencies on specific module paths.
"""
import os
import pandas as pd
import numpy as np
import joblib
import logging
import traceback
from typing import List, Tuple, Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path=None, feature_names_path=None):
    """
    Load the model and feature names.
    
    Args:
        model_path (str, optional): Path to the model file. If None, uses default path.
        feature_names_path (str, optional): Path to the feature names file. If None, uses default path.
        
    Returns:
        tuple: (model, feature_names)
    """
    try:
        # Set default paths if not provided
        if model_path is None:
            model_dir = os.environ.get('MODELS_DIR', os.path.join(os.path.dirname(__file__), 'models'))
            model_path = os.path.join(model_dir, 'enhanced_xgboost_model.joblib')
            logger.info(f"Using default model path: {model_path}")
        
        if feature_names_path is None:
            model_dir = os.environ.get('MODELS_DIR', os.path.join(os.path.dirname(__file__), 'models'))
            feature_names_path = os.path.join(model_dir, 'enhanced_feature_names.joblib')
            logger.info(f"Using default feature names path: {feature_names_path}")
        
        # Check if files exist
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            return None, None
        
        if not os.path.exists(feature_names_path):
            logger.error(f"Feature names file not found at {feature_names_path}")
            return None, None
        
        # Load model and feature names
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully")
        
        logger.info(f"Loading feature names from {feature_names_path}")
        feature_names = joblib.load(feature_names_path)
        logger.info(f"Feature names loaded successfully: {len(feature_names)} features")
        
        return model, feature_names
    except Exception as e:
        logger.error(f"Error loading model or feature names: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None

def preprocess_data(data: pd.DataFrame, feature_names: List[str]) -> np.ndarray:
    """
    Preprocess data for prediction without using the serialized preprocessor.
    
    Args:
        data (pd.DataFrame): Raw input data
        feature_names (List[str]): List of feature names required by the model
        
    Returns:
        np.ndarray: Preprocessed features ready for model prediction
    """
    # Create a copy of the data to avoid modifying the original
    df = data.copy()
    
    # Handle field mappings - ensure all required fields are present
    field_mappings = {
        'CURRENT_STATUS': 'MINE_STATUS',  # Map CURRENT_STATUS to MINE_STATUS if needed
    }
    
    for source, target in field_mappings.items():
        if source in df.columns and target not in df.columns:
            df[target] = df[source]
            logger.info(f"Mapped {source} to {target}")
    
    # Initialize the result DataFrame with zeros
    result = pd.DataFrame(0, index=df.index, columns=feature_names)
    
    # Process numeric features
    numeric_features = [
        'YEAR', 'SEAM_HEIGHT', 'AVG_EMP_TOTAL', 
        'PCT_HRS_UNDERGROUND', 'PCT_HRS_SURFACE', 'PCT_HRS_STRIP',
        'PCT_HRS_AUGER', 'PCT_HRS_CULM_BANK', 'PCT_HRS_DREDGE', 
        'PCT_HRS_OTHER_SURFACE', 'PCT_HRS_SHOP_YARD', 'PCT_HRS_MILL_PREP', 
        'PCT_HRS_OFFICE'
    ]
    
    for feature in numeric_features:
        if feature in df.columns and feature in feature_names:
            result[feature] = df[feature].fillna(0)
    
    # Add log transformed employee total
    if 'AVG_EMP_TOTAL' in df.columns and 'LOG_AVG_EMP_TOTAL' in feature_names:
        result['LOG_AVG_EMP_TOTAL'] = np.log1p(df['AVG_EMP_TOTAL'].fillna(0))
    
    # Process categorical features - US_STATE
    if 'US_STATE' in df.columns:
        for state in df['US_STATE'].unique():
            col_name = f'US_STATE_{state}'
            if col_name in feature_names:
                result[col_name] = (df['US_STATE'] == state).astype(int)
    
    # Process COMMODITY
    if 'COMMODITY' in df.columns:
        for commodity in ['Metal', 'Nonmetal', 'Sand & gravel', 'Stone']:
            col_name = f'COMMODITY_{commodity}'
            if col_name in feature_names:
                result[col_name] = (df['COMMODITY'] == commodity).astype(int)
    
    # Process TYPE_OF_MINE
    if 'TYPE_OF_MINE' in df.columns:
        for mine_type in ['Sand & gravel', 'Surface', 'Underground']:
            col_name = f'TYPE_OF_MINE_{mine_type}'
            if col_name in feature_names:
                result[col_name] = (df['TYPE_OF_MINE'] == mine_type).astype(int)
    
    # Process ADJ_STATUS
    if 'ADJ_STATUS' in df.columns:
        if 'ADJ_STATUS_Open' in feature_names:
            result['ADJ_STATUS_Open'] = (df['ADJ_STATUS'] == 'Open').astype(int)
    
    # Process PRIMARY field
    if 'PRIMARY' in df.columns:
        for primary in df['PRIMARY'].unique():
            col_name = f'PRIMARY_{primary}_1'
            if col_name in feature_names:
                result[col_name] = (df['PRIMARY'] == primary).astype(int)
    
    # Ensure all feature columns are present
    for col in feature_names:
        if col not in result.columns:
            result[col] = 0
    
    # Return only the columns needed by the model, in the correct order
    return result[feature_names].values

def predict_injury_rate(data: pd.DataFrame, model=None, feature_names=None) -> np.ndarray:
    """
    Predict injury rate for the given data.
    
    Args:
        data (pd.DataFrame): Data to predict on
        model (optional): Model to use for prediction. If None, loads the default model.
        feature_names (optional): Feature names. If None, loads the default feature names.
        
    Returns:
        np.ndarray: Predicted injury rates
    """
    # Load model and feature names if not provided
    if model is None or feature_names is None:
        model, feature_names = load_model()
        
    if model is None or feature_names is None:
        logger.error("Failed to load model or feature names")
        return np.array([])
    
    # Preprocess data
    X = preprocess_data(data, feature_names)
    
    # Make prediction
    predictions = model.predict(X)
    
    return predictions
