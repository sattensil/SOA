"""
Utility functions for the FastAPI application.
"""
import os
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union

# Import MLflow utilities if available
try:
    from scripts.mlflow_utils import load_model_from_registry, get_active_model_version
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

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

def load_model_artifacts(version: Optional[str] = None) -> Tuple[Optional[object], Optional[object], Optional[List[str]]]:
    """
    Load the model, preprocessor, and feature names.
    
    Args:
        version (str, optional): Model version to load from MLflow registry
                                 If None, loads the active version or falls back to local files
    
    Returns:
        Tuple of (model, preprocessor, feature_names)
    """
    try:
        # Try to load from MLflow registry if available
        if MLFLOW_AVAILABLE:
            try:
                # If version is specified, load that version
                if version is not None:
                    logger.info(f"Loading model version {version} from MLflow registry")
                    model, feature_names = load_model_from_registry(version)
                    return model, None, feature_names
                
                # Try to load active version
                active_version = get_active_model_version()
                if active_version != "latest":
                    logger.info(f"Loading active model version {active_version} from MLflow registry")
                    model, feature_names = load_model_from_registry(active_version)
                    return model, None, feature_names
            except Exception as e:
                logger.warning(f"Could not load model from MLflow registry: {str(e)}")
                logger.info("Falling back to local model files")
        
        # Fall back to prediction_utility for local files
        try:
            # Ensure the scripts directory is in PYTHONPATH or use relative import if appropriate
            # For now, assuming 'scripts' is a top-level directory accessible via PYTHONPATH
            from scripts.prediction_utility import load_model_and_preprocessor
        except ImportError as e_import:
            logger.error(f"Failed to import load_model_and_preprocessor from scripts.prediction_utility: {e_import}")
            # If this import fails, the API won't be able to load local models.
            # Depending on requirements, you might re-raise or return None to indicate failure.
            raise # Or return None, None, None if that's preferred error handling

        logger.info("Loading model from local files using prediction_utility")
        model, preprocessor, feature_names = load_model_and_preprocessor()
        # Return the model, preprocessor, and feature names
        return model, preprocessor, feature_names
    except Exception as e:
        logger.error(f"Error loading model artifacts: {str(e)}")
        return None, None, None

def preprocess_input_data(data: pd.DataFrame, preprocessor: object, feature_names: List[str]) -> np.ndarray:
    """
    Preprocess input data using the preprocessor from scripts.
    
    Args:
        data (pd.DataFrame): Input data to preprocess
        preprocessor (object): Preprocessor object from load_model_artifacts
        feature_names (List[str]): List of feature names
        
    Returns:
        np.ndarray: Preprocessed features
    """
    try:
        # First try to use the preprocessor directly if available
        if preprocessor is not None:
            logger.info("Using provided preprocessor for data transformation")
            X = preprocessor.transform(data)
            return X
        
        # If no preprocessor is available (e.g., when loading from MLflow),
        # use the prediction_utility's preprocess_data function
        from scripts.prediction_utility import preprocess_data
        logger.info("Using prediction_utility.preprocess_data for preprocessing")
        X = preprocess_data(data, feature_names)
        return X
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        # Fallback to returning empty array with correct shape
        return np.zeros((data.shape[0], len(feature_names)))

def get_model_metadata(version: Optional[str] = None) -> Dict[str, Union[str, int, List[str]]]:
    """
    Get metadata about the model.
    
    Args:
        version (str, optional): Model version to get metadata for
                                 If None, gets metadata for the active version
    
    Returns:
        Dictionary with model metadata
    """
    # Try to get active version from MLflow if available
    active_version = "latest"
    if MLFLOW_AVAILABLE:
        try:
            active_version = get_active_model_version()
        except Exception as e:
            logger.warning(f"Could not get active model version from MLflow: {str(e)}")
    
    # Load model artifacts for the specified version or active version
    model, _, feature_names = load_model_artifacts(version)
    
    # Get model version info from MLflow if available
    model_version = version or active_version
    last_updated = "2025-06-02"  # Default value
    
    # Get additional metadata from MLflow if available
    if MLFLOW_AVAILABLE and model_version != "latest":
        try:
            from scripts.mlflow_utils import list_model_versions
            versions = list_model_versions()
            for v in versions:
                if v["version"] == model_version:
                    last_updated = pd.to_datetime(v["last_updated_timestamp"], unit="ms").strftime("%Y-%m-%d")
                    break
        except Exception as e:
            logger.warning(f"Could not get model version info from MLflow: {str(e)}")
    
    return {
        "model_type": "XGBoost",
        "model_version": f"enhanced_xgboost_v{model_version}",
        "feature_count": len(feature_names) if feature_names else 0,
        "top_features": feature_names[:10] if feature_names else [],
        "objective": "count:poisson",
        "last_updated": last_updated
    }

def validate_input_data(data: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate input data for prediction.
    
    Args:
        data: Input DataFrame
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    import logging
    df = data.copy()
    field_mappings = {
        'CURRENT_STATUS': 'MINE_STATUS',
        'INJURIES_COUNT': 'NUM_INJURIES',
        'HOURS_WORKED': 'EMP_HRS_TOTAL'
    }
    for source, target in field_mappings.items():
        if source in df.columns:
            df[target] = df[source]
            logging.info(f"Mapped {source} to {target} during validation")
    required_fields = ["YEAR", "AVG_EMPLOYEE_CNT", "HOURS_WORKED"]
    for field in required_fields:
        if field not in df.columns:
            logging.error(f"Missing required field: {field}")
            return False, f"Missing required field: {field}"
    if df.empty:
        logging.error("Empty data provided")
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
