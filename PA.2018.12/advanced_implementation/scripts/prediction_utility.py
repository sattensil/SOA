
import os
import pandas as pd
import numpy as np
import joblib

def load_model_and_preprocessor(model_path=None, preprocessor_path=None, feature_names_path=None):
    """
    Load the model, preprocessor, and feature names.
    
    Args:
        model_path (str, optional): Path to the model file. If None, uses default path.
        preprocessor_path (str, optional): Path to the preprocessor file. If None, uses default path.
        feature_names_path (str, optional): Path to the feature names file. If None, uses default path.
        
    Returns:
        tuple: (model, preprocessor, feature_names)
    """
    # Set default paths if not provided
    if model_path is None:
        model_dir = os.environ.get('MODELS_DIR', os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models'))
        model_path = os.path.join(model_dir, 'enhanced_xgboost_model.joblib')
    
    if preprocessor_path is None:
        features_dir = os.environ.get('FEATURES_DIR', os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'features'))
        preprocessor_path = os.path.join(features_dir, 'enhanced_preprocessor.joblib')
    
    if feature_names_path is None:
        model_dir = os.environ.get('MODELS_DIR', os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models'))
        feature_names_path = os.path.join(model_dir, 'enhanced_feature_names.joblib')
    
    # Load model, preprocessor, and feature names
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    feature_names = joblib.load(feature_names_path)
    
    return model, preprocessor, feature_names

def preprocess_data(data, preprocessor):
    """
    Preprocess data using the saved preprocessor.
    
    Args:
        data (pd.DataFrame): Data to preprocess
        preprocessor: Fitted preprocessor
        
    Returns:
        np.ndarray: Preprocessed features
    """
    # Apply preprocessor
    X = preprocessor.transform(data)
    return X

def predict_injury_rate(data, model=None, preprocessor=None, feature_names=None):
    """
    Predict injury rate for the given data.
    
    Args:
        data (pd.DataFrame): Data to predict on
        model (optional): Model to use for prediction. If None, loads the default model.
        preprocessor (optional): Preprocessor to use. If None, loads the default preprocessor.
        feature_names (optional): Feature names. If None, loads the default feature names.
        
    Returns:
        np.ndarray: Predicted injury rates
    """
    # Load model, preprocessor, and feature names if not provided
    if model is None or preprocessor is None or feature_names is None:
        model, preprocessor, feature_names = load_model_and_preprocessor()
    
    # Preprocess data
    X = preprocess_data(data, preprocessor)
    
    # Make prediction
    predictions = model.predict(X)
    
    return predictions
