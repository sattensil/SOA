#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature engineering module for the Mine Safety Injury Rate Prediction model.
This script handles feature transformation, encoding, and preparation for modeling.
"""

import pandas as pd
import numpy as np
import os
import logging
import joblib
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Import data loader
from scripts.data_loader import load_and_process_data, PROCESSED_DATA_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
FEATURES_DIR = os.path.join(DATA_DIR, "features")
PREPROCESSOR_PATH = os.path.join(FEATURES_DIR, "preprocessor.joblib")


def create_feature_groups(data: pd.DataFrame) -> Dict[str, list]:
    """
    Identify and group features by type for preprocessing.
    
    Args:
        data (pd.DataFrame): The processed data
        
    Returns:
        Dict[str, list]: Dictionary of feature groups
    """
    # Identify categorical and numerical features
    categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove target and weight variables from features
    exclude_cols = ['NUM_INJURIES', 'EMP_HRS_TOTAL', 'INJ_RATE_PER2K']
    numerical_features = [col for col in numerical_features if col not in exclude_cols]
    
    logger.info(f"Categorical features: {categorical_features}")
    logger.info(f"Numerical features: {numerical_features}")
    
    return {
        'categorical': categorical_features,
        'numerical': numerical_features,
        'target': 'NUM_INJURIES',
        'weight': 'EMP_HRS_TOTAL'
    }


def create_preprocessor(feature_groups: Dict[str, list]) -> ColumnTransformer:
    """
    Create a scikit-learn preprocessor for feature transformation.
    
    Args:
        feature_groups (Dict[str, list]): Dictionary of feature groups
        
    Returns:
        ColumnTransformer: Scikit-learn preprocessor
    """
    logger.info("Creating feature preprocessor")
    
    # Create transformers for different feature types
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Combine transformers into a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, feature_groups['categorical']),
            ('num', numerical_transformer, feature_groups['numerical'])
        ],
        remainder='drop'  # Drop any columns not specified
    )
    
    return preprocessor


def engineer_features(data: pd.DataFrame = None, test_size: float = 0.25, 
                     random_state: int = 1234) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Perform feature engineering and split data into train and test sets.
    
    Args:
        data (pd.DataFrame): Processed data, if None, will load from processed data path
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: Dictionaries containing train and test data and metadata
    """
    # Load data if not provided
    if data is None:
        if os.path.exists(PROCESSED_DATA_PATH):
            logger.info(f"Loading processed data from {PROCESSED_DATA_PATH}")
            data = pd.read_csv(PROCESSED_DATA_PATH)
        else:
            logger.info("Processed data not found, loading and processing raw data")
            data = load_and_process_data(save=True)
    
    # Create feature groups
    feature_groups = create_feature_groups(data)
    
    # Create a combined variable for MINE_CHAR
    data['MINE_CHAR'] = data['TYPE_OF_MINE'] + ' ' + data['COMMODITY']
    data['MINE_CHAR'] = data['MINE_CHAR'].astype('category')
    
    # Take log of AVG_EMP_TOTAL to handle skewness
    data['LOG_AVG_EMP_TOTAL'] = np.log(data['AVG_EMP_TOTAL'])
    data = data.drop('AVG_EMP_TOTAL', axis=1)
    
    # Update feature groups with new features
    feature_groups = create_feature_groups(data)
    
    # Split data into train and test sets
    logger.info(f"Splitting data with test_size={test_size}, random_state={random_state}")
    train, test = train_test_split(
        data, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Reset indices
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    
    logger.info(f"Train size: {len(train)}, Test size: {len(test)}")
    
    # Create feature preprocessor
    preprocessor = create_preprocessor(feature_groups)
    
    # Extract features, target, and weights
    X_train = train.drop(['NUM_INJURIES', 'EMP_HRS_TOTAL', 'INJ_RATE_PER2K'], axis=1)
    X_test = test.drop(['NUM_INJURIES', 'EMP_HRS_TOTAL', 'INJ_RATE_PER2K'], axis=1)
    
    y_train = train['NUM_INJURIES']
    y_test = test['NUM_INJURIES']
    
    weights_train = train['EMP_HRS_TOTAL'] / 2000
    weights_test = test['EMP_HRS_TOTAL'] / 2000
    
    # Fit preprocessor on training data
    logger.info("Fitting preprocessor on training data")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after preprocessing
    feature_names = []
    
    # Get feature names from one-hot encoding
    if hasattr(preprocessor.named_transformers_['cat'].named_steps['onehot'], 'get_feature_names_out'):
        cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(
            feature_groups['categorical']
        )
        feature_names.extend(cat_features)
    
    # Add numerical feature names
    feature_names.extend(feature_groups['numerical'])
    
    # Save preprocessor
    os.makedirs(FEATURES_DIR, exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    logger.info(f"Saved preprocessor to {PREPROCESSOR_PATH}")
    
    # Prepare train and test data dictionaries
    train_data = {
        'X': X_train_processed,
        'y': y_train.values,
        'weights': weights_train.values,
        'feature_names': feature_names,
        'raw_data': train
    }
    
    test_data = {
        'X': X_test_processed,
        'y': y_test.values,
        'weights': weights_test.values,
        'feature_names': feature_names,
        'raw_data': test
    }
    
    return train_data, test_data


def load_preprocessor():
    """
    Load the saved preprocessor.
    
    Returns:
        ColumnTransformer: The loaded preprocessor
    """
    if os.path.exists(PREPROCESSOR_PATH):
        logger.info(f"Loading preprocessor from {PREPROCESSOR_PATH}")
        return joblib.load(PREPROCESSOR_PATH)
    else:
        logger.error(f"Preprocessor not found at {PREPROCESSOR_PATH}")
        raise FileNotFoundError(f"Preprocessor not found at {PREPROCESSOR_PATH}")


if __name__ == "__main__":
    # When run as a script, perform feature engineering
    train_data, test_data = engineer_features()
    
    print(f"Processed training data shape: {train_data['X'].shape}")
    print(f"Processed test data shape: {test_data['X'].shape}")
    print(f"Feature names: {train_data['feature_names'][:10]}... (showing first 10)")
    print(f"Target distribution in training data: {np.bincount(train_data['y'].astype(int))}")
