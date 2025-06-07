#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced feature engineering module for the Mine Safety Injury Rate Prediction model.
This script combines our existing feature engineering with additional techniques
from the exam solutions to improve model performance.
"""

import pandas as pd
import numpy as np
import os
import logging
import joblib
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Import modules
try:
    # When imported as a module
    from .data_loader import PROCESSED_DATA_PATH
except ImportError:
    # When run as a script
    from data_loader import PROCESSED_DATA_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
FEATURES_DIR = os.path.join(DATA_DIR, "features")
PREPROCESSOR_PATH = os.path.join(FEATURES_DIR, "enhanced_preprocessor.joblib")
TRAIN_DATA_PATH = os.path.join(FEATURES_DIR, "enhanced_train_data.joblib")
TEST_DATA_PATH = os.path.join(FEATURES_DIR, "enhanced_test_data.joblib")


def split_primary_column(X, train=None):
    """
    Split the PRIMARY column into dummy variables.
    
    Args:
        X (pd.DataFrame): Data with PRIMARY column
        train (pd.DataFrame, optional): Training data for consistent columns. Defaults to None.
        
    Returns:
        pd.DataFrame: Dummy variables for PRIMARY column
    """
    if train is None:
        train = X
    
    # Debug: Check PRIMARY field values
    if logger.level <= logging.DEBUG:
        logger.debug(f"PRIMARY field unique values: {X['PRIMARY'].nunique()}")
        logger.debug(f"Sample PRIMARY values: {X['PRIMARY'].sample(min(5, len(X))).values}")
        
    known_columns = train['PRIMARY'].astype(str).str.get_dummies(sep=', ').columns
    primary_dummies = X['PRIMARY'].astype(str).str.get_dummies(sep=', ')
    
    # Debug: Check dummy variables created
    if logger.level <= logging.DEBUG:
        logger.debug(f"Number of PRIMARY dummy variables: {len(known_columns)}")
        logger.debug(f"PRIMARY dummy variable names (first 5): {list(known_columns)[:5]}")
    
    # This adds missing columns (filling with 0) and removes extra columns
    primary_dummies = primary_dummies.reindex(columns=known_columns, fill_value=0)
    
    # Debug: Verify dummy variable creation
    logger.info(f"Created {primary_dummies.shape[1]} dummy variables from PRIMARY field")
    
    return primary_dummies


def create_enhanced_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create enhanced features based on the exam solutions and additional transformations.
    
    Args:
        data (pd.DataFrame): Processed data
        
    Returns:
        pd.DataFrame: Data with enhanced features
    """
    logger.info("Creating enhanced features")
    
    # Create a copy to avoid modifying the original
    enhanced_data = data.copy()
    
    # Apply the same filtering criteria as the official solution
    logger.info("Applying official solution filtering criteria")
    
    # Remove closed or similar to closed mine status (if not already done)
    no_good = ["Closed by MSHA", "Non-producing", "Permanently abandoned", "Temporarily closed"]
    if 'MINE_STATUS' in enhanced_data.columns:
        initial_count = len(enhanced_data)
        enhanced_data = enhanced_data[~enhanced_data['MINE_STATUS'].isin(no_good)]
        logger.info(f"Removed {initial_count - len(enhanced_data)} records with closed mine status")
    
    # Remove low employee hours (if not already done)
    if 'EMP_HRS_TOTAL' in enhanced_data.columns:
        initial_count = len(enhanced_data)
        enhanced_data = enhanced_data[enhanced_data['EMP_HRS_TOTAL'] >= 2000]
        logger.info(f"Removed {initial_count - len(enhanced_data)} records with low employee hours")
    
    # Create ADJ_STATUS variable as in the official solution
    if 'MINE_STATUS' in enhanced_data.columns:
        logger.info("Creating ADJ_STATUS variable as in the official solution")
        enhanced_data['ADJ_STATUS'] = 'Other'
        enhanced_data.loc[enhanced_data['MINE_STATUS'] == 'Active', 'ADJ_STATUS'] = 'Open'
        enhanced_data.loc[enhanced_data['MINE_STATUS'] == 'Full-time permanent', 'ADJ_STATUS'] = 'Open'
        enhanced_data.loc[enhanced_data['MINE_STATUS'] == 'Intermittent', 'ADJ_STATUS'] = 'Intermittent'
        enhanced_data['ADJ_STATUS'] = enhanced_data['ADJ_STATUS'].astype('category')
        logger.info(f"ADJ_STATUS distribution: {enhanced_data['ADJ_STATUS'].value_counts()}")
    
    # Log transform employee count (as noted in the exam solutions)
    # This transformation is valuable as it helps with the skewed distribution
    enhanced_data['LOG_AVG_EMP_TOTAL'] = np.log(enhanced_data['AVG_EMP_TOTAL'] + 1)  # Add 1 to handle zeros
    logger.info("Created log-transformed employee count feature")
    
    # Create interaction terms between employee count and mining activities
    # These interactions were identified as significant in the exam solution
    logger.info("Creating interaction terms between employee count and mining activities")
    
    # 1. Interaction between employee count and underground hours
    if 'PCT_HRS_UNDERGROUND' in enhanced_data.columns:
        enhanced_data['EMP_X_UNDERGROUND'] = enhanced_data['LOG_AVG_EMP_TOTAL'] * enhanced_data['PCT_HRS_UNDERGROUND']
        logger.info("Created interaction: EMP_X_UNDERGROUND")
    
    # 2. Interaction between employee count and strip mining hours
    if 'PCT_HRS_STRIP' in enhanced_data.columns:
        enhanced_data['EMP_X_STRIP'] = enhanced_data['LOG_AVG_EMP_TOTAL'] * enhanced_data['PCT_HRS_STRIP']
        logger.info("Created interaction: EMP_X_STRIP")
    
    # 3. Create combined mine type and commodity feature
    if 'MINE_TYPE' in enhanced_data.columns and 'COMMODITY' in enhanced_data.columns:
        logger.info("Creating combined mine type and commodity feature")
        enhanced_data['MINE_CHAR'] = enhanced_data['MINE_TYPE'] + '_' + enhanced_data['COMMODITY']
        logger.info(f"Created MINE_CHAR with {enhanced_data['MINE_CHAR'].nunique()} unique combinations")
        
        # Log the distribution of the combined feature
        top_combinations = enhanced_data['MINE_CHAR'].value_counts().head(10)
        logger.info(f"Top mine characteristic combinations:\n{top_combinations}")
    
    # 4. Additional interaction terms between employee count and other mining features
    logger.info("Creating additional interaction terms for improved minority class prediction")
    
    # Interaction between employee count and surface mining hours
    if 'PCT_HRS_SURFACE' in enhanced_data.columns:
        enhanced_data['EMP_X_SURFACE'] = enhanced_data['LOG_AVG_EMP_TOTAL'] * enhanced_data['PCT_HRS_SURFACE']
        logger.info("Created interaction: EMP_X_SURFACE")
    
    # Interaction between employee count and mill hours
    if 'PCT_HRS_MILL' in enhanced_data.columns:
        enhanced_data['EMP_X_MILL'] = enhanced_data['LOG_AVG_EMP_TOTAL'] * enhanced_data['PCT_HRS_MILL']
        logger.info("Created interaction: EMP_X_MILL")
    
    # 5. Create polynomial features for key numeric variables
    logger.info("Creating polynomial transformations of key numeric features")
    
    # Square of employee count (to capture non-linear relationships)
    enhanced_data['AVG_EMP_TOTAL_SQ'] = enhanced_data['AVG_EMP_TOTAL'] ** 2
    
    # 6. Create ratio features that might be predictive
    logger.info("Creating ratio features")
    
    # Hours per employee ratio (if both features exist)
    if 'EMP_HRS_TOTAL' in enhanced_data.columns and 'AVG_EMP_TOTAL' in enhanced_data.columns:
        # Avoid division by zero
        mask = enhanced_data['AVG_EMP_TOTAL'] > 0
        enhanced_data.loc[mask, 'HRS_PER_EMP'] = enhanced_data.loc[mask, 'EMP_HRS_TOTAL'] / enhanced_data.loc[mask, 'AVG_EMP_TOTAL']
        # Fill NaN values with median
        median_hrs_per_emp = enhanced_data.loc[mask, 'HRS_PER_EMP'].median()
        enhanced_data['HRS_PER_EMP'] = enhanced_data['HRS_PER_EMP'].fillna(median_hrs_per_emp)
        logger.info(f"Created HRS_PER_EMP ratio feature with median value: {median_hrs_per_emp:.2f}")
    
    # 7. Create risk score feature combining multiple risk factors
    # This is a domain-specific combined feature that weights different risk factors
    logger.info("Creating combined risk score feature")
    
    risk_score = np.zeros(len(enhanced_data))
    
    # Add risk from underground mining (higher weight due to higher risk)
    if 'PCT_HRS_UNDERGROUND' in enhanced_data.columns:
        risk_score += enhanced_data['PCT_HRS_UNDERGROUND'] * 2.0
    
    # Add risk from surface mining
    if 'PCT_HRS_SURFACE' in enhanced_data.columns:
        risk_score += enhanced_data['PCT_HRS_SURFACE'] * 1.0
    
    # Add risk from strip mining
    if 'PCT_HRS_STRIP' in enhanced_data.columns:
        risk_score += enhanced_data['PCT_HRS_STRIP'] * 1.5
    
    # Add risk from mill operations
    if 'PCT_HRS_MILL' in enhanced_data.columns:
        risk_score += enhanced_data['PCT_HRS_MILL'] * 0.8
    
    # Scale by log of employee count to account for exposure
    if 'LOG_AVG_EMP_TOTAL' in enhanced_data.columns:
        risk_score *= (1.0 + enhanced_data['LOG_AVG_EMP_TOTAL'] * 0.1)
    
    enhanced_data['RISK_SCORE'] = risk_score
    logger.info(f"Created RISK_SCORE feature with range: {enhanced_data['RISK_SCORE'].min():.2f} to {enhanced_data['RISK_SCORE'].max():.2f}")
    
    # 8. Create binary flags for high-risk categories
    logger.info("Creating binary flags for high-risk categories")
    
    # Flag for primarily underground mines (high risk)
    if 'PCT_HRS_UNDERGROUND' in enhanced_data.columns:
        enhanced_data['HIGH_UNDERGROUND'] = (enhanced_data['PCT_HRS_UNDERGROUND'] > 0.7).astype(int)
        logger.info(f"Created HIGH_UNDERGROUND flag, count: {enhanced_data['HIGH_UNDERGROUND'].sum()}")
    
    # Flag for large operations (different risk profile)
    if 'AVG_EMP_TOTAL' in enhanced_data.columns:
        # Use 75th percentile as threshold for "large" operations
        large_threshold = enhanced_data['AVG_EMP_TOTAL'].quantile(0.75)
        enhanced_data['LARGE_OPERATION'] = (enhanced_data['AVG_EMP_TOTAL'] > large_threshold).astype(int)
        logger.info(f"Created LARGE_OPERATION flag (>{large_threshold:.1f} employees), count: {enhanced_data['LARGE_OPERATION'].sum()}")
    
    # 9. Add features from the official solution that we might be missing
    logger.info("Adding additional features from the official solution")
    
    # Add SEAM_HEIGHT if available (significant in official solution)
    if 'SEAM_HEIGHT' in enhanced_data.columns:
        logger.info("SEAM_HEIGHT already present in the data")
    
    # Create flag for high office hours (protective factor in official solution)
    if 'PCT_HRS_OFFICE' in enhanced_data.columns:
        # Office hours have a strong negative association with injuries
        enhanced_data['HIGH_OFFICE_HRS'] = (enhanced_data['PCT_HRS_OFFICE'] > 0.5).astype(int)
        logger.info(f"Created HIGH_OFFICE_HRS flag, count: {enhanced_data['HIGH_OFFICE_HRS'].sum()}")
        
        # Create interaction between office hours and employee count
        enhanced_data['EMP_X_OFFICE'] = enhanced_data['LOG_AVG_EMP_TOTAL'] * enhanced_data['PCT_HRS_OFFICE']
        logger.info("Created interaction: EMP_X_OFFICE")
    
    # Ensure MINE_CHAR is properly encoded with the same reference level as official solution
    if 'MINE_CHAR' in enhanced_data.columns:
        # Convert to categorical type if it's not already
        if not pd.api.types.is_categorical_dtype(enhanced_data['MINE_CHAR']):
            enhanced_data['MINE_CHAR'] = enhanced_data['MINE_CHAR'].astype('category')
            
        # Check if 'Sand & gravel Sand & gravel' exists in the data
        reference_level = "Sand & gravel_Sand & gravel"
        alt_reference_level = "Sand & gravel_Sand & gravel"
        
        if reference_level in enhanced_data['MINE_CHAR'].cat.categories:
            # Reorder categories to make this the reference level
            categories = enhanced_data['MINE_CHAR'].cat.categories.tolist()
            categories.remove(reference_level)
            categories = [reference_level] + categories
            enhanced_data['MINE_CHAR'] = enhanced_data['MINE_CHAR'].cat.reorder_categories(categories)
            logger.info(f"Set '{reference_level}' as reference level for MINE_CHAR")
        elif alt_reference_level in enhanced_data['MINE_CHAR'].cat.categories:
            # Try alternative format
            categories = enhanced_data['MINE_CHAR'].cat.categories.tolist()
            categories.remove(alt_reference_level)
            categories = [alt_reference_level] + categories
            enhanced_data['MINE_CHAR'] = enhanced_data['MINE_CHAR'].cat.reorder_categories(categories)
            logger.info(f"Set '{alt_reference_level}' as reference level for MINE_CHAR")
        else:
            logger.warning(f"Reference level '{reference_level}' not found in MINE_CHAR categories")
    
    logger.info(f"Enhanced features created: {enhanced_data.shape[1]} features, {enhanced_data.shape[0]} records")
    return enhanced_data


def identify_feature_groups(data: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Identify categorical and numerical features.
    
    Args:
        data (pd.DataFrame): Data with features
        
    Returns:
        Tuple[List[str], List[str], List[str]]: Lists of categorical features, numerical features, and special features
    """
    # Identify categorical features (object and category types), excluding PRIMARY which gets special handling
    categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_features = [col for col in categorical_features if col != 'PRIMARY']
    
    # Identify numerical features (exclude the target and employee hours which is used as offset)
    numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_features = [col for col in numerical_features if col not in 
                         ['NUM_INJURIES', 'INJ_RATE_PER2K', 'EMP_HRS_TOTAL']]
    
    # Special handling for PRIMARY field
    special_features = ['PRIMARY'] if 'PRIMARY' in data.columns else []
    
    logger.info(f"Categorical features: {categorical_features}")
    logger.info(f"Numerical features: {numerical_features}")
    logger.info(f"Special features: {special_features}")
    
    return categorical_features, numerical_features, special_features


def create_preprocessor(categorical_features: List[str], numerical_features: List[str], special_features: List[str], train_data: pd.DataFrame = None) -> ColumnTransformer:
    """
    Create a preprocessor for the features.
    
    Args:
        categorical_features (List[str]): List of categorical feature names
        numerical_features (List[str]): List of numerical feature names
        special_features (List[str]): List of special feature names (e.g., PRIMARY)
        train_data (pd.DataFrame, optional): Training data for PRIMARY field parsing. Defaults to None.
        
    Returns:
        ColumnTransformer: Preprocessor for the features
    """
    logger.info("Creating feature preprocessor")
    
    # Create transformers for categorical and numerical features
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
    numerical_transformer = StandardScaler()
    
    # Create transformers list
    transformers = [
        ('cat', categorical_transformer, categorical_features),
        ('num', numerical_transformer, numerical_features)
    ]
    
    # Add PRIMARY field transformer if it exists in special_features
    if 'PRIMARY' in special_features:
        primary_transformer = Pipeline([
            ('split', FunctionTransformer(func=split_primary_column, kw_args={'train': train_data})),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        transformers.append(('primary_dummies', primary_transformer, ['PRIMARY']))
    
    # Combine transformers into a preprocessor
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'  # Drop any columns not specified
    )
    
    return preprocessor


def engineer_enhanced_features(test_size: float = 0.25, random_state: int = 1234) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Engineer enhanced features for the model.
    
    Args:
        test_size (float, optional): Proportion of data to use for testing. Defaults to 0.25.
        random_state (int, optional): Random state for reproducibility. Defaults to 1234.
        
    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: Dictionaries containing training and test data
    """
    # Load processed data
    if os.path.exists(PROCESSED_DATA_PATH):
        logger.info(f"Loading processed data from {PROCESSED_DATA_PATH}")
        data = pd.read_csv(PROCESSED_DATA_PATH)
    else:
        logger.error(f"Processed data file not found at {PROCESSED_DATA_PATH}")
        logger.info("Please run data_loader.py first to generate the processed data file")
        raise FileNotFoundError(f"Processed data file not found at {PROCESSED_DATA_PATH}. Run data_loader.py first.")
    
    # Create enhanced features
    enhanced_data = create_enhanced_features(data)
    
    # Identify feature groups
    categorical_features, numerical_features, special_features = identify_feature_groups(enhanced_data)
    
    # Split data into training and testing sets
    # Use stratified sampling based on injury rate (binned)
    logger.info(f"Splitting data with test_size={test_size}, random_state={random_state}")
    logger.info(f"Total records before splitting: {len(enhanced_data)}")
    
    # Debug: Check distribution of injury rates
    rate_summary = enhanced_data['INJ_RATE_PER2K'].describe()
    logger.info(f"Injury rate summary statistics:\n{rate_summary}")
    
    # Create bins for stratification (as direct stratification on continuous target is not possible)
    bins = [0, 0.05, 0.1, 0.2, 0.5, 1, float('inf')]
    enhanced_data['INJ_RATE_BIN'] = pd.cut(enhanced_data['INJ_RATE_PER2K'], bins=bins)
    
    # Debug: Check if there are any NaN values in the bins
    nan_count = enhanced_data['INJ_RATE_BIN'].isna().sum()
    logger.info(f"NaN values in injury rate bins: {nan_count}")
    
    # Handle NaN values by creating a separate bin for them
    if nan_count > 0:
        logger.info("Creating a separate bin for NaN values")
        # Convert to string category for easier handling
        enhanced_data['INJ_RATE_BIN'] = enhanced_data['INJ_RATE_BIN'].astype(str)
        # Replace 'nan' with 'Zero' for zero injury rates
        enhanced_data.loc[enhanced_data['INJ_RATE_BIN'] == 'nan', 'INJ_RATE_BIN'] = 'Zero'
    
    # Debug: Check bin distribution
    bin_counts = enhanced_data['INJ_RATE_BIN'].value_counts(sort=False)
    logger.info(f"Injury rate bin distribution after handling NaNs:\n{bin_counts}")
    
    # Split the data
    train_idx, test_idx = train_test_split(
        enhanced_data.index,
        test_size=test_size,
        random_state=random_state,
        stratify=enhanced_data['INJ_RATE_BIN']
    )
    
    train_data = enhanced_data.loc[train_idx].drop('INJ_RATE_BIN', axis=1)
    test_data = enhanced_data.loc[test_idx].drop('INJ_RATE_BIN', axis=1)
    
    # Debug: Verify train/test split sizes
    logger.info(f"Train size: {len(train_data)} ({len(train_data)/len(enhanced_data)*100:.2f}%), Test size: {len(test_data)} ({len(test_data)/len(enhanced_data)*100:.2f}%)")
    
    # Debug: Verify that train + test = total
    if len(train_data) + len(test_data) == len(enhanced_data):
        logger.info("✓ Train/test split verification successful")
    else:
        logger.warning(f"⚠ Train/test split verification failed: {len(train_data)} + {len(test_data)} != {len(enhanced_data)}")
    
    # Create and fit preprocessor
    preprocessor = create_preprocessor(categorical_features, numerical_features, special_features, train_data)
    
    logger.info("Fitting preprocessor on training data")
    X_train = preprocessor.fit_transform(train_data)
    X_test = preprocessor.transform(test_data)
    
    # Get feature names after preprocessing
    feature_names = []
    
    # Debug: Log transformer names to help identify all components
    transformer_names = [name for name, _, _ in preprocessor.transformers_]
    logger.info(f"Transformer names: {transformer_names}")
    
    # Get categorical feature names
    if 'cat' in transformer_names:
        cat_idx = [i for i, (name, _, _) in enumerate(preprocessor.transformers_) if name == 'cat'][0]
        cat_transformer = preprocessor.transformers_[cat_idx][1]
        for i, category in enumerate(categorical_features):
            for cat in cat_transformer.categories_[i][1:]:  # Skip first category (dropped)
                feature_names.append(f"{category}_{cat}")
        logger.info(f"Added {len(feature_names)} categorical feature names")
    
    # Get numerical feature names
    feature_names.extend(numerical_features)
    logger.info(f"Added {len(numerical_features)} numerical feature names")
    
    # Get PRIMARY feature names if available
    if 'primary_dummies' in transformer_names:
        primary_idx = [i for i, (name, _, _) in enumerate(preprocessor.transformers_) if name == 'primary_dummies'][0]
        primary_transformer = preprocessor.transformers_[primary_idx][1]
        # The OneHotEncoder is the second step in the pipeline
        primary_encoder = primary_transformer.named_steps['onehot']
        # Get the feature names from the dummy variables
        primary_columns = primary_encoder.get_feature_names_out()
        for col in primary_columns:
            feature_names.append(f"PRIMARY_{col}")
        logger.info(f"Added {len(primary_columns)} PRIMARY feature names")
    
    # Verify feature names match the transformed data shape
    logger.info(f"Total feature names: {len(feature_names)}")
    logger.info(f"X_train shape: {X_train.shape[1]}")
    
    # If there's still a mismatch, generate generic feature names
    if len(feature_names) != X_train.shape[1]:
        logger.warning(f"Mismatch between feature names ({len(feature_names)}) and X_train shape ({X_train.shape[1]})")
        logger.warning("Using generic feature names")
        feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
    
    logger.info(f"Total features after preprocessing: {X_train.shape[1]}")
    
    # Create sample weights (employee hours)
    train_weights = train_data['EMP_HRS_TOTAL'] / 2000
    test_weights = test_data['EMP_HRS_TOTAL'] / 2000
    
    # Save preprocessor
    os.makedirs(FEATURES_DIR, exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    logger.info(f"Saved preprocessor to {PREPROCESSOR_PATH}")
    
    # Prepare return dictionaries
    train_dict = {
        'X': X_train,
        'y': train_data['INJ_RATE_PER2K'].values,
        'weights': train_weights.values,
        'feature_names': feature_names,
        'raw_data': train_data
    }
    
    test_dict = {
        'X': X_test,
        'y': test_data['INJ_RATE_PER2K'].values,
        'weights': test_weights.values,
        'feature_names': feature_names,
        'raw_data': test_data
    }
    
    # Save train and test data
    joblib.dump(train_dict, TRAIN_DATA_PATH)
    joblib.dump(test_dict, TEST_DATA_PATH)
    
    return train_dict, test_dict


def main():
    """
    Main function to engineer enhanced features.
    
    This function checks if the processed data file exists and then engineers enhanced features.
    If the processed data file doesn't exist, it instructs the user to run data_loader.py first.
    """
    try:
        # Check if processed data file exists
        if not os.path.exists(PROCESSED_DATA_PATH):
            print(f"ERROR: Processed data file not found at {PROCESSED_DATA_PATH}")
            print("Please run data_loader.py first to generate the processed data file:")
            print("python scripts/data_loader.py")
            return
            
        train_data, test_data = engineer_enhanced_features()
        print(f"Enhanced features engineered: {train_data['X'].shape}, {test_data['X'].shape}")
        print(f"Feature names: {train_data['feature_names'][:5]}... (total: {len(train_data['feature_names'])})")
    except Exception as e:
        print(f"ERROR: {str(e)}")
        print("Please ensure data_loader.py has been run first to generate the processed data file.")


if __name__ == "__main__":
    main()
