#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data loading module for the Mine Safety Injury Rate Prediction model.
This script handles loading and initial cleaning of the MSHA Mine Data.
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Tuple, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                           "MSHA_Mine_Data_2013-2016.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_data.csv")


def load_raw_data() -> pd.DataFrame:
    """
    Load the raw MSHA Mine Data from CSV file.
    
    Returns:
        pd.DataFrame: Raw data loaded from CSV
    """
    logger.info(f"Loading raw data from {RAW_DATA_PATH}")
    try:
        data = pd.read_csv(RAW_DATA_PATH)
        logger.info(f"Successfully loaded {len(data)} records")
        return data
    except Exception as e:
        logger.error(f"Error loading raw data: {e}")
        raise


def perform_basic_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic data cleaning operations.
    
    Args:
        data (pd.DataFrame): Raw data to clean
        
    Returns:
        pd.DataFrame: Cleaned data
    """
    logger.info("Performing basic data cleaning")
    logger.info(f"Initial record count: {len(data)} records")
    
    # Drop rows with missing values in critical columns
    data_nomissing = data.dropna(subset=['MINE_STATUS', 'US_STATE', 'PRIMARY'])
    logger.info(f"Removed {len(data) - len(data_nomissing)} rows with missing values, {len(data_nomissing)} records remaining")
    
    # Debug: Check if this matches the PDF solution (which removed 27 rows)
    if len(data) - len(data_nomissing) == 27:
        logger.info("✓ Missing value removal matches PDF solution (27 rows removed)")
    else:
        logger.warning(f"⚠ Missing value removal differs from PDF solution: {len(data) - len(data_nomissing)} vs 27 expected")
    
    # Create a copy to avoid SettingWithCopyWarning
    data_reduced = data_nomissing.copy()
    
    # Handle field mappings for prediction requests
    field_mappings = {
        'CURRENT_STATUS': 'MINE_STATUS',
        'INJURIES_COUNT': 'NUM_INJURIES',  # Map INJURIES_COUNT to NUM_INJURIES if needed
        'HOURS_WORKED': 'EMP_HRS_TOTAL'    # Map HOURS_WORKED to EMP_HRS_TOTAL if needed
    }
    
    # Apply field mappings
    for source, target in field_mappings.items():
        if source in data_reduced.columns:
            logger.info(f"Found {source} column, value: {data_reduced[source].iloc[0] if len(data_reduced) > 0 else 'N/A'}")
            if target not in data_reduced.columns:
                data_reduced[target] = data_reduced[source]
                logger.info(f"Mapped {source} to {target}, new value: {data_reduced[target].iloc[0] if len(data_reduced) > 0 else 'N/A'}")
            else:
                logger.info(f"Both {source} and {target} exist. {target} value: {data_reduced[target].iloc[0] if len(data_reduced) > 0 else 'N/A'}")
    
    # For prediction requests, we need to handle missing NUM_INJURIES
    if 'NUM_INJURIES' not in data_reduced.columns:
        logger.info("NUM_INJURIES not found - this appears to be a prediction request, not training data")
        # For prediction requests, set NUM_INJURIES to 0 (will be ignored during prediction)
        data_reduced['NUM_INJURIES'] = 0
        logger.info("Added NUM_INJURIES column with default value 0")
    
    # Ensure EMP_HRS_TOTAL is properly set from HOURS_WORKED if needed
    if 'EMP_HRS_TOTAL' not in data_reduced.columns and 'HOURS_WORKED' in data_reduced.columns:
        data_reduced['EMP_HRS_TOTAL'] = data_reduced['HOURS_WORKED']
        logger.info(f"Created EMP_HRS_TOTAL from HOURS_WORKED: {data_reduced['EMP_HRS_TOTAL'].iloc[0] if len(data_reduced) > 0 else 'N/A'}")
    
    # Create target variable: injury rate per 2000 hours (approximately one work year)
    try:
        data_reduced['INJ_RATE_PER2K'] = data_reduced['NUM_INJURIES'] / (data_reduced['EMP_HRS_TOTAL'] / 2000)
        logger.info("Calculated INJ_RATE_PER2K from NUM_INJURIES and EMP_HRS_TOTAL")
    except Exception as e:
        logger.error(f"Error calculating injury rate: {str(e)}")
        # If this is a prediction request, we can continue without the injury rate
        data_reduced['INJ_RATE_PER2K'] = 0
        logger.info("Set default INJ_RATE_PER2K to 0 for prediction")
    
    # Ensure MINE_STATUS exists before filtering
    if 'MINE_STATUS' not in data_reduced.columns:
        if 'CURRENT_STATUS' in data_reduced.columns:
            data_reduced['MINE_STATUS'] = data_reduced['CURRENT_STATUS']
            logger.info(f"Created MINE_STATUS from CURRENT_STATUS before filtering")
        else:
            # If neither MINE_STATUS nor CURRENT_STATUS exists, assume all are active
            data_reduced['MINE_STATUS'] = 'ACTIVE'
            logger.info(f"Neither MINE_STATUS nor CURRENT_STATUS found, defaulting all to 'ACTIVE'")
    
    # Ensure MINE_STATUS is uppercase
    data_reduced['MINE_STATUS'] = data_reduced['MINE_STATUS'].astype(str).str.upper()
    logger.info(f"Converted MINE_STATUS to uppercase: {data_reduced['MINE_STATUS'].unique()}")
    
    # Remove closed mines and mines with low employee hours
    no_good = ["CLOSED BY MSHA", "NON-PRODUCING", "PERMANENTLY ABANDONED", "TEMPORARILY CLOSED"]
    data_reduced2 = data_reduced[~data_reduced['MINE_STATUS'].isin(no_good)]
    logger.info(f"Removed {len(data_reduced) - len(data_reduced2)} closed mines, {len(data_reduced2)} records remaining")
    
    # Filter out mines with less than 2000 employee hours
    data_reduced3 = data_reduced2[data_reduced2['EMP_HRS_TOTAL'] >= 2000]
    logger.info(f"Removed {len(data_reduced2) - len(data_reduced3)} mines with low employee hours, {len(data_reduced3)} records remaining")
    
    # Debug: Check total reduction from original data
    total_reduction = len(data) - len(data_reduced3)
    logger.info(f"Total reduction: {total_reduction} records ({total_reduction/len(data)*100:.2f}% of original data)")
    
    # Debug: Check final record count
    logger.info(f"Final record count after cleaning: {len(data_reduced3)} records")
    
    # Combine coal and non-coal categories
    data_reduced3['ADJ_STATUS'] = 'Other'
    data_reduced3.loc[data_reduced3['MINE_STATUS'] == 'Active', 'ADJ_STATUS'] = 'Open'
    data_reduced3.loc[data_reduced3['MINE_STATUS'] == 'Full-time permanent', 'ADJ_STATUS'] = 'Open'
    data_reduced3.loc[data_reduced3['MINE_STATUS'] == 'Intermittent', 'ADJ_STATUS'] = 'Intermittent'
    data_reduced3 = data_reduced3.drop('MINE_STATUS', axis=1)
    
    # Keep YEAR field for potential temporal analysis
    # data_reduced3 = data_reduced3.drop('YEAR', axis=1)  # Commented out to retain YEAR
    
    logger.info("Basic data cleaning completed")
    return data_reduced3


def save_processed_data(data: pd.DataFrame, output_path: str = PROCESSED_DATA_PATH) -> None:
    """
    Save the processed data to a CSV file.
    
    Args:
        data (pd.DataFrame): Processed data to save
        output_path (str): Path to save the processed data
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    logger.info(f"Saving processed data to {output_path}")
    data.to_csv(output_path, index=False)
    logger.info(f"Saved {len(data)} records to {output_path}")


def load_and_process_data(save: bool = True) -> pd.DataFrame:
    """
    Load and process the data in a single function.
    
    Args:
        save (bool): Whether to save the processed data
        
    Returns:
        pd.DataFrame: Processed data
    """
    raw_data = load_raw_data()
    processed_data = perform_basic_cleaning(raw_data)
    
    if save:
        save_processed_data(processed_data)
    
    return processed_data


if __name__ == "__main__":
    # When run as a script, load and process the data
    processed_data = load_and_process_data(save=True)
    print(f"Processed data shape: {processed_data.shape}")
    print(f"Processed data columns: {processed_data.columns.tolist()}")
    print(f"First few rows of processed data:\n{processed_data.head()}")
