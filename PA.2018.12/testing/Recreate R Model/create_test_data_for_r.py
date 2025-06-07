#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create test data for R model evaluation
"""

import os
import sys
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project directory to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)
sys.path.append(os.path.join(project_dir, "advanced_implementation", "scripts"))

def create_test_data():
    """Create test data for R model evaluation"""
    # Import the data loading function from display_model_metrics
    try:
        sys.path.append(os.path.join(project_dir, "advanced_implementation", "scripts"))
        from display_model_metrics import load_data, load_model
        logger.info("Successfully imported functions from display_model_metrics")
    except ImportError as e:
        logger.error(f"Error importing from display_model_metrics: {e}")
        raise

    # Load the test data using the same function as in display_model_metrics
    logger.info("Loading test data...")
    test_data = load_data()
    
    # Get the raw features
    # We need to extract the original features before preprocessing
    # First, load the original data
    try:
        from enhanced_feature_engineering import load_and_process_data
        logger.info("Loading original data...")
        data = load_and_process_data()
        
        # Get the test indices
        from sklearn.model_selection import train_test_split
        _, test_indices = train_test_split(
            range(len(data)), 
            test_size=0.25, 
            random_state=1234
        )
        
        # Extract the test data
        test_df = data.iloc[test_indices].copy()
        
        # Add the target variables
        test_df['injury_binary'] = test_data['y_binary']
        test_df['injury_class'] = test_data['y_multi']
        
        # Export to CSV
        output_path = os.path.join(script_dir, "test_data_for_r.csv")
        test_df.to_csv(output_path, index=False)
        logger.info(f"Exported test data to {output_path}")
        
        # Print data summary
        logger.info(f"Test data summary: {len(test_df)} samples")
        logger.info(f"Injury class distribution: {np.bincount(test_data['y_multi'])}")
        logger.info(f"Binary injury distribution: {np.bincount(test_data['y_binary'])}")
        logger.info(f"Raw injury count stats: min={test_data['y'].min()}, max={test_data['y'].max()}, mean={test_data['y'].mean():.4f}")
        
        # Ensure we have the EMP_HRS_TOTAL column for the R model
        if 'EMP_HRS_TOTAL' not in test_df.columns:
            logger.warning("EMP_HRS_TOTAL not found in test data, this is required for the R model")
        
        # Add NUM_INJURIES column if not present
        if 'NUM_INJURIES' not in test_df.columns:
            logger.info("Adding NUM_INJURIES column from raw injury counts")
            test_df['NUM_INJURIES'] = test_data['y']
        
        return output_path
    except Exception as e:
        logger.error(f"Error creating test data: {e}")
        raise

if __name__ == "__main__":
    create_test_data()
