#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Export test data to CSV for R model evaluation
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
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)
sys.path.append(os.path.join(project_dir, "advanced_implementation", "scripts"))

def load_data():
    """Load the test data for evaluation using the same approach as in display_model_metrics.py"""
    # Import the data loading function from the enhanced_feature_engineering module
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    sys.path.append(project_dir)
    sys.path.append(os.path.join(project_dir, "advanced_implementation", "scripts"))
    
    try:
        # Try direct import first
        from advanced_implementation.scripts.enhanced_feature_engineering import engineer_enhanced_features
        logger.info("Imported engineer_enhanced_features from advanced_implementation.scripts")
    except ImportError:
        try:
            # Try relative import
            from enhanced_feature_engineering import engineer_enhanced_features
            logger.info("Imported engineer_enhanced_features directly")
        except ImportError:
            # Try alternative import paths
            try:
                sys.path.append(os.path.join(project_dir, "advanced_implementation"))
                from scripts.enhanced_feature_engineering import engineer_enhanced_features
                logger.info("Imported engineer_enhanced_features from scripts")
            except ImportError:
                # Last resort - look for the module directly
                logger.error("Could not find enhanced_feature_engineering module")
                raise ImportError("Could not find enhanced_feature_engineering module")
    
    # Load the data using the same function as in training
    train_data, test_data = engineer_enhanced_features()
    
    # Create binary target
    y_test_binary = (test_data['y'] > 0).astype(int)
    
    # Create multi-class target (0: no injury, 1: one injury, 2: two injuries, 3+: three or more injuries)
    y_test_multi = np.zeros_like(test_data['y'], dtype=int)
    y_test_multi[test_data['y'] == 0] = 0  # No injury
    y_test_multi[test_data['y'] == 1] = 1  # One injury
    y_test_multi[test_data['y'] == 2] = 2  # Two injuries
    y_test_multi[test_data['y'] >= 3] = 3  # Three or more injuries
    
    # Print the distribution of y_test_multi
    print("\nMulti-class target distribution:")
    print(np.bincount(y_test_multi))
    
    return test_data, y_test_binary, y_test_multi

def export_test_data():
    """Export test data to CSV for R model evaluation"""
    print("Loading data...")
    test_data, y_test_binary, y_test_multi = load_data()
    
    # Get the features
    X_test = test_data['X']
    
    # Create a DataFrame with the original features
    # First, get the feature names
    feature_names = test_data.get('feature_names', [])
    if not feature_names:
        # If feature names not available, use generic column names
        feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
    
    # Create DataFrame with features
    test_df = pd.DataFrame(X_test, columns=feature_names)
    
    # Add target variable
    test_df["NUM_INJURIES"] = test_data["y"]
    
    # Add binary and multi-class targets
    test_df["injury_binary"] = y_test_binary
    test_df["injury_class"] = y_test_multi
    
    # Export to CSV
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data.csv")
    test_df.to_csv(output_path, index=False)
    print(f"Exported test data to {output_path}")
    
    # Print data summary
    print("\nTest data summary:")
    print(f"Number of samples: {len(test_df)}")
    print("\nInjury count distribution:")
    print(test_df["NUM_INJURIES"].value_counts().sort_index())
    print("\nInjury class distribution:")
    print(test_df["injury_class"].value_counts().sort_index())
    
    return output_path

if __name__ == "__main__":
    export_test_data()
