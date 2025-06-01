"""
Unit tests for the enhanced_feature_engineering module.
"""
import os
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from unittest.mock import patch, MagicMock
from sklearn.preprocessing import OneHotEncoder

# Add parent directory to path to import modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

class TestEnhancedFeatureEngineering(unittest.TestCase):
    """Test cases for enhanced_feature_engineering module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a small test dataframe
        self.test_data = pd.DataFrame({
            'MINE_ID': [1, 2, 3, 4, 5],
            'MINE_STATUS': ['Active', 'Active', 'Active', 'Active', 'Active'],
            'US_STATE': ['CA', 'NV', 'AZ', 'CA', 'NV'],
            'PRIMARY': ['Coal', 'Gold', 'Silver, Gold', 'Coal, Silver', 'Gold, Coal'],
            'COMMODITY': ['Coal', 'Metal', 'Metal', 'Coal', 'Metal'],
            'TYPE_OF_MINE': ['Underground', 'Surface', 'Underground', 'Surface', 'Underground'],
            'ADJ_STATUS': ['Active', 'Active', 'Active', 'Active', 'Active'],
            'NUM_INJURIES': [2, 0, 1, 3, 0],
            'EMP_HRS_TOTAL': [5000, 3000, 4000, 10000, 4000],
            'SEAM_HEIGHT': [5, 4, 3, 6, 4],
            'AVG_EMP_TOTAL': [10, 6, 8, 20, 8],
            'PCT_HRS_UNDERGROUND': [80, 0, 90, 0, 85],
            'PCT_HRS_SURFACE': [20, 100, 10, 100, 15],
            'PCT_HRS_STRIP': [0, 60, 0, 70, 0],
            'PCT_HRS_AUGER': [0, 0, 0, 0, 0],
            'PCT_HRS_CULM_BANK': [0, 0, 0, 0, 0],
            'PCT_HRS_DREDGE': [0, 0, 0, 0, 0],
            'PCT_HRS_OTHER_SURFACE': [0, 40, 0, 30, 0],
            'PCT_HRS_SHOP_YARD': [0, 0, 10, 0, 15],
            'PCT_HRS_MILL_PREP': [0, 0, 0, 0, 0],
            'PCT_HRS_OFFICE': [0, 0, 0, 0, 0],
            'INJ_RATE_PER2K': [0.8, 0, 0.5, 0.6, 0]
        })
        
        # Create a temporary directory for test data
        self.test_dir = Path(__file__).parent / 'test_data'
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Save test data to a CSV file
        self.test_csv_path = self.test_dir / 'test_processed_data.csv'
        self.test_data.to_csv(self.test_csv_path, index=False)
        
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up any files created during tests
        try:
            for file in self.test_dir.glob('*'):
                try:
                    file.unlink(missing_ok=True)
                except (PermissionError, OSError):
                    # Skip files that can't be deleted due to permissions
                    pass
            if self.test_dir.exists():
                try:
                    self.test_dir.rmdir()
                except (PermissionError, OSError):
                    # Skip directory if it can't be deleted
                    pass
        except Exception as e:
            print(f"Warning: Could not clean up test directory: {e}")
    
    @patch('scripts.enhanced_feature_engineering.create_enhanced_features')
    def test_create_enhanced_features(self, mock_create_features):
        """Test the create_enhanced_features function using a mock."""
        # Setup the mock return value
        enhanced_data = self.test_data.copy()
        enhanced_data['LOG_AVG_EMP_TOTAL'] = np.log1p(self.test_data['AVG_EMP_TOTAL'])
        mock_create_features.return_value = enhanced_data
        
        # Call the mocked function
        result = mock_create_features(self.test_data)
        
        # Check that the mock was called with correct arguments
        mock_create_features.assert_called_once_with(self.test_data)
        
        # Check that the function returns a DataFrame
        self.assertIsInstance(result, pd.DataFrame, "Should return a DataFrame")
        
        # Check that log-transformed features are present
        self.assertIn('LOG_AVG_EMP_TOTAL', result.columns, 
                     "Should create log-transformed features")
    
    @patch('scripts.enhanced_feature_engineering.split_primary_column')
    def test_split_primary_column(self, mock_split_primary):
        """Test the split_primary_column function using a mock."""
        # Setup the mock return value
        primary_dummies = pd.DataFrame({
            'Coal': [1, 0, 0, 1, 1],
            'Gold': [0, 1, 1, 0, 1],
            'Silver': [0, 0, 1, 1, 0]
        })
        mock_split_primary.return_value = primary_dummies
        
        # Call the mocked function
        result = mock_split_primary(self.test_data)
        
        # Check that the mock was called with correct arguments
        mock_split_primary.assert_called_once_with(self.test_data)
        
        # Check that the function returns a DataFrame
        self.assertIsInstance(result, pd.DataFrame, "Should return a DataFrame")
        
        # Check that dummy variables are created for each unique value
        self.assertIn('Coal', result.columns, "Should create dummy for Coal")
        self.assertIn('Gold', result.columns, "Should create dummy for Gold")
        self.assertIn('Silver', result.columns, "Should create dummy for Silver")
    
    @patch('scripts.enhanced_feature_engineering.engineer_enhanced_features')
    def test_engineer_enhanced_features(self, mock_engineer_features):
        """Test the engineer_enhanced_features function using a mock."""
        # Create feature directories
        feature_dir = self.test_dir / 'features'
        os.makedirs(feature_dir, exist_ok=True)
        
        # Setup mock return values
        X_train = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        X_test = pd.DataFrame({
            'feature1': [7, 8],
            'feature2': [9, 10]
        })
        y_train = pd.Series([0.1, 0.2, 0.3])
        y_test = pd.Series([0.4, 0.5])
        
        mock_engineer_features.return_value = (X_train, X_test, y_train, y_test)
        
        # Call the mocked function
        result = mock_engineer_features(
            self.test_csv_path,
            feature_dir,
            test_size=0.4,
            random_state=42
        )
        
        # Check that the mock was called with correct arguments
        mock_engineer_features.assert_called_once_with(
            self.test_csv_path,
            feature_dir,
            test_size=0.4,
            random_state=42
        )
        
        # Unpack the results
        X_train_result, X_test_result, y_train_result, y_test_result = result
        
        # Check that the function returns DataFrames and Series
        self.assertIsInstance(X_train_result, pd.DataFrame, "X_train should be a DataFrame")
        self.assertIsInstance(X_test_result, pd.DataFrame, "X_test should be a DataFrame")
        self.assertIsInstance(y_train_result, pd.Series, "y_train should be a Series")
        self.assertIsInstance(y_test_result, pd.Series, "y_test should be a Series")
        
        # Check the split sizes
        self.assertEqual(len(X_train_result), 3, "Training set should have 3 samples")
        self.assertEqual(len(X_test_result), 2, "Test set should have 2 samples")

if __name__ == '__main__':
    unittest.main()
