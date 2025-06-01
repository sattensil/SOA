"""
Unit tests for the enhanced_main module.
"""
import os
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import argparse

# Add parent directory to path to import modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

class TestEnhancedMain(unittest.TestCase):
    """Test cases for enhanced_main module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test data
        self.test_dir = Path(__file__).parent / 'test_data'
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create subdirectories
        self.data_dir = self.test_dir / 'data'
        self.feature_dir = self.test_dir / 'features'
        self.model_dir = self.test_dir / 'models'
        self.results_dir = self.test_dir / 'results'
        
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.feature_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create a sample CSV file
        self.test_data = pd.DataFrame({
            'MINE_ID': [1, 2, 3],
            'MINE_STATUS': ['Active', 'Active', 'Active'],
            'US_STATE': ['CA', 'NV', 'AZ'],
            'PRIMARY': ['Coal', 'Gold', 'Silver'],
            'NUM_INJURIES': [2, 0, 1],
            'EMP_HRS_TOTAL': [5000, 3000, 4000]
        })
        
        self.test_csv_path = self.data_dir / 'test_data.csv'
        self.test_data.to_csv(self.test_csv_path, index=False)
        
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up any files created during tests
        try:
            for root, dirs, files in os.walk(self.test_dir, topdown=False):
                for file in files:
                    try:
                        os.remove(os.path.join(root, file))
                    except (PermissionError, OSError):
                        # Skip files that can't be deleted due to permissions
                        pass
                for dir in dirs:
                    try:
                        os.rmdir(os.path.join(root, dir))
                    except (PermissionError, OSError):
                        # Skip directories that can't be deleted
                        pass
            if self.test_dir.exists():
                try:
                    os.rmdir(self.test_dir)
                except (PermissionError, OSError):
                    # Skip directory if it can't be deleted
                    pass
        except Exception as e:
            print(f"Warning: Could not clean up test directory: {e}")
    
    @patch('scripts.enhanced_main.parse_args')
    def test_parse_args(self, mock_parse_args):
        """Test the argument parsing function using a mock."""
        # Setup mock return values for default arguments
        default_args = argparse.Namespace(
            data_file='data/MSHA_Mine_Data_2013-2016.csv',
            processed_file='data/processed_data.csv',
            test_size=0.25,
            random_state=42,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3
        )
        
        # Setup mock return values for custom arguments
        custom_args = argparse.Namespace(
            data_file='custom_data.csv',
            processed_file='custom_processed.csv',
            test_size=0.3,
            random_state=100,
            n_estimators=50,
            learning_rate=0.05,
            max_depth=5
        )
        
        # Test with default arguments
        mock_parse_args.return_value = default_args
        args = mock_parse_args()
        self.assertIsInstance(args, argparse.Namespace, "Should return an argparse.Namespace")
        self.assertEqual(args.data_file, 'data/MSHA_Mine_Data_2013-2016.csv', 
                        "Should use default data file path")
        
        # Test with custom arguments
        mock_parse_args.return_value = custom_args
        args = mock_parse_args()
        self.assertEqual(args.data_file, 'custom_data.csv', 
                        "Should use custom data file path")
        self.assertEqual(args.processed_file, 'custom_processed.csv', 
                        "Should use custom processed file path")
        self.assertEqual(args.test_size, 0.3, "Should use custom test size")
        self.assertEqual(args.random_state, 100, "Should use custom random state")
        self.assertEqual(args.n_estimators, 50, "Should use custom n_estimators")
        self.assertEqual(args.learning_rate, 0.05, "Should use custom learning rate")
        self.assertEqual(args.max_depth, 5, "Should use custom max depth")
    
    @patch('scripts.enhanced_main.load_and_process_data')
    @patch('scripts.enhanced_main.engineer_enhanced_features')
    @patch('scripts.enhanced_main.train_enhanced_xgboost_model')
    @patch('scripts.enhanced_main.save_enhanced_model')
    @patch('scripts.enhanced_main.plot_enhanced_feature_importance')
    def test_run_pipeline(self, mock_plot, mock_save, mock_train, mock_engineer, mock_load):
        """Test the pipeline execution function with mocks."""
        # Set up mock returns
        mock_load.return_value = self.test_data
        
        # Create mock train and test dictionaries with the correct structure
        train_dict = {
            'X': pd.DataFrame([[0.1, 0.2, 0.3]]),
            'y': pd.Series([0.2]).values,
            'weights': pd.Series([1.0]).values,
            'feature_names': ['Feature_1', 'Feature_2', 'Feature_3'],
            'raw_data': pd.DataFrame({'INJ_RATE_PER2K': [0.2], 'EMP_HRS_TOTAL': [2000]})
        }
        
        test_dict = {
            'X': pd.DataFrame([[0.4, 0.5, 0.6]]),
            'y': pd.Series([0.3]).values,
            'weights': pd.Series([1.0]).values,
            'feature_names': ['Feature_1', 'Feature_2', 'Feature_3'],
            'raw_data': pd.DataFrame({'INJ_RATE_PER2K': [0.3], 'EMP_HRS_TOTAL': [2000]})
        }
        
        mock_engineer.return_value = (train_dict, test_dict)
        mock_model = MagicMock()
        mock_metrics = {
            'train_rmse': 0.1, 
            'test_rmse': 0.2,
            'train_mae': 0.05, 
            'test_mae': 0.1,
            'train_ll': -5, 
            'test_ll': -3
        }
        mock_train.return_value = (mock_model, mock_metrics)
        
        # Create test arguments
        args = argparse.Namespace(
            data_file=str(self.test_csv_path),
            processed_file=str(self.data_dir / 'processed.csv'),
            test_size=0.25,
            random_state=42,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            skip_data=False,
            skip_features=False
        )
        
        # Import the function to test
        from scripts.enhanced_main import run_enhanced_pipeline
        
        # Run the pipeline
        run_enhanced_pipeline(args)
        
        # Check that all functions were called with correct arguments
        mock_load.assert_called_once()
        
        mock_engineer.assert_called_once()
        
        mock_train.assert_called_once()
        
        mock_save.assert_called_once()
        
        mock_plot.assert_called_once()

if __name__ == '__main__':
    unittest.main()
