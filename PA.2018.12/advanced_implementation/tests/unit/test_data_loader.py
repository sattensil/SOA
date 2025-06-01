"""
Unit tests for the data_loader module.
"""
import os
import unittest
from unittest import mock
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path to import modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.data_loader import load_and_process_data, perform_basic_cleaning

class TestDataLoader(unittest.TestCase):
    """Test cases for data_loader module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a small test dataframe
        self.test_data = pd.DataFrame({
            'MINE_ID': [1, 2, 3, 4, 5],
            'US_STATE': ['PA', 'WV', 'KY', 'IL', None],
            'PRIMARY': ['Coal', 'Metal', 'Nonmetal', 'Coal', 'Coal'],
            'NUM_INJURIES': [10, 5, 0, 2, 8],
            'EMP_HRS_TOTAL': [20000, 15000, 10000, 1000, 18000],
            'MINE_STATUS': ['Active', 'Active', 'Active', 'Closed by MSHA', 'Active'],
            'SEAM_HEIGHT': [5, 4, 3, 6, 4]
        })
        
        # Create a temporary directory for test data
        self.test_dir = Path(__file__).parent / 'test_data'
        os.makedirs(self.test_dir, exist_ok=True)
        
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up any files created during tests
        for file in self.test_dir.glob('*'):
            file.unlink()
        if self.test_dir.exists():
            self.test_dir.rmdir()
    
    def test_basic_data_cleaning(self):
        """Test the basic data cleaning function."""
        # Test data cleaning with our test dataset
        cleaned_data = perform_basic_cleaning(self.test_data)
        
        # Check that rows with missing values are removed and other cleaning steps are applied
        self.assertEqual(len(cleaned_data), 3, "Should have 3 rows after all cleaning steps")
        
        # Check that closed mines are removed (MINE_STATUS is converted to ADJ_STATUS)
        self.assertNotIn('MINE_STATUS', cleaned_data.columns, "MINE_STATUS should be replaced with ADJ_STATUS")
        self.assertIn('ADJ_STATUS', cleaned_data.columns, "ADJ_STATUS should be present in cleaned data")
        self.assertFalse(any(cleaned_data['ADJ_STATUS'] == 'Closed'), "Should remove closed mines")
        
        # Check that mines with low employee hours are removed
        self.assertTrue(all(cleaned_data['EMP_HRS_TOTAL'] >= 2000), 
                       "Should remove mines with less than 2000 employee hours")
        
        # Check that injury rate is calculated correctly
        expected_rate = self.test_data.iloc[0]['NUM_INJURIES'] / (self.test_data.iloc[0]['EMP_HRS_TOTAL'] / 2000)
        self.assertAlmostEqual(cleaned_data.iloc[0]['INJ_RATE_PER2K'], expected_rate, 
                              places=5, msg="Injury rate calculation is incorrect")
    
    @mock.patch('scripts.data_loader.load_raw_data')
    def test_load_and_process_data(self, mock_load_raw_data):
        """Test the load_and_process_data function with a mock."""
        # Setup the mock to return our test data
        mock_load_raw_data.return_value = self.test_data
        
        # Test loading and processing the data
        processed_data = load_and_process_data(save=False)
        
        # Check that the function returns a DataFrame
        self.assertIsInstance(processed_data, pd.DataFrame, "Should return a DataFrame")
        
        # Check that the processed data has the expected shape
        self.assertEqual(processed_data.shape[0], 3, "Should have 3 rows after cleaning")
        
        # Verify the mock was called
        mock_load_raw_data.assert_called_once()

if __name__ == '__main__':
    unittest.main()
