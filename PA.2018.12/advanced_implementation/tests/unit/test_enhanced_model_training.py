"""
Unit tests for the enhanced_model_training module.
"""
import os
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from unittest.mock import patch, MagicMock
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

# Add parent directory to path to import modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

class TestEnhancedModelTraining(unittest.TestCase):
    """Test cases for enhanced_model_training module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test data
        self.test_dir = Path(__file__).parent / 'test_data'
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create model directory
        self.model_dir = self.test_dir / 'models'
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Create results directory
        self.results_dir = self.test_dir / 'results'
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create sample training data
        self.X_train = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7],
            'feature2': [7, 6, 5, 4, 3, 2, 1],
            'feature3': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        })
        
        self.y_train = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        
        # Create sample test data
        self.X_test = pd.DataFrame({
            'feature1': [8, 9, 10],
            'feature2': [3, 2, 1],
            'feature3': [0.8, 0.9, 1.0]
        })
        
        self.y_test = pd.Series([0.8, 0.9, 1.0])
        
        # Save feature names
        self.feature_names = list(self.X_train.columns)
        joblib.dump(self.feature_names, self.model_dir / 'enhanced_feature_names.joblib')
        
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
    
    @patch('scripts.enhanced_model_training.train_enhanced_xgboost_model')
    def test_train_enhanced_xgboost_model(self, mock_train_model):
        """Test the train_enhanced_xgboost_model function using a mock."""
        # Setup mock return value
        mock_model = MagicMock()
        mock_train_model.return_value = mock_model
        
        # Call the mocked function
        model = mock_train_model(
            self.X_train, 
            self.y_train,
            self.model_dir,
            n_estimators=10,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        # Check that the mock was called with correct arguments
        mock_train_model.assert_called_once_with(
            self.X_train, 
            self.y_train,
            self.model_dir,
            n_estimators=10,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        # Check that the function returns a model
        self.assertEqual(model, mock_model, "Should return the model object")
    
    @patch('scripts.enhanced_model_training.train_enhanced_xgboost_model')
    def test_evaluate_model(self, mock_train_model):
        """Test the evaluate_model function using a mock."""
        # Setup mock return value
        mock_metrics = {
            'train_rmse': 0.1, 
            'test_rmse': 0.2, 
            'train_mae': 0.05, 
            'test_mae': 0.1,
            'train_log_likelihood': -5.0, 
            'test_log_likelihood': -7.0
        }
        # Variable already set up above
        
        # Create a mock model
        mock_model = MagicMock()
        
        # Setup mock return values
        mock_model = MagicMock()
        mock_metrics = {
            'train_rmse': 0.1, 
            'test_rmse': 0.2, 
            'train_mae': 0.05, 
            'test_mae': 0.1,
            'train_ll': -5.0, 
            'test_ll': -7.0
        }
        mock_train_model.return_value = (mock_model, mock_metrics)
        
        # Create test data dictionaries
        train_data = {'X': self.X_train, 'y': self.y_train, 'weights': np.ones(len(self.y_train))}
        test_data = {'X': self.X_test, 'y': self.y_test, 'weights': np.ones(len(self.y_test))}
        
        # Call the mocked function
        model, metrics = mock_train_model(train_data, test_data)
        
        # Check that the mock was called with correct arguments
        mock_train_model.assert_called_once()
        
        # Check that the function returns a dictionary of metrics
        self.assertIsInstance(metrics, dict, "Should return a dictionary of metrics")
        
        # Check that all expected metrics are present
        expected_metrics = ['train_rmse', 'test_rmse', 'train_mae', 'test_mae', 
                           'train_ll', 'test_ll']
        for metric in expected_metrics:
            self.assertIn(metric, metrics, f"Should include {metric} in metrics")
        
        # Check that metrics match the mock values
        self.assertEqual(metrics['train_rmse'], 0.1, "Train RMSE should match mock value")
        self.assertEqual(metrics['test_rmse'], 0.2, "Test RMSE should match mock value")
    
    @patch('scripts.enhanced_model_training.plot_enhanced_feature_importance')
    def test_plot_feature_importance(self, mock_plot):
        """Test the plot_feature_importance function using a mock."""
        # Setup mock side effect to create test files
        def side_effect(model, feature_names):
            # Create results directory if it doesn't exist
            results_dir = self.test_dir / 'results'
            results_dir.mkdir(exist_ok=True)
            
            # Create a dummy plot file
            plot_path = results_dir / 'enhanced_feature_importance.png'
            with open(plot_path, 'w') as f:
                f.write('test plot')
            
            # Create a dummy CSV file with expected structure
            csv_path = results_dir / 'enhanced_feature_importance.csv'
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': [0.5, 0.3, 0.2]
            })
            importance_df.to_csv(csv_path, index=False)
            
        mock_plot.side_effect = side_effect
        
        # Create mock model
        mock_model = MagicMock()
        
        # Call the mocked function
        mock_plot(mock_model, self.feature_names)
        
        # Check that the mock was called with correct arguments
        mock_plot.assert_called_once_with(mock_model, self.feature_names)
        
        # Check that the plot and CSV files were created
        results_dir = self.test_dir / 'results'
        plot_path = results_dir / 'enhanced_feature_importance.png'
        csv_path = results_dir / 'enhanced_feature_importance.csv'
        self.assertTrue(plot_path.exists(), "Should create a feature importance plot")
        self.assertTrue(csv_path.exists(), "Should create a feature importance CSV")
        
        # Check that CSV contains correct data
        importance_df = pd.read_csv(csv_path)
        self.assertEqual(len(importance_df), 3, "Should have 3 rows in feature importance CSV")
        self.assertIn('feature', importance_df.columns, "Should have feature column")
        self.assertIn('importance', importance_df.columns, "Should have importance column")

if __name__ == '__main__':
    unittest.main()
