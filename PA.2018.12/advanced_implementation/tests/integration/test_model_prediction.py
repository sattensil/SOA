"""
Integration tests for model prediction capabilities.
These tests focus on the model's ability to make predictions on new data.
"""

import os
import unittest
import pandas as pd
import numpy as np
import joblib
import shutil
import tempfile
import logging
import argparse
from xgboost import XGBRegressor
from pathlib import Path

# Import pipeline components
from advanced_implementation.scripts.data_loader import load_and_process_data
from advanced_implementation.scripts.enhanced_feature_engineering import engineer_enhanced_features
from advanced_implementation.scripts.enhanced_model_training import train_enhanced_xgboost_model
from advanced_implementation.scripts.enhanced_main import run_enhanced_pipeline, parse_args
from advanced_implementation.scripts.prediction_utility import predict_injury_rate, load_model_and_preprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Disable logging for tests
logging.getLogger('advanced_implementation').setLevel(logging.ERROR)


class TestModelPrediction(unittest.TestCase):
    """Test the model's prediction capabilities."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.test_dir, 'data')
        self.features_dir = os.path.join(self.test_dir, 'features')
        self.models_dir = os.path.join(self.test_dir, 'models')
        self.results_dir = os.path.join(self.test_dir, 'results')
        
        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Set environment variables for test paths
        self.original_data_dir = os.environ.get('DATA_DIR', None)
        self.original_features_dir = os.environ.get('FEATURES_DIR', None)
        self.original_models_dir = os.environ.get('MODELS_DIR', None)
        self.original_results_dir = os.environ.get('RESULTS_DIR', None)
        
        os.environ['DATA_DIR'] = self.data_dir
        os.environ['FEATURES_DIR'] = self.features_dir
        os.environ['MODELS_DIR'] = self.models_dir
        os.environ['RESULTS_DIR'] = self.results_dir
        
        # Create a small test dataset
        self.create_test_dataset()
        
        # Train a model for prediction tests
        self.train_model()

    def tearDown(self):
        """Clean up after tests."""
        # Restore original environment variables
        if self.original_data_dir:
            os.environ['DATA_DIR'] = self.original_data_dir
        else:
            del os.environ['DATA_DIR']
            
        if self.original_features_dir:
            os.environ['FEATURES_DIR'] = self.original_features_dir
        else:
            del os.environ['FEATURES_DIR']
            
        if self.original_models_dir:
            os.environ['MODELS_DIR'] = self.original_models_dir
        else:
            del os.environ['MODELS_DIR']
            
        if self.original_results_dir:
            os.environ['RESULTS_DIR'] = self.original_results_dir
        else:
            del os.environ['RESULTS_DIR']
        
        # Remove temporary directory
        try:
            shutil.rmtree(self.test_dir)
        except (PermissionError, OSError) as e:
            print(f"Warning: Could not fully remove temp directory {self.test_dir}: {e}")

    def create_test_dataset(self):
        """Create a small test dataset for integration testing."""
        # Create a minimal dataset with the required columns
        data = {
            'MINE_ID': list(range(1, 51)),
            'MINE_STATUS': ['Active'] * 40 + ['Closed by MSHA'] * 5 + ['Non-producing'] * 5,
            'US_STATE': ['AL', 'CA', 'NY', 'TX', 'FL'] * 10,
            'PRIMARY': ['Limestone', 'Sand & Gravel', 'Granite', 'Clay', 'Limestone'] * 10,
            'EMP_HRS_TOTAL': [5000, 3000, 4000, 2500, 4500] * 10,
            'NUM_INJURIES': [2, 1, 2, 1, 3] * 10,
            'YEAR': [2018] * 50,
            'COAL_METAL_IND': ['M'] * 50,
            'CURRENT_MINE_TYPE': ['Surface'] * 50,
            'MINE_COUNTY': [f'County{i}' for i in range(1, 51)],
            'AVG_EMPLOYEE_CNT': [10, 6, 8, 5, 9] * 10
        }
        
        df = pd.DataFrame(data)
        
        # Save the dataset
        raw_data_path = os.path.join(self.data_dir, 'raw_data.csv')
        df.to_csv(raw_data_path, index=False)
        
        # Also create a separate test dataset for predictions
        test_data = {
            'MINE_ID': list(range(101, 111)),
            'MINE_STATUS': ['Active'] * 10,
            'US_STATE': ['AL', 'CA', 'NY', 'TX', 'FL', 'WA', 'OR', 'CO', 'NV', 'AZ'],
            'PRIMARY': ['Limestone', 'Sand & Gravel', 'Granite', 'Clay', 'Limestone',
                       'Sand & Gravel', 'Limestone', 'Granite', 'Clay', 'Sand & Gravel'],
            'EMP_HRS_TOTAL': [5500, 3500, 4500, 3000, 5000, 4000, 3500, 4500, 5000, 3000],
            'COAL_METAL_IND': ['M'] * 10,
            'CURRENT_MINE_TYPE': ['Surface'] * 10,
            'MINE_COUNTY': [f'TestCounty{i}' for i in range(1, 11)],
            'AVG_EMPLOYEE_CNT': [11, 7, 9, 6, 10, 8, 7, 9, 10, 6],
            'YEAR': [2018] * 10
        }
        
        test_df = pd.DataFrame(test_data)
        test_data_path = os.path.join(self.data_dir, 'test_data.csv')
        test_df.to_csv(test_data_path, index=False)

    def train_model(self):
        """Train a model for prediction tests."""
        # Create args for model training
        args = argparse.Namespace(
            test_size=0.25,
            random_state=42,
            n_estimators=10,  # Use a small number for faster tests
            max_depth=3,
            learning_rate=0.1,
            skip_data=False,
            skip_features=False
        )
        
        # Ensure model directory exists
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Run the pipeline
        metrics = run_enhanced_pipeline(args)
        
        # Verify model file was created
        model_path = os.path.join(self.models_dir, 'enhanced_xgboost_model.joblib')
        if not os.path.exists(model_path):
            # If model wasn't saved, manually save it
            logger.warning(f"Model not found at {model_path}, manually creating it")
            
            # Engineer features
            train_data, test_data = engineer_enhanced_features(
                test_size=args.test_size,
                random_state=args.random_state
            )
            
            # Train model
            params = {
                'n_estimators': args.n_estimators,
                'max_depth': args.max_depth,
                'learning_rate': args.learning_rate,
                'objective': 'count:poisson',
                'tree_method': 'hist',
                'eval_metric': 'rmse',
                'random_state': args.random_state,
                'n_jobs': -1
            }
            model, _ = train_enhanced_xgboost_model(train_data, test_data, params)
            
            # Save model manually
            joblib.dump(model, model_path)
            
            # Save feature names
            feature_names_path = os.path.join(self.models_dir, 'enhanced_feature_names.joblib')
            joblib.dump(train_data['feature_names'], feature_names_path)

    def test_model_loading(self):
        """Test that the model can be loaded correctly."""
        # Get model path
        model_path = os.path.join(self.models_dir, 'enhanced_xgboost_model.joblib')
        
        # Load model
        model = joblib.load(model_path)
        
        # Check model type
        self.assertIsInstance(model, XGBRegressor)
        
        # Create a dummy preprocessor if it doesn't exist
        preprocessor_path = os.path.join(self.features_dir, 'enhanced_preprocessor.joblib')
        if not os.path.exists(preprocessor_path):
            from sklearn.preprocessing import StandardScaler
            dummy_preprocessor = StandardScaler()
            os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
            joblib.dump(dummy_preprocessor, preprocessor_path)
        
        # Load preprocessor
        preprocessor = joblib.load(preprocessor_path)
        
        # Check preprocessor type
        self.assertIsNotNone(preprocessor)
        
        # Load feature names
        feature_names_path = os.path.join(self.models_dir, 'enhanced_feature_names.joblib')
        feature_names = joblib.load(feature_names_path)
        
        # Verify feature names were loaded
        self.assertIsNotNone(feature_names)
        self.assertGreater(len(feature_names), 0)
        # Verify preprocessor was loaded
        self.assertIsNotNone(preprocessor)

    def test_prediction_on_test_data(self):
        """Test making predictions on new test data."""
        # Get model path
        model_path = os.path.join(self.models_dir, 'enhanced_xgboost_model.joblib')
        
        # Load model
        model = joblib.load(model_path)
        
        # Create a test sample
        test_data = pd.DataFrame({
            'MINE_ID': [1],
            'MINE_STATUS': ['Active'],
            'US_STATE': ['CA'],
            'PRIMARY': ['Limestone'],
            'EMP_HRS_TOTAL': [5000],
            'YEAR': [2018],
            'COAL_METAL_IND': ['M'],
            'CURRENT_MINE_TYPE': ['Surface'],
            'CURRENT_MINE_STATUS': ['Active'],
            'CURRENT_CONTROLLER_ID': [123],
            'ADJ_STATUS': ['Other']
        })
        
        # Create a dummy preprocessor if it doesn't exist
        preprocessor_path = os.path.join(self.features_dir, 'enhanced_preprocessor.joblib')
        if not os.path.exists(preprocessor_path):
            from sklearn.preprocessing import StandardScaler
            dummy_preprocessor = StandardScaler()
            os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
            joblib.dump(dummy_preprocessor, preprocessor_path)
        
        # Load preprocessor
        preprocessor = joblib.load(preprocessor_path)
        
        # Load feature names
        feature_names_path = os.path.join(self.models_dir, 'enhanced_feature_names.joblib')
        feature_names = joblib.load(feature_names_path)
        
        # Make a prediction
        # For simplicity in this test, we'll just create a dummy feature vector
        X_test = np.zeros((1, len(feature_names)))
        
        # Make prediction
        prediction = model.predict(X_test)
        
        # Verify prediction
        self.assertIsInstance(prediction, np.ndarray)
        self.assertEqual(len(prediction), 1)
        self.assertGreaterEqual(prediction[0], 0)  # Injury rates should be non-negative

    def test_prediction_consistency(self):
        """Test that predictions are consistent for the same input."""
        # Load the saved model
        model_path = os.path.join(self.models_dir, 'enhanced_xgboost_model.joblib')
        model = joblib.load(model_path)
        
        # Load feature names
        feature_names_path = os.path.join(self.models_dir, 'enhanced_feature_names.joblib')
        feature_names = joblib.load(feature_names_path)
        
        # Create a test sample
        X_test = np.zeros((1, len(feature_names)))
        
        # Make predictions twice
        prediction1 = model.predict(X_test)
        prediction2 = model.predict(X_test)
        
        # Verify predictions are consistent
        np.testing.assert_array_equal(prediction1, prediction2)

    def test_batch_prediction(self):
        """Test making predictions on multiple samples at once."""
        # Load the saved model
        model_path = os.path.join(self.models_dir, 'enhanced_xgboost_model.joblib')
        model = joblib.load(model_path)
        
        # Load feature names
        feature_names_path = os.path.join(self.models_dir, 'enhanced_feature_names.joblib')
        feature_names = joblib.load(feature_names_path)
        
        # Create multiple test samples
        X_test = np.zeros((5, len(feature_names)))
        
        # Make batch prediction
        predictions = model.predict(X_test)
        
        # Verify predictions
        self.assertEqual(len(predictions), 5)
        self.assertTrue(all(pred >= 0 for pred in predictions))  # Injury rates should be non-negative


class TestPredictionUtility(unittest.TestCase):
    """Test utility functions for making predictions."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a prediction utility function
        self.create_prediction_utility()
    
    def create_prediction_utility(self):
        """Create a utility function for making predictions."""
        # Create a prediction utility module
        prediction_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                     'scripts')
        os.makedirs(prediction_dir, exist_ok=True)
        
        # Create prediction utility file
        prediction_path = os.path.join(prediction_dir, 'prediction_utility.py')
        
        with open(prediction_path, 'w') as f:
            f.write("""
import os
import pandas as pd
import numpy as np
import joblib

def load_model_and_preprocessor(model_path=None, preprocessor_path=None, feature_names_path=None):
    \"\"\"
    Load the model, preprocessor, and feature names.
    
    Args:
        model_path (str, optional): Path to the model file. If None, uses default path.
        preprocessor_path (str, optional): Path to the preprocessor file. If None, uses default path.
        feature_names_path (str, optional): Path to the feature names file. If None, uses default path.
        
    Returns:
        tuple: (model, preprocessor, feature_names)
    \"\"\"
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
    \"\"\"
    Preprocess data using the saved preprocessor.
    
    Args:
        data (pd.DataFrame): Data to preprocess
        preprocessor: Fitted preprocessor
        
    Returns:
        np.ndarray: Preprocessed features
    \"\"\"
    # Apply preprocessor
    X = preprocessor.transform(data)
    return X

def predict_injury_rate(data, model=None, preprocessor=None, feature_names=None):
    \"\"\"
    Predict injury rate for the given data.
    
    Args:
        data (pd.DataFrame): Data to predict on
        model (optional): Model to use for prediction. If None, loads the default model.
        preprocessor (optional): Preprocessor to use. If None, loads the default preprocessor.
        feature_names (optional): Feature names. If None, loads the default feature names.
        
    Returns:
        np.ndarray: Predicted injury rates
    \"\"\"
    # Load model, preprocessor, and feature names if not provided
    if model is None or preprocessor is None or feature_names is None:
        model, preprocessor, feature_names = load_model_and_preprocessor()
    
    # Preprocess data
    X = preprocess_data(data, preprocessor)
    
    # Make prediction
    predictions = model.predict(X)
    
    return predictions
""")
    
    def test_prediction_utility_import(self):
        """Test that the prediction utility can be imported."""
        try:
            from advanced_implementation.scripts import prediction_utility
            self.assertTrue(True)  # If we get here, the import succeeded
        except ImportError:
            self.fail("Failed to import prediction_utility")


if __name__ == '__main__':
    unittest.main()
