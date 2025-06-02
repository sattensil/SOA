"""
Integration tests for the FastAPI application.
"""
import os
import sys
import unittest
import tempfile
import json
from fastapi.testclient import TestClient

# Add the parent directory to the path so we can import the scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the FastAPI app
from api.main import app
from scripts.data_loader import perform_basic_cleaning as clean_data
from scripts.prediction_utility import preprocess_data
from scripts.model_training import train_xgboost_model

# Configure test client
client = TestClient(app)

class TestAPI(unittest.TestCase):
    """Test the FastAPI application."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create temporary directories for test data
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, 'data')
        self.models_dir = os.path.join(self.temp_dir, 'models')
        self.features_dir = os.path.join(self.data_dir, 'features')
        
        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)
        
        # Set environment variables
        os.environ['DATA_DIR'] = self.data_dir
        os.environ['MODELS_DIR'] = self.models_dir
        os.environ['FEATURES_DIR'] = self.features_dir
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directories
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_root_endpoint(self):
        """Test the root endpoint."""
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Mine Safety Injury Rate Prediction API"})
    
    def test_health_check(self):
        """Test the health check endpoint."""
        # This might fail if the model is not loaded
        response = client.get("/health")
        # We expect a 503 if the model is not loaded, which is OK for tests
        self.assertIn(response.status_code, [200, 503])
    
    def test_model_info(self):
        """Test the model info endpoint."""
        response = client.get("/model/info")
        # We expect a 503 if the model is not loaded, which is OK for tests
        self.assertIn(response.status_code, [200, 503])
        
        if response.status_code == 200:
            data = response.json()
            self.assertIn("model_type", data)
            self.assertIn("model_version", data)
    
    def test_prediction_endpoint(self):
        """Test the prediction endpoint with sample data."""
        # Sample mine data
        mine_data = {
            "MINE_ID": "1234567",
            "YEAR": 2016,
            "PRIMARY": "Coal",
            "CURRENT_MINE_TYPE": "Surface",
            "CURRENT_STATUS": "Active",
            "CURRENT_MINE_SUBTYPE": "Strip",
            "FIPS_CNTY": "01001",
            "AVG_EMPLOYEE_CNT": 45.0,
            "HOURS_WORKED": 90000.0,
            "COAL_METAL_IND": "C",
            "INJURIES_COUNT": None,
            "INJURY_RATE": None
        }
        
        # Make prediction request
        response = client.post("/predict", json=mine_data)
        
        # Check response - might be 503 if model not loaded
        if response.status_code == 200:
            data = response.json()
            self.assertIn("mine_id", data)
            self.assertIn("predicted_injury_rate", data)
            self.assertEqual(data["mine_id"], "1234567")
            self.assertIsInstance(data["predicted_injury_rate"], float)
        else:
            # If model not loaded, should get a 503
            self.assertEqual(response.status_code, 503)
    
    def test_batch_prediction_endpoint(self):
        """Test the batch prediction endpoint with sample data."""
        # Sample batch data
        batch_data = {
            "mines": [
                {
                    "MINE_ID": "1234567",
                    "YEAR": 2016,
                    "PRIMARY": "Coal",
                    "CURRENT_MINE_TYPE": "Surface",
                    "CURRENT_STATUS": "Active",
                    "CURRENT_MINE_SUBTYPE": "Strip",
                    "FIPS_CNTY": "01001",
                    "AVG_EMPLOYEE_CNT": 45.0,
                    "HOURS_WORKED": 90000.0,
                    "COAL_METAL_IND": "C",
                    "INJURIES_COUNT": None,
                    "INJURY_RATE": None
                },
                {
                    "MINE_ID": "7654321",
                    "YEAR": 2016,
                    "PRIMARY": "Metal",
                    "CURRENT_MINE_TYPE": "Underground",
                    "CURRENT_STATUS": "Active",
                    "CURRENT_MINE_SUBTYPE": "Drift",
                    "FIPS_CNTY": "01003",
                    "AVG_EMPLOYEE_CNT": 120.0,
                    "HOURS_WORKED": 240000.0,
                    "COAL_METAL_IND": "M",
                    "INJURIES_COUNT": None,
                    "INJURY_RATE": None
                }
            ]
        }
        
        # Make batch prediction request
        response = client.post("/predict/batch", json=batch_data)
        
        # Check response - might be 503 if model not loaded
        if response.status_code == 200:
            data = response.json()
            self.assertIn("predictions", data)
            self.assertIn("model_version", data)
            self.assertIn("average_predicted_rate", data)
            self.assertEqual(len(data["predictions"]), 2)
            self.assertIsInstance(data["average_predicted_rate"], float)
        else:
            # If model not loaded, should get a 503
            self.assertEqual(response.status_code, 503)

if __name__ == "__main__":
    unittest.main()
