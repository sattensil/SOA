#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integration tests for MLflow utilities.
"""

import os
import sys
import unittest
import tempfile
import shutil
import time
import json
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import xgboost as xgb
from xgboost import XGBRegressor
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion

# Add the parent directory to the path so we can import the scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import MLflow utilities
from scripts.mlflow_utils import (
    setup_mlflow,
    log_model_training,
    register_model_version,
    set_active_model_version,
    get_active_model_version,
    load_model_from_registry,
    list_model_versions,
    start_mlflow_server,
    MODEL_NAME,
    MODEL_VERSION_FILE
)


class TestMLflowUtils(unittest.TestCase):
    """Test MLflow utilities for model versioning and registry."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create temporary directories for test data
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, 'data')
        self.models_dir = os.path.join(self.temp_dir, 'models')
        self.features_dir = os.path.join(self.data_dir, 'features')
        self.mlruns_dir = os.path.join(self.temp_dir, 'mlruns')
        
        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)
        os.makedirs(self.mlruns_dir, exist_ok=True)
        
        # Set environment variables
        self.original_data_dir = os.environ.get('DATA_DIR')
        self.original_models_dir = os.environ.get('MODELS_DIR')
        self.original_features_dir = os.environ.get('FEATURES_DIR')
        self.original_mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
        
        os.environ['DATA_DIR'] = self.data_dir
        os.environ['MODELS_DIR'] = self.models_dir
        os.environ['FEATURES_DIR'] = self.features_dir
        os.environ['MLFLOW_TRACKING_URI'] = self.mlruns_dir
        
        # Create a simple XGBoost model for testing
        self.feature_names = ['feature1', 'feature2', 'feature3']
        self.model = XGBRegressor(n_estimators=10)
        X = np.random.rand(100, 3)
        y = np.random.rand(100)
        self.model.fit(X, y)
        
        # Create train and test data dictionaries
        self.train_data = {
            'X': X[:80],
            'y': y[:80],
            'feature_names': self.feature_names
        }
        self.test_data = {
            'X': X[80:],
            'y': y[80:],
            'feature_names': self.feature_names
        }
        
        # Model parameters
        self.params = {
            'n_estimators': 10,
            'learning_rate': 0.1,
            'max_depth': 3
        }
        
        # Metrics
        self.metrics = {
            'train_rmse': 0.1,
            'test_rmse': 0.15,
            'train_mae': 0.05,
            'test_mae': 0.08,
            'train_ll': -0.02,
            'test_ll': -0.03
        }
        
        # Set up MLflow
        mlflow.set_tracking_uri(self.mlruns_dir)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directories
        shutil.rmtree(self.temp_dir)
        
        # Restore original environment variables
        if self.original_data_dir:
            os.environ['DATA_DIR'] = self.original_data_dir
        else:
            os.environ.pop('DATA_DIR', None)
            
        if self.original_models_dir:
            os.environ['MODELS_DIR'] = self.original_models_dir
        else:
            os.environ.pop('MODELS_DIR', None)
            
        if self.original_features_dir:
            os.environ['FEATURES_DIR'] = self.original_features_dir
        else:
            os.environ.pop('FEATURES_DIR', None)
            
        if self.original_mlflow_tracking_uri:
            os.environ['MLFLOW_TRACKING_URI'] = self.original_mlflow_tracking_uri
        else:
            os.environ.pop('MLFLOW_TRACKING_URI', None)
    
    def test_setup_mlflow(self):
        """Test setting up MLflow."""
        # Set tracking URI directly for this test
        mlflow.set_tracking_uri(self.mlruns_dir)
        
        # Call setup_mlflow
        setup_mlflow()
        
        # Verify that the experiment exists
        experiment = mlflow.get_experiment_by_name("mine_safety_injury_rate_prediction")
        self.assertIsNotNone(experiment)
    
    @patch('scripts.mlflow_utils.MODELS_DIR', None)  # Patch to avoid file path issues
    def test_log_model_training(self, mock_models_dir):
        """Test logging model training to MLflow."""
        # Set the patched MODELS_DIR to our test directory
        mock_models_dir.return_value = self.models_dir
        
        # Mock the MLflow tracking functions to avoid actual logging
        with patch('mlflow.log_param') as mock_log_param, \
             patch('mlflow.log_metric') as mock_log_metric, \
             patch('mlflow.log_artifact') as mock_log_artifact, \
             patch('mlflow.xgboost.log_model') as mock_log_model, \
             patch('mlflow.start_run') as mock_start_run, \
             patch('mlflow.get_run') as mock_get_run:
            
            # Mock run context
            mock_run = MagicMock()
            mock_run.info.run_id = 'test_run_id'
            mock_start_run.return_value.__enter__.return_value = mock_run
            mock_get_run.return_value = mock_run
            
            # Log model training
            run_id = log_model_training(
                self.model,
                self.params,
                self.metrics,
                self.feature_names,
                self.train_data,
                self.test_data
            )
            
            # Verify run ID
            self.assertEqual(run_id, 'test_run_id')
            
            # Verify parameters were logged
            for key, value in self.params.items():
                mock_log_param.assert_any_call(key, value)
            
            # Verify metrics were logged
            for key, value in self.metrics.items():
                mock_log_metric.assert_any_call(key, value)
            
            # Verify model was logged
            mock_log_model.assert_called_once()
    
    def test_register_model_version(self):
        """Test registering a model version."""
        # Mock the MLflow client and functions
        with patch('scripts.mlflow_utils.MlflowClient') as mock_client_class, \
             patch('scripts.mlflow_utils.set_active_model_version') as mock_set_active, \
             patch('scripts.mlflow_utils.MODEL_VERSION_FILE', os.path.join(self.models_dir, "active_model_version.txt")):
            
            # Mock client
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            # Mock model version
            mock_version = MagicMock()
            mock_version.version = "1"
            mock_version.description = "Test model version"
            mock_client.create_model_version.return_value = mock_version
            
            # Register the model version
            description = "Test model version"
            model_version = register_model_version("test_run_id", description)
            
            # Verify model version
            self.assertIsNotNone(model_version)
            self.assertEqual(model_version.description, description)
            
            # Verify set_active_model_version was called
            mock_set_active.assert_called_once_with(model_version.version)
    
    def test_active_model_version(self):
        """Test setting and getting active model version."""
        # Patch the MODEL_VERSION_FILE to use our test directory
        with patch('scripts.mlflow_utils.MODEL_VERSION_FILE', os.path.join(self.models_dir, "active_model_version.txt")):
            # Set active model version
            test_version = "test_version"
            set_active_model_version(test_version)
            
            # Verify active version was set
            active_version = get_active_model_version()
            self.assertEqual(active_version, test_version)
            
            # Verify file was created
            self.assertTrue(os.path.exists(os.path.join(self.models_dir, "active_model_version.txt")))
            
            # Read file directly to verify content
            with open(os.path.join(self.models_dir, "active_model_version.txt"), 'r') as f:
                file_content = f.read().strip()
            self.assertEqual(file_content, test_version)
    
    @patch('scripts.mlflow_utils.mlflow.xgboost.load_model')
    @patch('scripts.mlflow_utils.joblib.load')
    def test_load_model_from_registry(self, mock_joblib_load, mock_load_model):
        """Test loading a model from the MLflow registry."""
        # Mock the MLflow client
        with patch('scripts.mlflow_utils.MlflowClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            # Mock model version
            mock_version = MagicMock()
            mock_version.version = "1"
            mock_version.run_id = "test_run_id"
            mock_client.get_latest_versions.return_value = [mock_version]
            mock_client.get_model_version.return_value = mock_version
            
            # Mock run
            mock_run = MagicMock()
            mock_run.info.artifact_uri = "file:///tmp/artifacts"
            mlflow.get_run = MagicMock(return_value=mock_run)
            
            # Mock model and feature names
            mock_model = MagicMock()
            mock_load_model.return_value = mock_model
            mock_feature_names = ['feature1', 'feature2', 'feature3']
            mock_joblib_load.return_value = mock_feature_names
            
            # Test loading with specific version
            model, feature_names = load_model_from_registry("1")
            self.assertEqual(model, mock_model)
            self.assertEqual(feature_names, mock_feature_names)
            mock_load_model.assert_called_with("models:/mine_safety_xgboost/1")
            
            # Test loading with latest version
            model, feature_names = load_model_from_registry("latest")
            self.assertEqual(model, mock_model)
            self.assertEqual(feature_names, mock_feature_names)
            
            # Test loading with active version
            set_active_model_version("1")
            model, feature_names = load_model_from_registry()
            self.assertEqual(model, mock_model)
            self.assertEqual(feature_names, mock_feature_names)
    
    def test_list_model_versions(self):
        """Test listing model versions."""
        # Mock the MLflow client
        with patch('scripts.mlflow_utils.MlflowClient') as mock_client_class, \
             patch('scripts.mlflow_utils.get_active_model_version') as mock_get_active:
            
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            # Mock active version
            mock_get_active.return_value = "1"
            
            # Mock model versions
            mock_version1 = MagicMock()
            mock_version1.version = "1"
            mock_version1.run_id = "run1"
            mock_version1.creation_timestamp = int(time.time() * 1000)
            mock_version1.last_updated_timestamp = int(time.time() * 1000)
            mock_version1.description = "First version"
            mock_version1.current_stage = "Production"
            
            mock_version2 = MagicMock()
            mock_version2.version = "2"
            mock_version2.run_id = "run2"
            mock_version2.creation_timestamp = int(time.time() * 1000)
            mock_version2.last_updated_timestamp = int(time.time() * 1000)
            mock_version2.description = "Second version"
            mock_version2.current_stage = "Staging"
            
            mock_client.get_latest_versions.return_value = [mock_version1, mock_version2]
            
            # Mock runs
            mock_run1 = MagicMock()
            mock_run1.data.metrics = {
                "test_rmse": 0.1,
                "test_mae": 0.05,
                "test_ll": -0.02
            }
            
            mock_run2 = MagicMock()
            mock_run2.data.metrics = {
                "test_rmse": 0.08,
                "test_mae": 0.04,
                "test_ll": -0.01
            }
            
            # Mock mlflow.get_run to return different runs based on run_id
            def mock_get_run(run_id):
                if run_id == "run1":
                    return mock_run1
                else:
                    return mock_run2
            
            mlflow.get_run = MagicMock(side_effect=mock_get_run)
            
            # List model versions
            versions = list_model_versions()
            
            # Verify versions
            self.assertEqual(len(versions), 2)
            
            # Verify first version
            self.assertEqual(versions[0]["version"], "1")
            self.assertEqual(versions[0]["run_id"], "run1")
            self.assertEqual(versions[0]["description"], "First version")
            self.assertEqual(versions[0]["status"], "Production")
            self.assertTrue(versions[0]["is_active"])
            self.assertEqual(versions[0]["test_rmse"], 0.1)
            self.assertEqual(versions[0]["test_mae"], 0.05)
            self.assertEqual(versions[0]["test_ll"], -0.02)
            
            # Verify second version
            self.assertEqual(versions[1]["version"], "2")
            self.assertEqual(versions[1]["run_id"], "run2")
            self.assertEqual(versions[1]["description"], "Second version")
            self.assertEqual(versions[1]["status"], "Staging")
            self.assertFalse(versions[1]["is_active"])
            self.assertEqual(versions[1]["test_rmse"], 0.08)
            self.assertEqual(versions[1]["test_mae"], 0.04)
            self.assertEqual(versions[1]["test_ll"], -0.01)
    
    def test_start_mlflow_server(self):
        """Test starting the MLflow server."""
        # Mock subprocess.Popen
        with patch('subprocess.Popen') as mock_popen:
            # Start MLflow server
            start_mlflow_server()
            
            # Verify server was started
            mock_popen.assert_called_once()
            
            # Verify command contains mlflow ui
            args = mock_popen.call_args[0][0]
            self.assertIn("mlflow", args)
            self.assertIn("ui", args)
            self.assertIn("5000", args)


if __name__ == "__main__":
    unittest.main()
