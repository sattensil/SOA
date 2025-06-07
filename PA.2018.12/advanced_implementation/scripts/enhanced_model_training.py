#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced model training module for the Mine Safety Injury Rate Prediction model.
Implements a two-stage model with binary classification followed by Tweedie regression.
"""

import pandas as pd
import numpy as np
import os
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, List, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.fmin import generate_trials_to_calculate

# Import modules
try:
    # When imported as a module
    from .enhanced_feature_engineering import engineer_enhanced_features
except ImportError:
    # When run as a script
    from enhanced_feature_engineering import engineer_enhanced_features

# Import ADASYN for oversampling
try:
    from imblearn.over_sampling import ADASYN
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    logger = logging.getLogger(__name__)
    logger.warning("imblearn not installed. ADASYN oversampling will not be available.")
    logger.warning("Install with: pip install imbalanced-learn")

# Define directories for saving models and results
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'model_training.log'))
    ]
)
logger = logging.getLogger(__name__)

# Define constants
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data():
    """
    Load and preprocess the data for model training.
    
    Returns:
        Tuple containing training, validation, and test data with binary targets
    """
    # Engineer enhanced features
    train_data, test_data = engineer_enhanced_features()
    
    # Extract features and target
    X_train = train_data['X']
    y_train = train_data['y']
    X_test = test_data['X']
    y_test = test_data['y']
    feature_names = train_data['feature_names']
    
    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=53
    )
    
    # Create binary targets for classification (0 = no injury, 1 = injury)
    y_train_binary = (y_train > 0).astype(int)
    y_val_binary = (y_val > 0).astype(int)
    y_test_binary = (y_test > 0).astype(int)
    
    # Create filtered datasets for regression (only injury cases)
    train_injury_mask = y_train > 0
    val_injury_mask = y_val > 0
    test_injury_mask = y_test > 0
    
    X_train_injury = X_train[train_injury_mask]
    y_train_injury = y_train[train_injury_mask]
    X_val_injury = X_val[val_injury_mask]
    y_val_injury = y_val[val_injury_mask]
    X_test_injury = X_test[test_injury_mask]
    y_test_injury = y_test[test_injury_mask]
    
    logger.info(f"Data loaded: Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    logger.info(f"Binary class distribution - Train: {np.bincount(y_train_binary)}, Val: {np.bincount(y_val_binary)}, Test: {np.bincount(y_test_binary)}")
    logger.info(f"Injury-only data - Train: {X_train_injury.shape}, Val: {X_val_injury.shape}, Test: {X_test_injury.shape}")
    
    # Return both full datasets and injury-only datasets
    full_data = (X_train, y_train, y_train_binary, X_val, y_val, y_val_binary, X_test, y_test, y_test_binary)
    injury_data = (X_train_injury, y_train_injury, X_val_injury, y_val_injury, X_test_injury, y_test_injury)
    
    return full_data, injury_data, feature_names

def train_binary_classifier(full_data, feature_names):
    """
    Train a binary classifier to predict injury vs. no injury.
    
    Args:
        full_data: Tuple containing training, validation, and test data with binary targets
        feature_names: List of feature names
        
    Returns:
        Tuple containing the trained model and evaluation metrics
    """
    # Unpack data
    X_train, y_train, y_train_binary, X_val, y_val, y_val_binary, X_test, y_test, y_test_binary = full_data
    
    logger.info("Training binary classification model (Stage 1)")
    
    # Define objective function for hyperparameter optimization
    def objective(params):
        # Handle class weight parameter separately
        class_weight = params.pop('class_weight')
        scale_pos_weight = class_weight  # XGBoost uses scale_pos_weight for binary classification
        
        # Cast integer parameters
        params['max_depth'] = int(params['max_depth'])
        params['n_estimators'] = int(params['n_estimators'])
        params['min_child_weight'] = int(params['min_child_weight'])
        
        # Create classifier with these parameters
        clf = XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            scale_pos_weight=scale_pos_weight,
            random_state=53,
            use_label_encoder=False,
            **params
        )
        
        # Train with early stopping
        clf.fit(
            X_train, y_train_binary,
            eval_set=[(X_val, y_val_binary)],
            verbose=False
        )
        
        # Get predictions and calculate loss (1 - F1 score)
        y_val_pred = clf.predict(X_val)
        f1 = f1_score(y_val_binary, y_val_pred)
        loss = 1 - f1
        
        return {'loss': loss, 'status': STATUS_OK}
    
    # Define hyperparameter search space for binary classification
    space = {
        'learning_rate': hp.loguniform('learning_rate', -5, -1),  # 0.001 to 0.1
        'n_estimators': hp.quniform('n_estimators', 100, 500, 10),
        'max_depth': hp.quniform('max_depth', 3, 10, 1),
        'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
        'subsample': hp.uniform('subsample', 0.5, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
        'gamma': hp.uniform('gamma', 0, 5),
        'reg_alpha': hp.loguniform('reg_alpha', -5, 2),
        'reg_lambda': hp.loguniform('reg_lambda', -5, 2),
        'class_weight': hp.uniform('class_weight', 5, 100)  # Weight for minority class
    }
    
    # Initial parameters for binary classification
    initial_params = {
        'learning_rate': 0.05,
        'n_estimators': 200,
        'max_depth': 5,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'class_weight': 20.0
    }
    
    # Generate trials with initial parameters
    trials = generate_trials_to_calculate([initial_params])
    
    # Run hyperparameter optimization
    logger.info("Starting hyperparameter optimization for binary classifier")
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=10,  # Reduced for faster execution
        trials=trials,
        verbose=False
    )
    
    logger.info(f"Best hyperparameters for binary classifier: {best}")
    
    # Prepare final model with best parameters
    class_weight = best.pop('class_weight')
    best['max_depth'] = int(best['max_depth'])
    best['n_estimators'] = int(best['n_estimators'])
    best['min_child_weight'] = int(best['min_child_weight'])
    
    # Train final binary classification model
    clf = XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
        **best
    )
    
    # Apply class weights
    sample_weight = np.ones(len(y_train_binary))
    sample_weight[y_train_binary == 1] = class_weight
    
    clf.fit(
        X_train, 
        y_train_binary,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val_binary)],
        verbose=False
    )
    
    # Evaluate binary classifier
    train_pred = clf.predict(X_train)
    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train_binary, train_pred)
    val_accuracy = accuracy_score(y_val_binary, val_pred)
    test_accuracy = accuracy_score(y_test_binary, test_pred)
    
    train_precision = precision_score(y_train_binary, train_pred)
    val_precision = precision_score(y_val_binary, val_pred)
    test_precision = precision_score(y_test_binary, test_pred)
    
    train_recall = recall_score(y_train_binary, train_pred)
    val_recall = recall_score(y_val_binary, val_pred)
    test_recall = recall_score(y_test_binary, test_pred)
    
    train_f1 = f1_score(y_train_binary, train_pred)
    val_f1 = f1_score(y_val_binary, val_pred)
    test_f1 = f1_score(y_test_binary, test_pred)
    
    # Confusion matrices
    train_cm = confusion_matrix(y_train_binary, train_pred)
    val_cm = confusion_matrix(y_val_binary, val_pred)
    test_cm = confusion_matrix(y_test_binary, test_pred)
    
    # Compile metrics
    metrics = {
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy,
        'train_precision': train_precision,
        'val_precision': val_precision,
        'test_precision': test_precision,
        'train_recall': train_recall,
        'val_recall': val_recall,
        'test_recall': test_recall,
        'train_f1': train_f1,
        'val_f1': val_f1,
        'test_f1': test_f1,
        'train_confusion_matrix': train_cm.tolist(),
        'val_confusion_matrix': val_cm.tolist(),
        'test_confusion_matrix': test_cm.tolist()
    }
    
    # Log key metrics
    logger.info(f"Binary classifier metrics:")
    logger.info(f"  Train - Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
    logger.info(f"  Val   - Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
    logger.info(f"  Test  - Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    importance = clf.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    top_n = min(20, len(feature_names))
    plt.title('Binary Classifier Feature Importance')
    plt.barh(range(top_n), importance[indices][:top_n], align='center')
    plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]])
    plt.xlabel('Importance')
    
    # Save plot
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "binary_classifier_feature_importance.png")
    plt.savefig(plot_path)
    logger.info(f"Binary classifier feature importance plot saved to {plot_path}")
    
    return clf, metrics


def train_injury_regressor(injury_data, feature_names):
    """
    Train a Tweedie regression model to predict injury counts for cases with injuries.
    Uses ADASYN to balance injury classes before training.
    
    Args:
        injury_data: Tuple containing training, validation, and test data for injury cases
        feature_names: List of feature names
        
    Returns:
        Tuple containing the trained model and evaluation metrics
    """
    # Unpack data
    X_train_injury, y_train_injury, X_val_injury, y_val_injury, X_test_injury, y_test_injury = injury_data
    
    logger.info("Training injury regression model (Stage 2)")
    
    # Apply ADASYN to balance injury classes
    if HAS_IMBLEARN:
        logger.info("Applying ADASYN to balance injury classes")
        
        # Create injury classes for ADASYN (we need categorical targets)
        # Cap at 3+ injuries to ensure enough samples in each class
        y_train_injury_class = np.minimum(y_train_injury, 3).astype(int)
        
        # Check class distribution
        class_counts = np.bincount(y_train_injury_class)
        logger.info(f"Original class distribution: {class_counts}")
        
        try:
            # Dynamically adjust n_neighbors based on smallest class size
            min_samples = min(count for count in class_counts if count > 0)
            n_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
            
            logger.info(f"Using n_neighbors={n_neighbors} for ADASYN (min class samples: {min_samples})")
            
            # Apply ADASYN with adjusted parameters
            adasyn = ADASYN(random_state=53, n_neighbors=n_neighbors)
            X_train_injury_resampled, y_train_injury_class_resampled = adasyn.fit_resample(
                X_train_injury, y_train_injury_class
            )
            
            # Log resampled distribution
            resampled_counts = np.bincount(y_train_injury_class_resampled)
            logger.info(f"Resampled class distribution: {resampled_counts}")
            
            # Convert class labels back to actual injury counts for regression
            # Map each synthetic sample to a realistic injury count based on its class
            class_to_counts_map = {}
            for cls in np.unique(y_train_injury_class):
                class_to_counts_map[cls] = y_train_injury[y_train_injury_class == cls]
            
            y_train_injury_resampled = np.zeros(len(y_train_injury_class_resampled))
            
            # For original samples, keep their original values
            original_indices = adasyn.sample_indices_
            for i in original_indices:
                y_train_injury_resampled[i] = y_train_injury[i % len(y_train_injury)]
            
            # For synthetic samples, assign a value from the same class
            synthetic_indices = [i for i in range(len(y_train_injury_class_resampled)) if i not in original_indices]
            for i in synthetic_indices:
                cls = y_train_injury_class_resampled[i]
                if len(class_to_counts_map[cls]) > 0:
                    y_train_injury_resampled[i] = np.random.choice(class_to_counts_map[cls])
                else:
                    # Fallback to class value if no samples (shouldn't happen)
                    y_train_injury_resampled[i] = float(cls)
            
            # Replace original training data with resampled data
            X_train_injury = X_train_injury_resampled
            y_train_injury = y_train_injury_resampled
            
            logger.info(f"ADASYN applied successfully. New training data shape: {X_train_injury.shape}")
            
        except Exception as e:
            logger.warning(f"ADASYN resampling failed: {str(e)}. Using original data.")
            logger.warning("This might happen if a class has too few samples.")
            # Continue with original data
    else:
        logger.warning("imblearn not installed. Using original data without ADASYN.")
    
    # Define objective function for hyperparameter optimization
    def objective(params):
        # Extract Tweedie variance power
        power = params.pop('tweedie_variance_power')
        
        # Cast integer parameters
        params['max_depth'] = int(params['max_depth'])
        params['n_estimators'] = int(params['n_estimators'])
        params['min_child_weight'] = int(params['min_child_weight'])
        
        # Create regressor with these parameters
        reg = XGBRegressor(
            objective='reg:tweedie',
            tweedie_variance_power=power,
            eval_metric=f'tweedie-nloglik@{power}',
            random_state=53,
            **params
        )
        
        # Train with early stopping
        reg.fit(
            X_train_injury, y_train_injury,
            eval_set=[(X_val_injury, y_val_injury)],
            verbose=False
        )
        
        # Calculate loss (MAE)
        val_pred = reg.predict(X_val_injury)
        loss = mean_absolute_error(y_val_injury, val_pred)
        
        return {'loss': loss, 'status': STATUS_OK}
    
    # Define hyperparameter search space for regression
    # Note: Tweedie variance power closer to 1 (Poisson-like) for injury count prediction
    space = {
        'tweedie_variance_power': hp.uniform('tweedie_variance_power', 1.0, 1.2),  # Closer to Poisson
        'learning_rate': hp.loguniform('learning_rate', -5, -1),  # 0.001 to 0.1
        'n_estimators': hp.quniform('n_estimators', 100, 500, 10),
        'max_depth': hp.quniform('max_depth', 3, 10, 1),
        'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
        'subsample': hp.uniform('subsample', 0.5, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
        'gamma': hp.uniform('gamma', 0, 5),
        'reg_alpha': hp.loguniform('reg_alpha', -5, 2),
        'reg_lambda': hp.loguniform('reg_lambda', -5, 2)
    }
    
    # Initial parameters for regression
    initial_params = {
        'tweedie_variance_power': 1.1,  # Closer to Poisson
        'learning_rate': 0.05,
        'n_estimators': 200,
        'max_depth': 5,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0
    }
    
    # Generate trials with initial parameters
    trials = generate_trials_to_calculate([initial_params])
    
    # Run hyperparameter optimization
    logger.info("Starting hyperparameter optimization for injury regressor")
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=10,  # Reduced for faster execution
        trials=trials,
        verbose=False
    )
    
    logger.info(f"Best hyperparameters for injury regressor: {best}")
    
    # Prepare final model with best parameters
    best_power = best.pop('tweedie_variance_power')
    best_params = best.copy()
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['min_child_weight'] = int(best_params['min_child_weight'])
    
    # Create sample weights to prioritize higher injury counts (2+ injuries)
    sample_weights = np.ones_like(y_train_injury)
    sample_weights[y_train_injury > 1] = 5.0  # 5x weight for samples with 2+ injuries
    sample_weights[y_train_injury > 2] = 10.0  # 10x weight for samples with 3+ injuries
    
    # Train final model with best parameters and sample weights
    final_reg = XGBRegressor(
        objective='reg:tweedie',
        tweedie_variance_power=best_power,
        eval_metric=f'tweedie-nloglik@{best_power}',
        random_state=53,
        **best_params
    )
    
    final_reg.fit(
        X_train_injury, y_train_injury,
        sample_weight=sample_weights,
        eval_set=[(X_val_injury, y_val_injury)],
        verbose=False
    )
    
    # Evaluate regression model
    train_pred = final_reg.predict(X_train_injury)
    val_pred = final_reg.predict(X_val_injury)
    test_pred = final_reg.predict(X_test_injury)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train_injury, train_pred)
    val_mse = mean_squared_error(y_val_injury, val_pred)
    test_mse = mean_squared_error(y_test_injury, test_pred)
    
    train_rmse = np.sqrt(train_mse)
    val_rmse = np.sqrt(val_mse)
    test_rmse = np.sqrt(test_mse)
    
    train_mae = mean_absolute_error(y_train_injury, train_pred)
    val_mae = mean_absolute_error(y_val_injury, val_pred)
    test_mae = mean_absolute_error(y_test_injury, test_pred)
    
    # Calculate rounded accuracy
    y_train_rounded = np.round(y_train_injury).astype(int)
    y_val_rounded = np.round(y_val_injury).astype(int)
    y_test_rounded = np.round(y_test_injury).astype(int)
    
    train_pred_rounded = np.round(train_pred).astype(int)
    val_pred_rounded = np.round(val_pred).astype(int)
    test_pred_rounded = np.round(test_pred).astype(int)
    
    train_accuracy = accuracy_score(y_train_rounded, train_pred_rounded)
    val_accuracy = accuracy_score(y_val_rounded, val_pred_rounded)
    test_accuracy = accuracy_score(y_test_rounded, test_pred_rounded)
    
    # Compile metrics
    metrics = {
        'train_mse': train_mse,
        'val_mse': val_mse,
        'test_mse': test_mse,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'val_mae': val_mae,
        'test_mae': test_mae,
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy
    }
    
    # Log key metrics
    logger.info(f"Injury regressor metrics:")
    logger.info(f"  Train - RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, Accuracy: {train_accuracy:.4f}")
    logger.info(f"  Val   - RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, Accuracy: {val_accuracy:.4f}")
    logger.info(f"  Test  - RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, Accuracy: {test_accuracy:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    importance = final_reg.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    top_n = min(20, len(feature_names))
    plt.title('Injury Regressor Feature Importance')
    plt.barh(range(top_n), importance[indices][:top_n], align='center')
    plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]])
    plt.xlabel('Importance')
    
    # Save plot
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "injury_regressor_feature_importance.png")
    plt.savefig(plot_path)
    logger.info(f"Injury regressor feature importance plot saved to {plot_path}")
    
    return final_reg, metrics


def predict_two_stage(clf, reg, X):
    """
    Make predictions using the two-stage model.
    
    Args:
        clf: Binary classifier (stage 1)
        reg: Regression model (stage 2)
        X: Features to predict on
        
    Returns:
        Array of predictions
    """
    # Stage 1: Binary classification
    binary_preds = clf.predict(X)
    
    # Stage 2: Regression for injury cases
    injury_preds = np.zeros_like(binary_preds, dtype=float)
    injury_mask = binary_preds > 0
    
    if np.any(injury_mask):
        injury_preds[injury_mask] = reg.predict(X[injury_mask])
    
    return injury_preds


def evaluate_two_stage_model(clf, reg, full_data):
    """
    Evaluate the two-stage model on training, validation, and test data.
    
    Args:
        clf: Binary classifier (stage 1)
        reg: Regression model (stage 2)
        full_data: Tuple containing training, validation, and test data
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Unpack data
    X_train, y_train, _, X_val, y_val, _, X_test, y_test, _ = full_data
    
    # Make predictions
    train_pred = predict_two_stage(clf, reg, X_train)
    val_pred = predict_two_stage(clf, reg, X_val)
    test_pred = predict_two_stage(clf, reg, X_test)
    
    # Calculate regression metrics
    train_mse = mean_squared_error(y_train, train_pred)
    val_mse = mean_squared_error(y_val, val_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    
    train_rmse = np.sqrt(train_mse)
    val_rmse = np.sqrt(val_mse)
    test_rmse = np.sqrt(test_mse)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    # Calculate rounded accuracy
    train_pred_rounded = np.round(train_pred).astype(int)
    val_pred_rounded = np.round(val_pred).astype(int)
    test_pred_rounded = np.round(test_pred).astype(int)
    
    y_train_rounded = np.round(y_train).astype(int)
    y_val_rounded = np.round(y_val).astype(int)
    y_test_rounded = np.round(y_test).astype(int)
    
    train_accuracy = np.mean(y_train_rounded == train_pred_rounded)
    val_accuracy = np.mean(y_val_rounded == val_pred_rounded)
    test_accuracy = np.mean(y_test_rounded == test_pred_rounded)
    
    # Calculate class-specific accuracy
    metrics = {
        'train_mse': train_mse,
        'val_mse': val_mse,
        'test_mse': test_mse,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'val_mae': val_mae,
        'test_mae': test_mae,
        'train_rounded_accuracy': train_accuracy,
        'val_rounded_accuracy': val_accuracy,
        'test_rounded_accuracy': test_accuracy
    }
    
    # Calculate class-specific accuracy
    for i in range(5):  # Assuming classes 0-4
        train_mask = y_train_rounded == i
        val_mask = y_val_rounded == i
        test_mask = y_test_rounded == i
        
        if np.any(train_mask):
            train_class_acc = np.mean(train_pred_rounded[train_mask] == i)
            metrics[f'train_class_{i}_accuracy'] = train_class_acc
        
        if np.any(val_mask):
            val_class_acc = np.mean(val_pred_rounded[val_mask] == i)
            metrics[f'val_class_{i}_accuracy'] = val_class_acc
        
        if np.any(test_mask):
            test_class_acc = np.mean(test_pred_rounded[test_mask] == i)
            metrics[f'test_class_{i}_accuracy'] = test_class_acc
    
    return metrics


def save_models(clf, reg, feature_names):
    """
    Save the two-stage model to disk.
    
    Args:
        clf: Binary classifier (stage 1)
        reg: Regression model (stage 2)
        feature_names: List of feature names
    """
    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save models
    clf_path = os.path.join(MODELS_DIR, "binary_classifier.joblib")
    reg_path = os.path.join(MODELS_DIR, "injury_regressor.joblib")
    feature_path = os.path.join(MODELS_DIR, "feature_names.joblib")
    
    joblib.dump(clf, clf_path)
    joblib.dump(reg, reg_path)
    joblib.dump(feature_names, feature_path)
    
    logger.info(f"Models saved to {MODELS_DIR}")


def main():
    """Main function to train and evaluate the two-stage model."""
    logger.info("Starting two-stage model training")
    
    # Create directories if they don't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load data
    full_data, injury_data, feature_names = load_data()
    
    # Train binary classifier (Stage 1)
    clf, clf_metrics = train_binary_classifier(full_data, feature_names)
    
    # Train injury regressor (Stage 2)
    reg, reg_metrics = train_injury_regressor(injury_data, feature_names)
    
    # Evaluate the two-stage model
    logger.info("Evaluating two-stage model")
    combined_metrics = evaluate_two_stage_model(clf, reg, full_data)
    
    # Save models
    save_models(clf, reg, feature_names)
    
    # Print evaluation metrics
    print("\nTwo-Stage Model Evaluation:")
    print(f"MSE - Train: {combined_metrics['train_mse']:.4f}, Val: {combined_metrics['val_mse']:.4f}, Test: {combined_metrics['test_mse']:.4f}")
    print(f"RMSE - Train: {combined_metrics['train_rmse']:.4f}, Val: {combined_metrics['val_rmse']:.4f}, Test: {combined_metrics['test_rmse']:.4f}")
    print(f"MAE - Train: {combined_metrics['train_mae']:.4f}, Val: {combined_metrics['val_mae']:.4f}, Test: {combined_metrics['test_mae']:.4f}")
    print(f"Rounded Accuracy - Train: {combined_metrics['train_rounded_accuracy']:.4f}, Val: {combined_metrics['val_rounded_accuracy']:.4f}, Test: {combined_metrics['test_rounded_accuracy']:.4f}")
    
    print("\nClass-Specific Accuracy:")
    for i in range(5):  # Assuming classes 0-4
        train_key = f'train_class_{i}_accuracy'
        val_key = f'val_class_{i}_accuracy'
        test_key = f'test_class_{i}_accuracy'
        
        train_acc = combined_metrics.get(train_key, float('nan'))
        val_acc = combined_metrics.get(val_key, float('nan'))
        test_acc = combined_metrics.get(test_key, float('nan'))
        
        print(f"Class {i}: Train={train_acc:.4f}, Val={val_acc:.4f}, Test={test_acc:.4f}")
    
    logger.info("Two-stage model training and evaluation complete")

if __name__ == "__main__":
    main()
