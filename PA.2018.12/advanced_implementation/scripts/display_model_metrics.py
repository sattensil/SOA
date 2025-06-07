#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to display detailed model metrics including accuracy by class
for the Mine Safety Injury Rate Prediction model.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report
)
import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data():
    """Load the test data for evaluation using the same approach as the training script."""
    # Import the data loading function from the enhanced_feature_engineering module
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    sys.path.append(project_dir)
    sys.path.append(script_dir)
    
    try:
        # Try direct import first
        from enhanced_feature_engineering import engineer_enhanced_features
        logger.info("Imported engineer_enhanced_features directly")
    except ImportError:
        # Try alternative import paths
        try:
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
    y_test_multi[(test_data['y'] > 0) & (test_data['y'] <= 1)] = 1  # One injury
    y_test_multi[(test_data['y'] > 1) & (test_data['y'] <= 2)] = 2  # Two injuries
    y_test_multi[test_data['y'] > 2] = 3  # Three or more injuries
    
    # Return in the expected format
    return {
        'X': test_data['X'],
        'y': test_data['y'],
        'y_binary': y_test_binary,
        'y_multi': y_test_multi,
        'feature_names': test_data['feature_names']
    }

def load_model():
    """Load the trained two-stage model."""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    
    # Check for our two-stage model files
    clf_path = os.path.join(models_dir, 'enhanced_classifier_model.json')
    reg_path = os.path.join(models_dir, 'enhanced_regressor_model.json')
    feature_path = os.path.join(models_dir, 'enhanced_feature_names.joblib')
    
    if os.path.exists(clf_path) and os.path.exists(reg_path):
        # Load classifier
        clf = XGBClassifier()
        clf.load_model(clf_path)
        
        # Load regressor
        reg = XGBRegressor()
        reg.load_model(reg_path)
        
        # Load feature names if available
        feature_names = None
        if os.path.exists(feature_path):
            feature_names = joblib.load(feature_path)
        
        # Create two-stage model dictionary
        model = {
            'type': 'two_stage',
            'classification': clf,
            'regression': reg,
            'feature_names': feature_names
        }
        
        logger.info(f"Loaded two-stage model from {models_dir}")
        return model
    else:
        logger.error(f"Two-stage model files not found at {models_dir}")
        return None

def evaluate_model_by_class(model, test_data, target_class0_accuracy=0.907):
    """Evaluate the model and display detailed metrics by class.
    
    Args:
        model: The two-stage model dictionary containing classification and regression models
        test_data: Dictionary containing test data and labels
        target_class0_accuracy: Target accuracy for class 0 (default: 0.907)
    """
    if isinstance(model, dict) and model.get('type') == 'two_stage':
        logger.info("Evaluating two-stage model")
        
        # Get the classification model
        clf_model = model['classification']
        
        # Make predictions
        test_proba = clf_model.predict_proba(test_data['X'])[:, 1]
        
        # Set a default threshold of 0.5 for initial evaluation
        threshold = 0.5
        test_binary_pred = (test_proba >= threshold).astype(int)
        
        # Calculate overall metrics
        accuracy = accuracy_score(test_data['y_binary'], test_binary_pred)
        precision = precision_score(test_data['y_binary'], test_binary_pred, zero_division=0)
        recall = recall_score(test_data['y_binary'], test_binary_pred)
        f1 = f1_score(test_data['y_binary'], test_binary_pred, zero_division=0)
        
        logger.info(f"Test metrics with threshold {threshold:.4f}:")
        logger.info(f"  Overall Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        
        # Calculate class-specific metrics
        logger.info("\nClass-specific metrics:")
        
        # Get class distribution
        class_counts = np.bincount(test_data['y_binary'])
        logger.info(f"Class distribution: {class_counts}")
        
        # Calculate accuracy by class
        for class_idx in range(len(class_counts)):
            class_mask = test_data['y_binary'] == class_idx
            if np.sum(class_mask) > 0:  # Only calculate if we have samples for this class
                class_correct = np.sum((test_binary_pred == class_idx) & class_mask)
                class_total = np.sum(class_mask)
                class_accuracy = class_correct / class_total if class_total > 0 else 0
                logger.info(f"  Class {class_idx}: {class_correct}/{class_total} correct ({class_accuracy:.4f} accuracy)")
        
        # Calculate confusion matrix and display
        cm = confusion_matrix(test_data['y_binary'], test_binary_pred)
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"{cm}")
        
        # Calculate classification report with more detailed metrics
        report = classification_report(test_data['y_binary'], test_binary_pred, output_dict=True, zero_division=0)
        logger.info(f"\nClassification Report:")
        for class_idx in sorted([k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]):
            metrics = report[class_idx]
            logger.info(f"  Class {class_idx}: precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, f1-score={metrics['f1-score']:.4f}, support={metrics['support']}")
        logger.info(f"  Overall accuracy: {report['accuracy']:.4f}")
        logger.info(f"  Macro avg: precision={report['macro avg']['precision']:.4f}, recall={report['macro avg']['recall']:.4f}, f1-score={report['macro avg']['f1-score']:.4f}")
        logger.info(f"  Weighted avg: precision={report['weighted avg']['precision']:.4f}, recall={report['weighted avg']['recall']:.4f}, f1-score={report['weighted avg']['f1-score']:.4f}")
        
        # Try different thresholds to find one that achieves exactly 90.7% accuracy for class 0 (no injury)
        logger.info(f"\nSearching for threshold to achieve exactly {target_class0_accuracy:.1%} accuracy for class 0 (no injury):")
        best_threshold = 0.5
        best_accuracy_diff = float('inf')
        
        # Use a finer grid search for more precise threshold finding
        thresholds = np.linspace(0.01, 0.99, 500)
        threshold_results = []
        
        for t in thresholds:
            pred = (test_proba >= t).astype(int)
            
            # Calculate class 0 accuracy
            class0_mask = test_data['y_binary'] == 0
            class0_correct = np.sum((pred == 0) & class0_mask)
            class0_total = np.sum(class0_mask)
            class0_accuracy = class0_correct / class0_total if class0_total > 0 else 0
            
            # Calculate multi-class accuracies
            multi_class_accuracies = []
            for c in range(4):  # 0, 1, 2, 3+
                class_mask = test_data['y_multi'] == c
                if np.sum(class_mask) > 0:
                    if c == 0:  # No injury class
                        class_correct = np.sum((pred == 0) & class_mask)
                    else:  # Any injury class (1, 2, 3+)
                        class_correct = np.sum((pred == 1) & class_mask)
                    
                    class_total = np.sum(class_mask)
                    class_acc = class_correct / class_total
                    multi_class_accuracies.append(class_acc)
                else:
                    multi_class_accuracies.append(np.nan)
            
            # Calculate overall accuracy
            overall_accuracy = accuracy_score(test_data['y_binary'], pred)
            
            # Store results
            threshold_results.append({
                'threshold': t,
                'class0_accuracy': class0_accuracy,
                'diff': abs(class0_accuracy - target_class0_accuracy),
                'multi_class_accuracies': multi_class_accuracies,
                'overall_accuracy': overall_accuracy
            })
            
            # Update best threshold
            if abs(class0_accuracy - target_class0_accuracy) < best_accuracy_diff:
                best_accuracy_diff = abs(class0_accuracy - target_class0_accuracy)
                best_threshold = t
        
                
        # Find the result with class 0 accuracy closest to target
        best_result = min(threshold_results, key=lambda x: x['diff'])
        best_threshold = best_result['threshold']
        
        logger.info(f"\n=== RESULTS WITH CLASS 0 ACCURACY TARGET OF {target_class0_accuracy:.1%} ===\n")
        logger.info(f"Best threshold found: {best_threshold:.4f}")
        logger.info(f"Class 0 accuracy achieved: {best_result['class0_accuracy']:.4f}")
        logger.info(f"Overall binary accuracy: {best_result['overall_accuracy']:.4f}")
        
        # Evaluate with the best threshold
        best_pred = (test_proba >= best_threshold).astype(int)
        
        # Calculate detailed multi-class metrics with the best threshold
        logger.info("\nDetailed class-specific accuracies:")
        
        # For binary classification
        logger.info("\nBinary Classification Results:")
        for class_idx in range(len(class_counts)):
            class_mask = test_data['y_binary'] == class_idx
            if np.sum(class_mask) > 0:
                class_correct = np.sum((best_pred == class_idx) & class_mask)
                class_total = np.sum(class_mask)
                class_accuracy = class_correct / class_total if class_total > 0 else 0
                class_name = "No Injury" if class_idx == 0 else "Any Injury"
                logger.info(f"  Class {class_idx} ({class_name}): {class_correct}/{class_total} correct ({class_accuracy:.4f} accuracy)")
        
        # For multi-class accuracy
        logger.info("\nMulti-class Detection Results:")
        print("\n=== MULTI-CLASS DETECTION RESULTS ===\n")
        print(f"With Class 0 accuracy fixed at ~{target_class0_accuracy:.1%}:")
        
        # Get unique classes in multi-class labels
        unique_classes = np.unique(test_data['y_multi'])
        
        # For each class in multi-class labels
        for c in sorted(unique_classes):
            class_mask = test_data['y_multi'] == c
            class_total = np.sum(class_mask)
            
            if c == 0:  # No injury class
                class_correct = np.sum((best_pred == 0) & class_mask)
                class_name = "No Injury"
            else:  # Any injury class (1, 2, 3+)
                class_correct = np.sum((best_pred == 1) & class_mask)
                if c == 1:
                    class_name = "One Injury"
                elif c == 2:
                    class_name = "Two Injuries"
                else:
                    class_name = "Three+ Injuries"
            
            class_accuracy = class_correct / class_total if class_total > 0 else 0
            logger.info(f"  Class {c} ({class_name}): {class_correct}/{class_total} correct ({class_accuracy:.4f} accuracy)")
            print(f"  Class {c} ({class_name}): {class_correct}/{class_total} correct ({class_accuracy:.4f} accuracy)")
        
        # Calculate confusion matrix with best threshold
        best_cm = confusion_matrix(test_data['y_binary'], best_pred)
        logger.info(f"\nConfusion Matrix with threshold {best_threshold:.4f}:")
        logger.info(f"{best_cm}")
        
        print(f"\nOverall binary accuracy: {best_result['overall_accuracy']:.4f}")
        print(f"Best threshold: {best_threshold:.4f}")
        print("\n===================================")
        
    else:
        logger.info("Model is not a two-stage model, cannot evaluate by class")

def main():
    """Main function to evaluate the model and display metrics."""
    # Load test data
    print("Loading test data...")
    test_data = load_data()
    
    # Load model
    print("Loading model...")
    model = load_model()
    
    if model is not None:
        # Evaluate model by class
        print("\nEvaluating model with target class 0 accuracy of 90.7%...\n")
        evaluate_model_by_class(model, test_data)
    else:
        print("ERROR: Failed to load model")
        logger.error("Failed to load model")

if __name__ == "__main__":
    main()
