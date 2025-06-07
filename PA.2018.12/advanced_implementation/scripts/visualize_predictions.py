#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization script for comparing actual vs. predicted values from the enhanced model.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from typing import Dict, Any, Tuple
import logging
from matplotlib.colors import LinearSegmentedColormap

# Set the style for all plots
plt.style.use('ggplot')
sns.set_palette("Greens")

# Create custom green colormap
greens = LinearSegmentedColormap.from_list('custom_greens', ['#CCFFCC', '#006400'], N=100)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import modules
try:
    # When imported as a module
    from .enhanced_feature_engineering import engineer_enhanced_features
except ImportError:
    # When run as a script
    from enhanced_feature_engineering import engineer_enhanced_features

# Define constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
MODEL_PATH = os.path.join(MODELS_DIR, "enhanced_xgboost_model.joblib")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "enhanced_feature_names.joblib")


def load_model_and_data() -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """
    Load the trained model and test data.
    
    Returns:
        Tuple[Any, Dict[str, Any], Dict[str, Any]]: Model, train data, and test data
    """
    logger.info(f"Loading model from {MODEL_PATH}")
    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"Successfully loaded model: {type(model)}")
        
        # Check if it's a two-stage model
        if isinstance(model, dict) and model.get('type') == 'two_stage':
            logger.info("Detected two-stage model with classification and regression components")
            logger.info(f"Classification model: {type(model['classification'])}")
            logger.info(f"Regression model: {type(model['regression'])}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
    logger.info("Loading feature names")
    try:
        feature_names = joblib.load(FEATURE_NAMES_PATH)
        logger.info(f"Loaded {len(feature_names)} feature names")
    except Exception as e:
        logger.error(f"Error loading feature names: {e}")
        raise
    
    logger.info("Loading train and test data")
    try:
        train_data, test_data = engineer_enhanced_features()
        logger.info(f"Loaded train data with {train_data['X'].shape[0]} samples and {train_data['X'].shape[1]} features")
        logger.info(f"Loaded test data with {test_data['X'].shape[0]} samples and {test_data['X'].shape[1]} features")
        
        # Check for zero-inflation in the data
        train_zeros = (train_data['y'] == 0).sum()
        train_nonzeros = (train_data['y'] > 0).sum()
        test_zeros = (test_data['y'] == 0).sum()
        test_nonzeros = (test_data['y'] > 0).sum()
        
        logger.info(f"Train data: {train_zeros} zeros ({train_zeros/len(train_data['y']):.1%}), {train_nonzeros} non-zeros")
        logger.info(f"Test data: {test_zeros} zeros ({test_zeros/len(test_data['y']):.1%}), {test_nonzeros} non-zeros")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
    
    return model, train_data, test_data


def create_prediction_dataframe(model: Any, data: Dict[str, Any], dataset_name: str) -> pd.DataFrame:
    """
    Create a dataframe with actual and predicted values.
    
    Args:
        model (Any): Trained model (single XGBoost or two-stage dict)
        data (Dict[str, Any]): Data dictionary with X, y, and weights
        dataset_name (str): Name of the dataset (train or test)
        
    Returns:
        pd.DataFrame: DataFrame with actual and predicted values
    """
    logger.info(f"Creating prediction dataframe for {dataset_name} data")
    
    # Get predictions based on model type
    if isinstance(model, dict) and model.get('type') == 'two_stage':
        # Two-stage model: classification * regression
        clf_model = model['classification']
        reg_model = model['regression']
        
        # Get probability of having injuries
        injury_proba = clf_model.predict_proba(data['X'])[:, 1]
        
        # Get predicted injury rate (if injury occurs)
        injury_rate = reg_model.predict(data['X'])
        
        # Final prediction: probability * rate
        predictions = injury_proba * injury_rate
        
        # Create dataframe with additional info
        df = pd.DataFrame({
            'actual': data['y'],
            'predicted': predictions,
            'injury_probability': injury_proba,
            'injury_rate_if_positive': injury_rate,
            'weight': data['weights'],
            'dataset': dataset_name
        })
    else:
        # Standard single model
        predictions = model.predict(data['X'])
        
        # Create dataframe
        df = pd.DataFrame({
            'actual': data['y'],
            'predicted': predictions,
            'weight': data['weights'],
            'dataset': dataset_name
        })
    
    # Add raw data if available
    if 'raw_data' in data:
        # Add key columns from raw data if they exist
        for col in ['YEAR', 'US_STATE', 'PRIMARY', 'EMP_HRS_TOTAL', 'NUM_INJURIES']:
            if col in data['raw_data'].columns:
                df[col] = data['raw_data'][col].values
    
    # Add residual column
    df['residual'] = df['actual'] - df['predicted']
    
    return df


def calculate_metrics(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate various model evaluation metrics.
    
    Args:
        train_df (pd.DataFrame): DataFrame with train predictions
        test_df (pd.DataFrame): DataFrame with test predictions
        
    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    metrics = {}
    
    # Standard metrics
    metrics['train_rmse'] = np.sqrt(((train_df['actual'] - train_df['predicted']) ** 2).mean())
    metrics['test_rmse'] = np.sqrt(((test_df['actual'] - test_df['predicted']) ** 2).mean())
    metrics['train_mae'] = np.abs(train_df['actual'] - train_df['predicted']).mean()
    metrics['test_mae'] = np.abs(test_df['actual'] - test_df['predicted']).mean()
    
    # Weighted metrics
    metrics['train_weighted_rmse'] = np.sqrt((((train_df['actual'] - train_df['predicted']) ** 2) * train_df['weight']).sum() / train_df['weight'].sum())
    metrics['test_weighted_rmse'] = np.sqrt((((test_df['actual'] - test_df['predicted']) ** 2) * test_df['weight']).sum() / test_df['weight'].sum())
    metrics['train_weighted_mae'] = (np.abs(train_df['actual'] - train_df['predicted']) * train_df['weight']).sum() / train_df['weight'].sum()
    metrics['test_weighted_mae'] = (np.abs(test_df['actual'] - test_df['predicted']) * test_df['weight']).sum() / test_df['weight'].sum()
    
    # Zero vs non-zero metrics
    train_zeros = train_df[train_df['actual'] == 0]
    test_zeros = test_df[test_df['actual'] == 0]
    train_nonzeros = train_df[train_df['actual'] > 0]
    test_nonzeros = test_df[test_df['actual'] > 0]
    
    metrics['train_zero_avg_pred'] = train_zeros['predicted'].mean() if len(train_zeros) > 0 else 0
    metrics['test_zero_avg_pred'] = test_zeros['predicted'].mean() if len(test_zeros) > 0 else 0
    
    metrics['train_nonzero_avg_pred'] = train_nonzeros['predicted'].mean() if len(train_nonzeros) > 0 else 0
    metrics['train_nonzero_avg_actual'] = train_nonzeros['actual'].mean() if len(train_nonzeros) > 0 else 0
    metrics['test_nonzero_avg_pred'] = test_nonzeros['predicted'].mean() if len(test_nonzeros) > 0 else 0
    metrics['test_nonzero_avg_actual'] = test_nonzeros['actual'].mean() if len(test_nonzeros) > 0 else 0
    
    metrics['train_nonzero_rmse'] = np.sqrt(((train_nonzeros['actual'] - train_nonzeros['predicted']) ** 2).mean()) if len(train_nonzeros) > 0 else 0
    metrics['test_nonzero_rmse'] = np.sqrt(((test_nonzeros['actual'] - test_nonzeros['predicted']) ** 2).mean()) if len(test_nonzeros) > 0 else 0
    
    # Count metrics
    metrics['train_zero_count'] = len(train_zeros)
    metrics['train_nonzero_count'] = len(train_nonzeros)
    metrics['test_zero_count'] = len(test_zeros)
    metrics['test_nonzero_count'] = len(test_nonzeros)
    
    # Add rounded accuracy metrics (for comparison with notebook)
    from sklearn.metrics import accuracy_score
    
    # Convert predictions to integers by rounding
    train_rounded_preds = np.round(train_df['predicted']).astype(int)
    test_rounded_preds = np.round(test_df['predicted']).astype(int)
    
    # Calculate accuracy
    train_actual_int = np.round(train_df['actual']).astype(int)
    test_actual_int = np.round(test_df['actual']).astype(int)
    
    metrics['train_rounded_accuracy'] = accuracy_score(train_actual_int, train_rounded_preds)
    metrics['test_rounded_accuracy'] = accuracy_score(test_actual_int, test_rounded_preds)
    
    # Binary classification metrics (if we have injury_probability column)
    if 'injury_probability' in train_df.columns and 'injury_probability' in test_df.columns:
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        # Create binary targets
        train_binary = (train_df['actual'] > 0).astype(int)
        test_binary = (test_df['actual'] > 0).astype(int)
        
        # Calculate AUC and Average Precision
        train_auc = roc_auc_score(train_binary, train_df['injury_probability'])
        test_auc = roc_auc_score(test_binary, test_df['injury_probability'])
        
        train_ap = average_precision_score(train_binary, train_df['injury_probability'])
        test_ap = average_precision_score(test_binary, test_df['injury_probability'])
        
        # Add to metrics
        classification_metrics = {
            'train_auc': train_auc,
            'test_auc': test_auc,
            'train_ap': train_ap,
            'test_ap': test_ap
        }
    else:
        classification_metrics = {}
    
    # Add classification metrics if available
    metrics.update(classification_metrics)
    
    return metrics


def visualize_rounded_predictions(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Create a heatmap visualization comparing rounded predicted vs. actual values.
    
    Args:
        train_df (pd.DataFrame): DataFrame with train predictions
        test_df (pd.DataFrame): DataFrame with test predictions
    """
    logger.info("Creating rounded predictions heatmap visualization")
    
    # Round predictions and actual values
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    train_df['rounded_pred'] = np.round(train_df['predicted']).astype(int)
    train_df['rounded_actual'] = np.round(train_df['actual']).astype(int)
    test_df['rounded_pred'] = np.round(test_df['predicted']).astype(int)
    test_df['rounded_actual'] = np.round(test_df['actual']).astype(int)
    
    # Combine train and test data
    train_df['dataset'] = 'train'
    test_df['dataset'] = 'test'
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Get unique values for both rounded actual and predicted
    all_values = sorted(set(combined_df['rounded_actual'].unique()) | set(combined_df['rounded_pred'].unique()))
    
    # Create a confusion matrix-like table
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    for i, dataset in enumerate(['train', 'test']):
        subset = combined_df[combined_df['dataset'] == dataset]
        
        # Create a pivot table for the heatmap
        pivot_table = pd.crosstab(
            subset['rounded_actual'], 
            subset['rounded_pred'],
            margins=False
        )
        
        # Fill in missing values with zeros
        for val in all_values:
            if val not in pivot_table.index:
                pivot_table.loc[val] = 0
            if val not in pivot_table.columns:
                pivot_table[val] = 0
        
        # Sort both axes
        pivot_table = pivot_table.sort_index().sort_index(axis=1)
        
        # Create the heatmap
        sns.heatmap(
            pivot_table, 
            annot=True, 
            fmt='.0f', 
            cmap=greens,
            ax=axes[i],
            cbar_kws={'label': 'Count'}
        )
        
        # Add diagonal line
        min_val = min(pivot_table.index.min(), pivot_table.columns.min())
        max_val = max(pivot_table.index.max(), pivot_table.columns.max())
        axes[i].plot([min_val - 0.5, max_val + 0.5], [min_val - 0.5, max_val + 0.5], 'r--', linewidth=1)
        
        axes[i].set_title(f'{dataset.capitalize()} Dataset - Rounded Predictions vs Actual')
    
    # Save the plot
    rounded_path = os.path.join(RESULTS_DIR, 'rounded_predictions_heatmap.png')
    logger.info(f"Attempting to save rounded predictions heatmap to {rounded_path}")
    plt.savefig(rounded_path, dpi=300)
    logger.info(f"Successfully saved rounded predictions heatmap to {rounded_path}")


def visualize_actual_vs_predicted(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Create visualizations comparing actual vs. predicted values.
    
    Args:
        train_df (pd.DataFrame): DataFrame with train predictions
        test_df (pd.DataFrame): DataFrame with test predictions
    """
    # Create directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Combine train and test data
    combined_df = pd.concat([train_df, test_df])
    combined_df['residual'] = combined_df['actual'] - combined_df['predicted']
    combined_df['abs_error'] = np.abs(combined_df['residual'])
    combined_df['squared_error'] = combined_df['residual'] ** 2
    
    # 1. Scatter plot of actual vs. predicted
    plt.figure(figsize=(12, 10))
    
    # Create a colormap based on weights
    weights = combined_df['weight']
    sizes = np.log1p(weights) * 10  # Log transform for better visualization
    
    # Plot train and test data with different markers and colors
    train_mask = combined_df['dataset'] == 'train'
    test_mask = combined_df['dataset'] == 'test'
    
    # Plot train data in light green
    plt.scatter(combined_df.loc[train_mask, 'actual'], 
               combined_df.loc[train_mask, 'predicted'], 
               s=sizes[train_mask],
               alpha=0.4, 
               color='#90EE90',  # Light green
               label='Train')
    
    # Plot test data in dark green
    plt.scatter(combined_df.loc[test_mask, 'actual'], 
               combined_df.loc[test_mask, 'predicted'], 
               s=sizes[test_mask],
               alpha=0.6, 
               color='#006400',  # Dark green
               marker='D',  # Diamond marker
               label='Test')
    
    # Perfect prediction line
    max_val = max(combined_df['actual'].max(), combined_df['predicted'].max())
    plt.plot([0, max_val], [0, max_val], 'g--', linewidth=2, label='Perfect Prediction')
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.xlabel('Actual Injury Rate', fontsize=14)
    plt.ylabel('Predicted Injury Rate', fontsize=14)
    plt.title('Actual vs Predicted Injury Rate', fontsize=16, fontweight='bold')
    plt.legend(loc='upper left')
    
    # Add annotation for metrics with green background
    metrics_text = f"Test RMSE: {calculate_metrics(train_df, test_df)['test_rmse']:.4f}\nTest MAE: {calculate_metrics(train_df, test_df)['test_mae']:.4f}"
    plt.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.5", fc="#CCFFCC", ec="#006400", alpha=0.9),
                 fontsize=12)
    
    scatter_path = os.path.join(RESULTS_DIR, 'actual_vs_predicted_scatter.png')
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=300)  # Higher resolution
    logger.info(f"Saved enhanced green scatter plot to {scatter_path}")
    
    # 2. Histogram of residuals
    plt.figure(figsize=(12, 8))
    
    # Use a custom green color palette
    green_colors = ['#CCFFCC', '#90EE90', '#3CB371', '#2E8B57', '#006400']
    
    # Create histogram with green colors
    n, bins, patches = plt.hist(combined_df['residual'], bins=50, alpha=0.8, color='#3CB371', edgecolor='white')
    
    # Color the bars based on distance from zero
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = (bin_centers - min(bin_centers)) / (max(bin_centers) - min(bin_centers))
    
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', plt.cm.Greens(c))
    
    # Add vertical line at zero
    plt.axvline(x=0, color='#006400', linestyle='--', linewidth=2, label='Zero Error')
    
    # Add mean residual line
    mean_residual = combined_df['residual'].mean()
    plt.axvline(x=mean_residual, color='red', linestyle='-', linewidth=2, 
                label=f'Mean Residual: {mean_residual:.4f}')
    
    plt.xlabel('Residual (Actual - Predicted)', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.title('Distribution of Prediction Errors', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Add summary statistics annotation
    stats_text = (f"Mean: {combined_df['residual'].mean():.4f}\n"
                  f"Std Dev: {combined_df['residual'].std():.4f}\n"
                  f"Min: {combined_df['residual'].min():.4f}\n"
                  f"Max: {combined_df['residual'].max():.4f}")
    
    plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.5", fc="#CCFFCC", ec="#006400", alpha=0.9),
                 fontsize=12, verticalalignment='top')
    
    hist_path = os.path.join(RESULTS_DIR, 'residuals_histogram.png')
    plt.tight_layout()
    plt.savefig(hist_path, dpi=300)  # Higher resolution
    logger.info(f"Saved enhanced green residuals histogram to {hist_path}")
    
    # 3. Box plot of residuals by state (if available)
    if 'US_STATE' in combined_df.columns:
        # Get top 10 states by number of records
        top_states = combined_df['US_STATE'].value_counts().head(10).index.tolist()
        state_df = combined_df[combined_df['US_STATE'].isin(top_states)].copy()
        
        plt.figure(figsize=(16, 10))
        
        # Calculate mean absolute error by state for sorting
        state_mae = combined_df.groupby('US_STATE')['abs_error'].mean().sort_values(ascending=False)
        sorted_states = state_mae.index.tolist()
        
        # Filter to top 20 states by error for better visualization
        top_states = sorted_states[:20]
        state_subset = combined_df[combined_df['US_STATE'].isin(top_states)]
        
        # Create custom green palette
        green_palette = sns.light_palette("green", n_colors=len(top_states), reverse=True)
        
        # Create boxplot with green colors, sorted by error magnitude
        ax = sns.boxplot(x='US_STATE', y='residual', data=state_subset, 
                        order=top_states, palette=green_palette)
        
        # Add swarm plot for individual points
        sns.swarmplot(x='US_STATE', y='residual', data=state_subset, 
                     order=top_states, color='#006400', size=3, alpha=0.5)
        
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('State (Top 20 by Error Magnitude)', fontsize=14)
        plt.ylabel('Residual (Actual - Predicted)', fontsize=14)
        plt.title('Prediction Errors by State', fontsize=16, fontweight='bold')
        plt.axhline(y=0, color='#006400', linestyle='--', linewidth=2)
        
        # Add state MAE values as text
        for i, state in enumerate(top_states):
            mae_value = state_mae[state]
            ax.text(i, ax.get_ylim()[0] * 0.9, f'MAE: {mae_value:.3f}', 
                    ha='center', va='bottom', rotation=90, fontsize=9, 
                    color='#006400', fontweight='bold')
        
        state_path = os.path.join(RESULTS_DIR, 'residuals_by_state.png')
        plt.tight_layout()
        plt.savefig(state_path, dpi=300)  # Higher resolution
        logger.info(f"Saved enhanced green state residuals to {state_path}")
    
    # Save prediction data to CSV with additional metrics
    csv_path = os.path.join(RESULTS_DIR, 'actual_vs_predicted.csv')
    
    # Add absolute error and squared error columns
    combined_df['abs_error'] = np.abs(combined_df['residual'])
    combined_df['squared_error'] = combined_df['residual'] ** 2
    
    # Add binary indicator for whether the mine had any injuries
    combined_df['has_injury'] = (combined_df['actual'] > 0).astype(int)
    
    # Add percentile rank of prediction within dataset
    combined_df['pred_percentile'] = combined_df.groupby('dataset')['predicted'].rank(pct=True)
    
    # Add flag for large errors (top 5% of absolute errors)
    error_threshold = combined_df['abs_error'].quantile(0.95)
    combined_df['large_error'] = (combined_df['abs_error'] > error_threshold).astype(int)
    
    # Save to CSV
    combined_df.to_csv(csv_path, index=False)
    logger.info(f"Saved enhanced prediction data to {csv_path}")
    
    # 5. Calculate and print metrics
    metrics = {}
    for dataset in ['train', 'test']:
        subset = combined_df[combined_df['dataset'] == dataset]
        metrics[f'{dataset}_rmse'] = np.sqrt(np.mean((subset['actual'] - subset['predicted'])**2))
        metrics[f'{dataset}_mae'] = np.mean(np.abs(subset['actual'] - subset['predicted']))
        metrics[f'{dataset}_weighted_rmse'] = np.sqrt(np.average(
            (subset['actual'] - subset['predicted'])**2, 
            weights=subset['weight']
        ))
        metrics[f'{dataset}_weighted_mae'] = np.average(
            np.abs(subset['actual'] - subset['predicted']), 
            weights=subset['weight']
        )
    
    # Print metrics
    print("\nPrediction Metrics:")
    print(f"Train RMSE: {metrics['train_rmse']:.4f}")
    print(f"Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"Train MAE: {metrics['train_mae']:.4f}")
    print(f"Test MAE: {metrics['test_mae']:.4f}")
    print(f"Train Weighted RMSE: {metrics['train_weighted_rmse']:.4f}")
    print(f"Test Weighted RMSE: {metrics['test_weighted_rmse']:.4f}")
    print(f"Train Weighted MAE: {metrics['train_weighted_mae']:.4f}")
    print(f"Test Weighted MAE: {metrics['test_weighted_mae']:.4f}")
    
    # Save metrics to JSON
    import json
    metrics_path = os.path.join(RESULTS_DIR, 'prediction_metrics.json')


def main():
    """
    Main function to run the visualization script.
    """
    logger.info("Starting visualization script")
    
    # Load model and data
    model, train_data, test_data = load_model_and_data()
    
    # Create prediction dataframes
    train_df = create_prediction_dataframe(model, train_data, 'train')
    test_df = create_prediction_dataframe(model, test_data, 'test')
    
    # Calculate metrics
    metrics = calculate_metrics(train_df, test_df)
    
    # Print metrics
    logger.info("Model evaluation metrics:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value:.6f}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])
    metrics_path = os.path.join(RESULTS_DIR, 'model_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Visualize predictions
    visualize_actual_vs_predicted(train_df, test_df)
    
    # Visualize rounded predictions
    visualize_rounded_predictions(train_df, test_df)
    
    # If two-stage model, visualize components
    if 'injury_probability' in train_df.columns and 'injury_rate_if_positive' in train_df.columns:
        logger.info("Detected two-stage model, creating component visualizations")
        visualize_two_stage_components(train_df, test_df)
    
    logger.info("Visualization script completed successfully")


if __name__ == "__main__":
    main()
