#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Minimal version of the edu-2018-12-exam-pa-rmd solution in Python
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(1234)

# Custom log-likelihood function for Poisson model
def ll_function(targets, predicted_values):
    """Calculate log-likelihood for Poisson model"""
    p_v_zero = np.where(predicted_values <= 0, 0, predicted_values)
    p_v_pos = np.where(predicted_values <= 0, 0.000001, predicted_values)
    return np.sum(targets * np.log(p_v_pos)) - np.sum(p_v_zero)

# Main function
def main():
    """Main function to run the analysis"""
    # Load data
    print("Loading data...")
    data_all = pd.read_csv("MSHA_Mine_Data_2013-2016.csv")
    
    # Basic data cleaning
    print("Cleaning data...")
    data_nomissing = data_all.dropna(subset=['MINE_STATUS', 'US_STATE', 'PRIMARY'])
    data_reduced = data_nomissing.copy()
    data_reduced = data_reduced.drop(['PRIMARY', 'US_STATE'], axis=1)
    
    # Create target variable
    data_reduced['INJ_RATE_PER2K'] = data_reduced['NUM_INJURIES'] / (data_reduced['EMP_HRS_TOTAL'] / 2000)
    
    # Remove closed mines and low employee hours
    no_good = ["Closed by MSHA", "Non-producing", "Permanently abandoned", "Temporarily closed"]
    data_reduced2 = data_reduced[~data_reduced['MINE_STATUS'].isin(no_good)]
    data_reduced3 = data_reduced2[data_reduced2['EMP_HRS_TOTAL'] >= 2000]
    
    # Combine coal and non-coal categories
    data_reduced3['ADJ_STATUS'] = 'Other'
    data_reduced3.loc[data_reduced3['MINE_STATUS'] == 'Active', 'ADJ_STATUS'] = 'Open'
    data_reduced3.loc[data_reduced3['MINE_STATUS'] == 'Full-time permanent', 'ADJ_STATUS'] = 'Open'
    data_reduced3.loc[data_reduced3['MINE_STATUS'] == 'Intermittent', 'ADJ_STATUS'] = 'Intermittent'
    data_reduced3 = data_reduced3.drop('MINE_STATUS', axis=1)
    
    # Remove year as it doesn't show significant differences
    data_reduced3 = data_reduced3.drop('YEAR', axis=1)
    
    # Simple train-test split
    print("Splitting data...")
    train, test = train_test_split(
        data_reduced3, 
        test_size=0.25, 
        random_state=1234
    )
    
    # Reset indices
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    
    print(f"Train size: {len(train)}, Test size: {len(test)}")
    
    # Build a simple decision tree
    print("Building decision tree model...")
    X_train = train.drop(['NUM_INJURIES', 'EMP_HRS_TOTAL', 'INJ_RATE_PER2K'], axis=1)
    X_test = test.drop(['NUM_INJURIES', 'EMP_HRS_TOTAL', 'INJ_RATE_PER2K'], axis=1)
    
    # Convert categorical variables to dummy variables
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)
    
    # Ensure X_test has the same columns as X_train
    for col in X_train.columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[X_train.columns]
    
    # Target and weights
    y_train = train['NUM_INJURIES']
    y_test = test['NUM_INJURIES']
    weights_train = train['EMP_HRS_TOTAL'] / 2000
    weights_test = test['EMP_HRS_TOTAL'] / 2000
    
    # Fit tree model
    tree = DecisionTreeRegressor(
        min_samples_leaf=25,
        max_depth=3,
        random_state=153
    )
    tree.fit(X_train, y_train, sample_weight=weights_train)
    
    # Predictions
    train_pred = tree.predict(X_train)
    test_pred = tree.predict(X_test)
    
    # Evaluate with log-likelihood
    ll_train = ll_function(train['NUM_INJURIES'], train_pred * weights_train)
    ll_test = ll_function(test['NUM_INJURIES'], test_pred * weights_test)
    
    print(f"Tree - Train log-likelihood: {ll_train:.2f}")
    print(f"Tree - Test log-likelihood: {ll_test:.2f}")
    
    # Plot tree
    plt.figure(figsize=(20, 10))
    plot_tree(tree, feature_names=X_train.columns, filled=True, max_depth=3)
    plt.title('Decision Tree (Simplified Model)')
    plt.tight_layout()
    plt.savefig('decision_tree.png')
    
    # Feature importance
    print("\nFeature importance:")
    importance = sorted(zip(X_train.columns, tree.feature_importances_), 
                      key=lambda x: x[1], reverse=True)
    for feature, imp in importance[:10]:
        print(f"{feature}: {imp:.4f}")
    
    # Build a simple GLM
    print("\nBuilding GLM model...")
    # Create a combined variable for MINE_CHAR
    data_reduced4 = data_reduced3.copy()
    data_reduced4['MINE_CHAR'] = data_reduced4['TYPE_OF_MINE'] + ' ' + data_reduced4['COMMODITY']
    data_reduced4['MINE_CHAR'] = data_reduced4['MINE_CHAR'].astype('category')
    
    # Take log of AVG_EMP_TOTAL
    data_reduced4['LOG_AVG_EMP_TOTAL'] = np.log(data_reduced4['AVG_EMP_TOTAL'])
    data_reduced4 = data_reduced4.drop('AVG_EMP_TOTAL', axis=1)
    
    # Split again
    train_glm, test_glm = train_test_split(
        data_reduced4, 
        test_size=0.25, 
        random_state=1234
    )
    
    # Reset indices
    train_glm = train_glm.reset_index(drop=True)
    test_glm = test_glm.reset_index(drop=True)
    
    # Simple formula
    formula = """NUM_INJURIES ~ SEAM_HEIGHT + PCT_HRS_UNDERGROUND + 
                 PCT_HRS_MILL_PREP + PCT_HRS_OFFICE + LOG_AVG_EMP_TOTAL"""
    
    # Offset
    offset = np.log(train_glm['EMP_HRS_TOTAL'] / 2000)
    
    # Fit GLM
    try:
        glm = smf.glm(
            formula=formula,
            data=train_glm,
            family=sm.families.Poisson(),
            offset=offset
        ).fit()
        
        print(glm.summary())
        
        # Predictions
        test_offset = np.log(test_glm['EMP_HRS_TOTAL'] / 2000)
        glm_pred = glm.predict(test_glm, offset=test_offset)
        
        # Evaluate
        ll_glm = ll_function(test_glm['NUM_INJURIES'], glm_pred)
        print(f"GLM - Test log-likelihood: {ll_glm:.2f}")
        
        # Effect of coefficients
        print("\nEffect of a unit change in a predictor, in percent:")
        coef_effects = 100 * (np.exp(glm.params) - 1)
        print(coef_effects)
    except Exception as e:
        print(f"Error fitting GLM: {e}")
    
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()
