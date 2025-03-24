# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 16:09:08 2025

@author: Kaiwei Luo
"""

# 01_hourly_fire_model_training.py
# Training Random Forest model to predict hourly fire activity
# This script processes fire and weather data to train a model for hourly fire potential

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import xarray as xr

# Read data --------------------------------------------------------------
# Load fire activity data
print("Loading fire activity data...")
active_days_hourly_combo = pd.read_csv("D:/000_collections/222_hourly fire potential/023_outputs/NAfires_active days_hourly burning or not_2017_2023/activedays_hourly_1.csv")
NAfires_daily_combo = pd.read_csv("D:/000_collections/222_hourly fire potential/023_outputs/NAfires_daily_combo/NAfires_daily_combo.csv")

# Data preprocessing -----------------------------------------------------
# Round coordinates for consistent matching
active_days_hourly_combo['lat'] = active_days_hourly_combo['lat'].round(4)
active_days_hourly_combo['long'] = active_days_hourly_combo['long'].round(4)
NAfires_daily_combo['lat'] = NAfires_daily_combo['lat'].round(4)
NAfires_daily_combo['long'] = NAfires_daily_combo['long'].round(4)

# Merge hourly fire activity with daily fire weather data
print("Merging hourly and daily data...")
active_days_hourly_combo_match = active_days_hourly_combo.merge(
    NAfires_daily_combo[['year', 'day', 'lat', 'long', 'BUI', 'DMC', 'DC', 'FWI']],
    on=['year', 'day', 'lat', 'long'],
    how='left'
)

# Remove rows with missing BUI values
active_days_hourly_combo_match = active_days_hourly_combo_match.dropna(subset=['BUI'])

# Extract features for training
training_data = active_days_hourly_combo_match[
    ['biome', 'month'] + 
    ['BUI', 'DMC', 'dailyspanlabel'] + 
    active_days_hourly_combo_match.loc[:, 'rh':'isi'].columns.tolist()
]

# Load biome data for reference
biome_data_path = "D:/000_collections/030_Chapter3/032_Codes from Xie Qian/biome.nc"
biome_data = xr.open_dataset(biome_data_path)['gez_code']

# One-hot encoding for categorical variables ------------------------------
# Define all possible biome and month categories
all_biomes = [11, 12, 13, 16, 21, 22, 23, 24, 25, 31, 32, 33, 34, 35, 41, 42, 43, 50, 90]
all_months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# Initialize OneHotEncoder with predefined categories
encoder = OneHotEncoder(sparse=False, drop=None, categories=[all_biomes, all_months])

# Perform one-hot encoding
onehot_encoded = encoder.fit_transform(training_data[['biome', 'month']])
onehot_columns = encoder.get_feature_names_out(['biome', 'month'])

# Combine encoded features with other features
training_data = pd.concat([
    training_data.drop(['biome', 'month'], axis=1),
    pd.DataFrame(onehot_encoded, columns=onehot_columns, index=training_data.index)
], axis=1)

print(f"Feature columns: {training_data.columns.values}")

# Prepare for model training ---------------------------------------------
# Remove rows with missing values
training_data = training_data.dropna()

# Extract features and target variable
X = training_data.drop(columns=['dailyspanlabel'])
y = training_data['dailyspanlabel']

# Set up cross-validation and random forest model
print("Training Random Forest with cross-validation...")
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1000)
rf = RandomForestClassifier(n_estimators=500, max_features=3, random_state=1000, n_jobs=-1)

# Cross-validation with predictions
y_pred_prob = np.zeros(len(y))
y_pred = []

# Perform cross-validation
for train_idx, test_idx in tqdm(skf.split(X, y), total=3):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Compute sample weights to handle class imbalance
    sample_weights = compute_sample_weight('balanced', y_train)
    
    # Train model
    rf.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Predict probabilities and classes
    y_pred_prob[test_idx] = rf.predict_proba(X_test)[:, 1]
    y_pred.extend(rf.predict(X_test))

# Find optimal threshold -------------------------------------------------
from sklearn.metrics import roc_curve, precision_recall_curve

# Calculate ROC curve
fpr, tpr, roc_thresholds = roc_curve(y, y_pred_prob)

# Calculate Precision-Recall curve
precision, recall, pr_thresholds = precision_recall_curve(y, y_pred_prob)

# Method 1: Youden's J statistic (TPR - FPR)
j_scores = tpr - fpr
optimal_idx_roc = np.argmax(j_scores)
optimal_threshold_roc = roc_thresholds[optimal_idx_roc]

# Method 2: F1 score optimization
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
optimal_idx_pr = np.argmax(f1_scores)
optimal_threshold_pr = pr_thresholds[optimal_idx_pr]

print(f"Optimal Threshold (ROC): {optimal_threshold_roc}")
print(f"Optimal Threshold (Precision-Recall): {optimal_threshold_pr}")

# Convert predictions using optimal threshold
y_pred = (y_pred_prob >= optimal_threshold_pr).astype(int)

# Calculate performance metrics ------------------------------------------
conf_matrix = confusion_matrix(y, y_pred, labels=[0, 1])
roc_auc = roc_auc_score(y.map({0: 0, 1: 1}), y_pred_prob)
accuracy = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred, pos_label=1)
precision = precision_score(y, y_pred, pos_label=1)
recall = recall_score(y, y_pred, pos_label=1)
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])

# Output results
results = {
    'Accuracy': accuracy,
    'F1': f1,
    'Precision': precision,
    'Recall': recall,
    'Specificity': specificity,
    'AUC': roc_auc
}

print("Performance Metrics:")
for metric, value in results.items():
    print(f"  {metric}: {value:.4f}")

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Feature Importance:")
print(feature_importance.head(10))

# Save the model ---------------------------------------------------------
import joblib
joblib.dump(rf, "D:/000_collections/222_hourly fire potential/023_outputs/rf_active_days_hourmodel.pkl")
print("Model saved successfully!")

# Visualize model performance --------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

def plot_model_performance(y_true, y_pred, y_pred_prob, save_path="model_performance.png"):
    """
    Create a comprehensive visualization of model performance
    """
    fig = plt.figure(figsize=(10, 8))
    
    # 1. Confusion Matrix
    plt.subplot(221)
    conf_matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    
    # 2. ROC Curve
    plt.subplot(222)
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    
    # 3. Precision-Recall Curve
    plt.subplot(223)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    plt.plot(recall, precision, label=f'PR Curve (F1 = {f1:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    
    # 4. Performance Metrics
    plt.subplot(224)
    metrics = pd.Series(results)
    metrics.plot(kind='bar', color='skyblue')
    plt.title('Performance Metrics')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# Plot and save performance visualization
plot_model_performance(y, y_pred, y_pred_prob, save_path="model_performance.png")

# Plot feature importance
plt.figure(figsize=(10, 6))
top_features = feature_importance.head(10)
sns.barplot(x='importance', y='feature', data=top_features)
plt.title('Top 10 Feature Importance')
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.close()

print("Analysis complete. Visualizations saved.")