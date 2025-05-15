#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Baseline model for structural break detection.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report

def load_data():
    """Load training and test data."""
    train_df = pd.read_csv('../data/train.csv')
    test_df = pd.read_csv('../data/test.csv')
    return train_df, test_df

def create_features(df):
    """Create features for the model."""
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Rolling statistics
    df['rolling_mean_3'] = df['value'].rolling(window=3).mean()
    df['rolling_std_3'] = df['value'].rolling(window=3).std()
    
    # Lag features
    df['lag_1'] = df['value'].shift(1)
    df['lag_2'] = df['value'].shift(2)
    
    # Differences
    df['diff_1'] = df['value'].diff(1)
    df['diff_2'] = df['value'].diff(2)
    
    # Fill NaN values
    df = df.fillna(method='bfill')
    
    return df

def train_model(train_df):
    """Train a model for structural break detection."""
    # Create features
    train_df = create_features(train_df)
    
    # Define features and target
    features = ['value', 'rolling_mean_3', 'rolling_std_3', 
               'lag_1', 'lag_2', 'diff_1', 'diff_2']
    X = train_df[features]
    y = train_df['has_structural_break']
    
    # Create and train model
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    model.fit(X, y)
    
    # Evaluate on training data
    y_pred = model.predict(X)
    print("Training F1 Score:", f1_score(y, y_pred))
    print(classification_report(y, y_pred))
    
    return model, features

def predict(model, test_df, features):
    """Make predictions on test data."""
    # Create features
    test_df = create_features(test_df)
    
    # Make predictions
    X_test = test_df[features]
    test_df['has_structural_break'] = model.predict(X_test)
    
    return test_df

def main():
    """Main function."""
    # Load data
    train_df, test_df = load_data()
    
    # Train model
    model, features = train_model(train_df)
    
    # Predict on test data
    results = predict(model, test_df, features)
    
    # Save predictions
    submission = results[['timestamp', 'has_structural_break']]
    submission.to_csv('../outputs/submission.csv', index=False)
    print("Predictions saved to '../outputs/submission.csv'")

if __name__ == "__main__":
    main() 