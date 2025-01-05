# src/utils.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

def load_data():
    """Load training and test data."""
    try:
        train_df = pd.read_csv('data/train.csv')
        test_df = pd.read_csv('data/test_no_labels.csv')
        logging.info(f"Loaded training data with shape: {train_df.shape}")
        logging.info(f"Loaded test data with shape: {test_df.shape}")
        return train_df, test_df
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def save_predictions(test_df, predictions, output_dir='output'):
    """Save predictions to CSV file."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'test_with_label.csv')
        
        # Ensure predictions maintain the same order as test_df
        test_df['label'] = predictions
        test_df.to_csv(output_path, index=False)
        logging.info(f"Saved predictions to {output_path}")
    except Exception as e:
        logging.error(f"Error saving predictions: {str(e)}")
        raise

def create_train_val_split(X, y, test_size=0.2):
    """Create train/validation split while preserving label distribution."""
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )