# src/main.py
import logging
import numpy as np
from preprocessing import preprocess_text
from model import train_model_with_grid_search, add_statistical_features
from utils import load_data, save_predictions
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    """Main function to run the hate speech classification pipeline."""
    logging.info("Starting hate speech classification pipeline")
    
    # Load data
    train_df, test_df = load_data()
    
    # Preprocess text
    logging.info("Preprocessing training data...")
    train_df['processed_text'] = train_df['text'].apply(
        lambda x: preprocess_text(x)
    )
    
    logging.info("Preprocessing test data...")
    test_df['processed_text'] = test_df['text'].apply(
        lambda x: preprocess_text(x)
    )
    
    # Split into train and validation sets
    X = train_df['processed_text']
    y = train_df['label']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    # Train model with grid search
    logging.info("Training model...")
    model, vectorizer, scaler = train_model_with_grid_search(X_train, y_train, X_val, y_val)
    
    # Generate predictions
    logging.info("Generating predictions...")
    X_test = test_df['processed_text']
    X_test_tfidf = vectorizer.transform(X_test)
    X_test_stats = add_statistical_features(X_test)
    X_test_combined = np.hstack([
        X_test_tfidf.toarray(),
        X_test_stats
    ])
    X_test_scaled = scaler.transform(X_test_combined)
    predictions = model.predict(X_test_scaled)
    
    # Save predictions
    save_predictions(test_df, predictions)
    
    logging.info("Pipeline completed")

if __name__ == "__main__":
    main()