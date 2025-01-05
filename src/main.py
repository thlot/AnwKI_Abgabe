#%%  # This marks a Jupyter cell
import logging
from preprocessing import preprocess_text
from model import create_vectorizer, train_model
from utils import load_data, save_predictions, create_train_val_split
from tqdm import tqdm

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
    
    # Create and fit vectorizer
    vectorizer = create_vectorizer()
    X_train = vectorizer.fit_transform(train_df['processed_text'])
    y_train = train_df['label']
    
    # Split into train/validation sets
    X_train_split, X_val, y_train_split, y_val = create_train_val_split(
        X_train, y_train
    )
    
    # Train model
    logging.info("Training model...")
    model = train_model(X_train_split, y_train_split, X_val, y_val)
    
    # Generate predictions
    logging.info("Generating predictions...")
    X_test = vectorizer.transform(test_df['processed_text'])
    predictions = model.predict(X_test)
    
    # Save predictions
    save_predictions(test_df, predictions)
    
    logging.info("Pipeline completed")

if __name__ == "__main__":
    main()