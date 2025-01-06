from model import MemoryEfficientClassifier
import logging
import gc
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Initialize classifier
        classifier = MemoryEfficientClassifier()
        
        # Train model
        logger.info("Training model...")
        classifier.train('train.csv')
        
        # Generate predictions
        logger.info("Generating predictions...")
        classifier.predict_and_save('test_no_labels.csv', 'test_with_label.csv')
        
        # Final cleanup
        gc.collect()
        
        logger.info("Process completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()