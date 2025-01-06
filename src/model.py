import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
import re
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
import gc
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryEfficientClassifier:
    def __init__(self):
        """Initialize the classifier with memory-efficient components"""
        # Use HashingVectorizer instead of TfidfVectorizer to save memory
        self.vectorizer = HashingVectorizer(
            n_features=2**16,
            ngram_range=(1, 2),
            alternate_sign=False
        )
        
        # Use SGDClassifier for online learning
        self.classifier = SGDClassifier(
            loss='log_loss',
            max_iter=5,
            tol=1e-3,
            n_jobs=1,  # Reduce parallel processing to save memory
            class_weight='balanced'
        )
        
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        """Efficiently preprocess a single text string"""
        try:
            if not isinstance(text, str):
                return ''
            
            # Lowercase
            text = text.lower()
            # Remove special characters
            text = re.sub(r'[^a-z\s]', '', text)
            # Tokenize and remove stopwords
            tokens = word_tokenize(text)
            tokens = [t for t in tokens if t not in self.stop_words]
            return ' '.join(tokens)
        except Exception as e:
            logger.warning(f"Error preprocessing text: {str(e)}")
            return ''

    def data_generator(self, file_path, chunk_size=100):
        """Generator to load and preprocess data in small chunks"""
        try:
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                # Preprocess texts
                chunk['text'] = chunk['text'].apply(self.preprocess_text)
                
                # Convert to features
                X = self.vectorizer.transform(chunk['text'])
                
                if 'label' in chunk.columns:
                    y = chunk['label'].values
                    yield X, y
                else:
                    yield X
                
                # Force garbage collection
                del chunk
                gc.collect()
                
        except Exception as e:
            logger.error(f"Error in data generator: {str(e)}")
            raise

    def train(self, train_file):
        """Train the model using small batches"""
        logger.info("Starting training...")
        try:
            # Process data in small chunks
            for i, (X, y) in enumerate(self.data_generator(train_file)):
                self.classifier.partial_fit(X, y, classes=np.array([0, 1, 2]))
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {(i + 1) * 100} samples")
                
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def predict_and_save(self, test_file, output_file, chunk_size=100):
        """Generate and save predictions in chunks"""
        logger.info("Starting prediction...")
        try:
            # Read the original test file in chunks and keep only necessary columns
            reader = pd.read_csv(test_file, chunksize=chunk_size)
            first_chunk = True
            
            for i, chunk in enumerate(reader):
                # Get predictions for this chunk
                X = next(self.data_generator(test_file, chunk_size=len(chunk)))
                predictions = self.classifier.predict(X)
                
                # Add predictions to chunk
                chunk['label'] = predictions
                
                # Save chunk
                mode = 'w' if first_chunk else 'a'
                header = first_chunk
                chunk.to_csv(output_file, mode=mode, header=header, index=False)
                
                first_chunk = False
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {(i + 1) * chunk_size} predictions")
                
                # Clean up
                del chunk, X, predictions
                gc.collect()
                
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise