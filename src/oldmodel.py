# src/model.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
import logging

def create_model():
    """Create and return the classification model."""
    return LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )

def train_model(X_train, y_train, X_val=None, y_val=None):
    """Train the model and optionally evaluate on validation data."""
    model = create_model()
    model.fit(X_train, y_train)
    
    if X_val is not None and y_val is not None:
        val_pred = model.predict(X_val)
        val_f1 = f1_score(y_val, val_pred, average='weighted')
        logging.info(f"Validation F1 Score: {val_f1:.4f}")
        logging.info("\nClassification Report:\n" + classification_report(y_val, val_pred))
    
    return model

def create_vectorizer():
    """Create and return the TF-IDF vectorizer."""
    return TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2
    )