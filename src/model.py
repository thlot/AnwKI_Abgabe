# src/model.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, classification_report
import logging
import numpy as np

def create_vectorizer():
    """Create an advanced TF-IDF vectorizer with character n-grams."""
    return TfidfVectorizer(
        max_features=50000,  # Increased from 10000
        ngram_range=(1, 3),  # Include bigrams and trigrams
        min_df=2,
        max_df=0.95,
        analyzer='char_wb',  # Character n-grams including word boundaries
        strip_accents='unicode',
        sublinear_tf=True  # Apply sublinear scaling for term frequencies
    )

def create_model():
    """Create an ensemble of models for better performance."""
    # SVM with optimized parameters (based on V04 lecture slides)
    svm = LinearSVC(
        C=1.0,
        class_weight='balanced',
        dual=False,
        max_iter=2000
    )
    
    # Random Forest as complementary model
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=32,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    
    # Return SVM as primary model (faster and usually better for text)
    return svm

def train_model(X_train, y_train, X_val=None, y_val=None):
    """Train model with grid search for optimal parameters."""
    # Create pipeline
    pipeline = Pipeline([
        ('vectorizer', create_vectorizer()),
        ('classifier', create_model())
    ])
    
    # Define parameter grid for grid search
    param_grid = {
        'vectorizer__max_features': [30000, 50000],
        'vectorizer__ngram_range': [(1, 2), (1, 3)],
        'classifier__C': [0.1, 1.0, 10.0]
    }
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        n_jobs=-1,
        scoring='f1_weighted',
        verbose=1
    )
    
    # Fit the model
    logging.info("Starting grid search...")
    grid_search.fit(X_train, y_train)
    
    # Log best parameters and score
    logging.info(f"Best parameters: {grid_search.best_params_}")
    logging.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Evaluate on validation set if provided
    if X_val is not None and y_val is not None:
        val_pred = grid_search.predict(X_val)
        val_f1 = f1_score(y_val, val_pred, average='weighted')
        logging.info(f"Validation F1 Score: {val_f1:.4f}")
        logging.info("\nClassification Report:\n" + classification_report(y_val, val_pred))
    
    return grid_search.best_estimator_

def make_prediction(model, X_test):
    """Make predictions with additional uncertainty handling."""
    try:
        predictions = model.predict(X_test)
        # Add confidence check (if model supports predict_proba)
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(X_test)
            # For low confidence predictions, use fallback strategy
            confidence = np.max(probas, axis=1)
            low_confidence = confidence < 0.5
            # Default to 'neither' for very uncertain predictions
            predictions[low_confidence] = 2
        return predictions
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        # Fallback to safe prediction
        return np.full(X_test.shape[0], 2)  # Default to 'neither'