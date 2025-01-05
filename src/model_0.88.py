# src/model.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, classification_report
import logging

def create_vectorizer():
    """Create an advanced TF-IDF vectorizer with optimal parameters."""
    return TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        strip_accents='unicode',
        sublinear_tf=True
    )

def train_model_with_grid_search(X_train, y_train, X_val=None, y_val=None):
    """Train model with grid search for optimal parameters."""
    # Create and fit vectorizer
    vectorizer = create_vectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    
    # Create base classifier
    classifier = LinearSVC(
        class_weight='balanced',
        dual=False,
        max_iter=2000,
        random_state=42
    )
    
    # Define parameter grid
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'loss': ['squared_hinge'],
        'tol': [1e-4, 1e-3]
    }
    
    # Perform grid search
    grid_search = GridSearchCV(
        classifier,
        param_grid,
        cv=5,
        n_jobs=-1,
        scoring='f1_weighted',
        verbose=1
    )
    
    # Fit the model
    logging.info("Starting grid search...")
    grid_search.fit(X_train_vectorized, y_train)
    
    # Log best parameters and score
    logging.info(f"Best parameters: {grid_search.best_params_}")
    logging.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Evaluate on validation set if provided
    if X_val is not None and y_val is not None:
        X_val_vectorized = vectorizer.transform(X_val)
        val_pred = grid_search.predict(X_val_vectorized)
        val_f1 = f1_score(y_val, val_pred, average='weighted')
        logging.info(f"Validation F1 Score: {val_f1:.4f}")
        logging.info("\nClassification Report:\n" + classification_report(y_val, val_pred))
    
    return grid_search.best_estimator_, vectorizer