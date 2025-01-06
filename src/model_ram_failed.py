# src/model.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import logging

def create_vectorizer():
    """Create an advanced TF-IDF vectorizer with optimized parameters."""
    return TfidfVectorizer(
        max_features=100000,  # Increased features
        ngram_range=(1, 4),   # Added up to 4-grams
        min_df=2,
        max_df=0.9,          # Adjusted to remove more common words
        strip_accents='unicode',
        sublinear_tf=True,   # Apply sublinear scaling
        analyzer='char_wb',   # Use char n-grams with word boundaries
        binary=False,        # Use frequency information
    )

def add_statistical_features(X):
    """Add statistical text features."""
    features = []
    for text in X:
        # Calculate statistical features
        length = len(text)
        word_count = len(text.split())
        avg_word_length = length / max(word_count, 1)
        unique_chars = len(set(text))
        
        # Create feature vector
        features.append([
            length,
            word_count,
            avg_word_length,
            unique_chars,
            text.count('!') / max(length, 1),  # Exclamation mark ratio
            text.count('?') / max(length, 1),  # Question mark ratio
            sum(1 for c in text if c.isupper()) / max(length, 1),  # Uppercase ratio
        ])
    return np.array(features)

def create_ensemble():
    """Create an ensemble of different models."""
    svm = LinearSVC(
        class_weight='balanced',
        dual=False,
        max_iter=3000,
        random_state=42
    )
    
    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        max_depth=32,
        n_jobs=-1,
        random_state=42
    )
    
    return [('svm', svm), ('rf', rf)]

def train_model_with_grid_search(X_train, y_train, X_val=None, y_val=None):
    """Train model with advanced features and ensemble methods."""
    # Create and fit vectorizer
    logging.info("Creating and fitting vectorizer...")
    vectorizer = create_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Add statistical features
    logging.info("Adding statistical features...")
    X_train_stats = add_statistical_features(X_train)
    
    # Combine features
    X_train_combined = np.hstack([
        X_train_tfidf.toarray(),
        X_train_stats
    ])
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_combined)
    
    # Create models
    models = create_ensemble()
    
    # Grid search parameters
    param_grid = [{
        'C': [0.1, 1.0, 10.0],
        'tol': [1e-4],
        'class_weight': ['balanced'],
        'max_iter': [3000]
    }]
    
    # Train each model separately for better control
    best_f1 = 0
    best_model = None
    
    for name, model in models:
        logging.info(f"\nTraining {name}...")
        if isinstance(model, LinearSVC):
            grid = GridSearchCV(
                model,
                param_grid,
                cv=5,
                n_jobs=-1,
                scoring='f1_weighted',
                verbose=1
            )
        else:
            # Random Forest parameters
            grid = GridSearchCV(
                model,
                {
                    'n_estimators': [100, 200],
                    'max_depth': [16, 32],
                    'class_weight': ['balanced']
                },
                cv=5,
                n_jobs=-1,
                scoring='f1_weighted',
                verbose=1
            )
        
        grid.fit(X_train_scaled, y_train)
        logging.info(f"Best {name} parameters: {grid.best_params_}")
        logging.info(f"Best {name} cross-validation score: {grid.best_score_:.4f}")
        
        if grid.best_score_ > best_f1:
            best_f1 = grid.best_score_
            best_model = grid.best_estimator_
    
    # Evaluate on validation set if provided
    if X_val is not None and y_val is not None:
        X_val_tfidf = vectorizer.transform(X_val)
        X_val_stats = add_statistical_features(X_val)
        X_val_combined = np.hstack([
            X_val_tfidf.toarray(),
            X_val_stats
        ])
        X_val_scaled = scaler.transform(X_val_combined)
        
        val_pred = best_model.predict(X_val_scaled)
        val_f1 = f1_score(y_val, val_pred, average='weighted')
        logging.info(f"\nValidation F1 Score: {val_f1:.4f}")
        logging.info("\nClassification Report:\n" + classification_report(y_val, val_pred))
    
    return best_model, vectorizer, scaler