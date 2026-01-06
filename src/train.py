"""
Model Training Module
Trains a classification model and saves it to the models folder.
"""

import pickle
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from src.config import (
    PROCESSED_DATA_FILE,
    MODEL_FILE,
    TEST_SIZE,
    RANDOM_STATE,
    MODEL_TYPE
)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder

from src.feature_engineering import create_tfidf_features


def train_model():
    """
    Train the classification model and save it.
    """
    print("=" * 50)
    print("Starting Model Training")
    print("=" * 50)
    
    # Create features
    print("\n1. Creating features...")
    X, y, vectorizer = create_tfidf_features()
    
    # Encode labels
    print("\n2. Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"   Number of classes: {len(label_encoder.classes_)}")
    print(f"   Classes: {label_encoder.classes_.tolist()}")
    
    # Split data
    print(f"\n3. Splitting data (test size: {TEST_SIZE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    
    # Initialize model
    print(f"\n4. Initializing {MODEL_TYPE} model...")
    if MODEL_TYPE == "logistic_regression":
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=1.0)
    elif MODEL_TYPE == "naive_bayes":
        model = MultinomialNB(alpha=1.0)
    elif MODEL_TYPE == "svm":
        model = LinearSVC(random_state=RANDOM_STATE, max_iter=1000, C=1.0)
    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")
    
    # Train model
    print("\n5. Training model...")
    model.fit(X_train, y_train)
    print("   Model training completed!")
    
    # Calculate training accuracy
    train_accuracy = model.score(X_train, y_train)
    print(f"   Training accuracy: {train_accuracy:.4f}")
    
    # Save everything together in one file
    print(f"\n6. Saving model to {MODEL_FILE}...")
    model_data = {
        'model': model,
        'vectorizer': vectorizer,  # Need vectorizer for predictions
        'label_encoder': label_encoder,
        'model_type': MODEL_TYPE
    }
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model_data, f)
    print("   Model saved successfully!")
    
    print("\nModel training completed!")
    
    return model, label_encoder, X_test, y_test


if __name__ == "__main__":
    train_model()

