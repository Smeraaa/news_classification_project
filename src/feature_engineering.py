"""
Feature Engineering Module
Converts text into numerical format using TF-IDF vectorization.
"""

import pandas as pd
import pickle
from pathlib import Path
import sys

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from src.config import (
    PROCESSED_DATA_FILE, 
    MAX_FEATURES,
    MIN_DF,
    MAX_DF,
    NGRAM_RANGE
)

from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_features():
    """
    Create TF-IDF features from preprocessed text data.
    Returns: X (features), y (labels), vectorizer
    """
    print("=" * 50)
    print("Starting Feature Engineering")
    print("=" * 50)
    
    # Load processed data
    print(f"\n1. Loading processed data from {PROCESSED_DATA_FILE}...")
    df = pd.read_csv(PROCESSED_DATA_FILE)
    print(f"   Loaded {len(df)} samples")
    
    # Separate features and labels
    X_text = df['cleaned_text'].values
    y = df['category'].values
    
    # Initialize TF-IDF Vectorizer
    print("\n2. Creating TF-IDF vectorizer...")
    print(f"   Max features: {MAX_FEATURES}")
    print(f"   Min document frequency: {MIN_DF}")
    print(f"   Max document frequency: {MAX_DF}")
    print(f"   N-gram range: {NGRAM_RANGE}")
    
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        min_df=MIN_DF,
        max_df=MAX_DF,
        ngram_range=NGRAM_RANGE,
        stop_words='english'
    )
    
    # Transform text to TF-IDF features
    print("\n3. Transforming text to TF-IDF features...")
    X = vectorizer.fit_transform(X_text)
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # Vectorizer will be saved together with the model in train.py
    print("\n4. Vectorizer will be saved with model during training...")
    
    print("\nFeature engineering completed!")
    
    return X, y, vectorizer


if __name__ == "__main__":
    create_tfidf_features()

