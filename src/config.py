"""
Configuration file for news classification project.
Contains all paths and hyperparameters.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model paths
MODELS_DIR = BASE_DIR / "models"
MODEL_FILE = MODELS_DIR / "news_classifier.pkl"
VECTORIZER_FILE = MODELS_DIR / "tfidf_vectorizer.pkl"

# Results paths
RESULTS_DIR = BASE_DIR / "results"
METRICS_FILE = RESULTS_DIR / "metrics.txt"

# Dataset configuration
DATASET_NAME = "ag_news"  # AG News dataset
RAW_DATA_FILE = RAW_DATA_DIR / "ag_news.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "processed_data.csv"

# Model hyperparameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_TYPE = "logistic_regression"  # Options: "logistic_regression", "naive_bayes", "svm"

# Feature engineering parameters
MAX_FEATURES = 5000
MIN_DF = 2
MAX_DF = 0.95
NGRAM_RANGE = (1, 2)  # Unigrams and bigrams

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

