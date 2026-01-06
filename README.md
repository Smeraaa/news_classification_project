# News Article Classification Project

A machine learning pipeline that classifies news articles into categories using TF-IDF vectorization and logistic regression. The project is implemented using Python scripts only (no Jupyter notebooks) and can be run entirely from the terminal.

## Project Overview

This project implements a complete machine learning pipeline for news article classification. Given a dataset containing news articles and their categories, the system preprocesses the text data, extracts features using TF-IDF, trains a classification model, and evaluates its performance. The solution follows best practices for ML project structure and is fully runnable from the command line.

## Dataset Source

This project uses the **AG News Dataset**, which is a publicly available dataset for news classification. The dataset contains news articles from 4 categories:
- World
- Sports
- Business
- Sci/Tech

**Dataset Source**: 
- **Primary Link**: https://huggingface.co/datasets/ag_news
- **Alternative**: The dataset can also be accessed via the Hugging Face `datasets` library in Python using `load_dataset("ag_news")`
- **Dataset Information**: Contains 120,000 training samples with 4 categories (World, Sports, Business, Sci/Tech)

The dataset is automatically downloaded when you run the pipeline for the first time. The raw data is saved to `data/raw/ag_news.csv`.

## Folder Structure

```
news_classification_project/
│
├── data/
│   ├── raw/              # Raw dataset files
│   └── processed/        # Preprocessed data
│
├── src/
│   ├── config.py         # Configuration settings and paths
│   ├── data_preprocessing.py    # Data loading and cleaning
│   ├── feature_engineering.py   # TF-IDF vectorization
│   ├── train.py          # Model training
│   └── evaluate.py       # Model evaluation
│
├── models/
│   └── news_classifier.pkl      # Trained model (includes vectorizer)
│
├── results/
│   └── metrics.txt       # Evaluation metrics and confusion matrix
│
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
└── main.py              # Main entry point
```

### Folder Structure Explanation

- **data/raw/**: Contains the original dataset downloaded from the source
- **data/processed/**: Contains cleaned and preprocessed text data ready for feature engineering
- **src/**: Contains all Python modules implementing the ML pipeline
  - `config.py`: Centralized configuration for paths, hyperparameters, and model settings
  - `data_preprocessing.py`: Handles data loading, missing value treatment, and text cleaning
  - `feature_engineering.py`: Converts text to numerical features using TF-IDF
  - `train.py`: Trains the classification model and saves it
  - `evaluate.py`: Evaluates model performance and saves metrics
- **models/**: Stores trained models and vectorizers as pickle files
- **results/**: Contains evaluation results including accuracy and confusion matrix
- **main.py**: Entry point that orchestrates the entire pipeline

## Steps to Run the Project

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone or download the project** to your local machine

2. **Navigate to the project directory**:
   ```bash
   cd news_classification_project
   ```

3. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Pipeline

Simply run the main script from the terminal:

```bash
python main.py
```

This will execute the complete pipeline:
1. **Data Preprocessing**: Downloads (if needed) and loads the dataset, handles missing values, and cleans text data
2. **Feature Engineering**: Converts text to TF-IDF features
3. **Model Training**: Trains a logistic regression model and saves it
4. **Model Evaluation**: Evaluates the model and saves metrics to `results/metrics.txt`

The final accuracy will be printed to the console, and detailed metrics will be saved in `results/metrics.txt`.

### Running Individual Components

You can also run individual components separately:

```bash
# Data preprocessing only
python src/data_preprocessing.py

# Feature engineering only
python src/feature_engineering.py

# Model training only
python src/train.py

# Model evaluation only
python src/evaluate.py
```

## Model Used

The project uses **Logistic Regression** as the classification model. The model configuration can be changed in `src/config.py` by modifying the `MODEL_TYPE` parameter. Supported options are:
- `"logistic_regression"` (default)
- `"naive_bayes"`
- `"svm"`

### Feature Engineering

Text is converted to numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization with the following parameters:
- Maximum features: 5000
- N-gram range: (1, 2) - includes unigrams and bigrams
- Minimum document frequency: 2
- Maximum document frequency: 0.95
- English stopwords removal

### Training Configuration

- Train-test split: 80% training, 20% testing
- Random state: 42 (for reproducibility)
- Stratified splitting to maintain class distribution

## Final Result Summary

The model achieves high accuracy on the AG News dataset. Typical results:
- **Test Accuracy**: ~90-92%
- **Categories**: 4 classes (World, Sports, Business, Sci/Tech)
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix

Detailed results including per-class metrics and confusion matrix are saved in `results/metrics.txt`.

## GitHub Repository Link

[Your GitHub repository link here]

## Video Explanation Link

[Your video explanation link here]

## Key Features

- ✅ Pure Python implementation (no Jupyter notebooks)
- ✅ Modular and maintainable code structure
- ✅ Automatic dataset download
- ✅ Comprehensive text preprocessing
- ✅ TF-IDF feature extraction
- ✅ Model persistence (saved models can be reused)
- ✅ Detailed evaluation metrics
- ✅ Fully runnable from terminal
- ✅ Well-documented code

## Dependencies

- `pandas`: Data manipulation and analysis
- `scikit-learn`: Machine learning algorithms and utilities
- `numpy`: Numerical computing
- `datasets`: Hugging Face datasets library for easy dataset access

## Notes

- The dataset is automatically downloaded on first run
- All models and vectorizers are saved as pickle files for reuse
- The pipeline is designed to be reproducible (fixed random state)
- Configuration can be easily modified in `src/config.py`

## License

This project is created for educational purposes as part of an AI/ML assignment.

