"""
Data Preprocessing Module
Loads dataset, handles missing values, and cleans text data.
"""

import pandas as pd
import re
import string
from pathlib import Path
import sys

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from src.config import RAW_DATA_FILE, PROCESSED_DATA_FILE, DATASET_NAME

try:
    from datasets import load_dataset
except ImportError:
    print("Warning: datasets library not found. Please install it using: pip install datasets")


def download_ag_news_dataset():
    """
    Download AG News dataset if not already present.
    Returns: DataFrame with 'text' and 'label' columns
    """
    if RAW_DATA_FILE.exists():
        print(f"Dataset already exists at {RAW_DATA_FILE}")
        return pd.read_csv(RAW_DATA_FILE)
    
    print("Downloading AG News dataset...")
    try:
        from datasets import load_dataset
        
        # Load AG News dataset
        dataset = load_dataset(DATASET_NAME, split='train')
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(dataset)
        
        # AG News has 'text' and 'label' columns
        # Label mapping: 1=World, 2=Sports, 3=Business, 4=Sci/Tech
        label_mapping = {1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tech'}
        df['category'] = df['label'].map(label_mapping)
        
        # Keep only text and category columns
        df = df[['text', 'category']].copy()
        df.columns = ['text', 'category']
        
        # Save to raw data directory
        df.to_csv(RAW_DATA_FILE, index=False)
        print(f"Dataset downloaded and saved to {RAW_DATA_FILE}")
        
        return df
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nAlternative: Please download AG News dataset manually and place it at:")
        print(f"{RAW_DATA_FILE}")
        print("The CSV should have 'text' and 'category' columns.")
        raise


# Note: stopwords removal is handled by TF-IDF vectorizer, so this function is not used
# Keeping it here for potential future use
def remove_stopwords(text, stopwords_list):
    """Remove stopwords from text."""
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopwords_list]
    return ' '.join(filtered_words)


def clean_text(text):
    """
    Clean text data:
    - Convert to lowercase
    - Remove special characters and symbols
    - Remove extra whitespace
    """
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters, keep only alphanumeric and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def preprocess_data():
    """
    Main preprocessing function:
    1. Load dataset
    2. Handle missing values
    3. Clean text data
    4. Save processed data
    """
    print("=" * 50)
    print("Starting Data Preprocessing")
    print("=" * 50)
    
    # Load dataset
    print("\n1. Loading dataset...")
    if RAW_DATA_FILE.exists():
        df = pd.read_csv(RAW_DATA_FILE)
        print(f"   Loaded dataset from {RAW_DATA_FILE}")
    else:
        df = download_ag_news_dataset()
    
    print(f"   Dataset shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Check if required columns exist
    if 'text' not in df.columns or 'category' not in df.columns:
        # Try to find columns with similar names
        text_col = None
        category_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'text' in col_lower or 'article' in col_lower or 'description' in col_lower:
                text_col = col
            if 'category' in col_lower or 'label' in col_lower or 'class' in col_lower:
                category_col = col
        
        if text_col and category_col:
            df = df.rename(columns={text_col: 'text', category_col: 'category'})
        else:
            raise ValueError("Dataset must have 'text' and 'category' columns")
    
    # Handle missing values
    print("\n2. Handling missing values...")
    initial_count = len(df)
    df = df.dropna(subset=['text', 'category'])
    removed_count = initial_count - len(df)
    print(f"   Removed {removed_count} rows with missing values")
    print(f"   Remaining rows: {len(df)}")
    
    # Clean text data
    print("\n3. Cleaning text data...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Filter out empty text after cleaning
    df = df[df['cleaned_text'].str.len() > 0]
    print(f"   Final cleaned articles: {len(df)}")
    
    # Display category distribution
    print("\n4. Category distribution:")
    category_counts = df['category'].value_counts()
    for category, count in category_counts.items():
        print(f"   {category}: {count}")
    
    # Save processed data
    print(f"\n5. Saving processed data to {PROCESSED_DATA_FILE}...")
    df[['cleaned_text', 'category']].to_csv(PROCESSED_DATA_FILE, index=False)
    print("   Data preprocessing completed!")
    
    return df[['cleaned_text', 'category']]


if __name__ == "__main__":
    preprocess_data()

