"""
Model Evaluation Module
Loads the saved model, evaluates using accuracy and confusion matrix,
and saves results to results/metrics.txt
"""

import pickle
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from src.config import MODEL_FILE, METRICS_FILE, PROCESSED_DATA_FILE

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def evaluate_model():
    """
    Load saved model, evaluate it, and save metrics.
    """
    print("=" * 50)
    print("Starting Model Evaluation")
    print("=" * 50)
    
    # Load model
    print(f"\n1. Loading model from {MODEL_FILE}...")
    with open(MODEL_FILE, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    vectorizer = model_data['vectorizer']
    label_encoder = model_data['label_encoder']
    model_type = model_data.get('model_type', 'unknown')
    
    print(f"   Model type: {model_type}")
    print(f"   Classes: {label_encoder.classes_.tolist()}")
    print("   Vectorizer loaded from model file!")
    
    # Load processed data for evaluation
    print("\n3. Loading processed data...")
    df = pd.read_csv(PROCESSED_DATA_FILE)
    y = df['category'].values
    
    # Convert text to features using the saved vectorizer
    print("   Transforming text to features...")
    X = vectorizer.transform(df['cleaned_text'].values)
    print(f"   Feature matrix shape: {X.shape}")
    
    # Encode labels
    y_encoded = label_encoder.transform(y)
    
    # Split data using same parameters as training
    print("\n4. Splitting data for evaluation...")
    from src.config import TEST_SIZE, RANDOM_STATE
    _, X_test, _, y_test = train_test_split(
        X, y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )
    
    # Make predictions
    print("\n5. Making predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    print("\n6. Calculating metrics...")
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    class_report = classification_report(
        y_test, 
        y_pred, 
        target_names=label_encoder.classes_,
        output_dict=True
    )
    
    print(f"\n   Test Accuracy: {accuracy:.4f}")
    print(f"\n   Confusion Matrix:")
    print(cm)
    
    # Save metrics to file
    print(f"\n7. Saving metrics to {METRICS_FILE}...")
    with open(METRICS_FILE, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("NEWS CLASSIFICATION MODEL EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Type: {model_type}\n\n")
        
        f.write(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        
        f.write("Classification Report:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write("-" * 60 + "\n")
        
        for class_name in label_encoder.classes_:
            metrics = class_report[class_name]
            f.write(f"{class_name:<15} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                   f"{metrics['f1-score']:<12.4f} {int(metrics['support']):<10}\n")
        
        f.write("-" * 60 + "\n")
        f.write(f"{'Macro Avg':<15} {class_report['macro avg']['precision']:<12.4f} "
               f"{class_report['macro avg']['recall']:<12.4f} "
               f"{class_report['macro avg']['f1-score']:<12.4f} "
               f"{int(class_report['macro avg']['support']):<10}\n")
        f.write(f"{'Weighted Avg':<15} {class_report['weighted avg']['precision']:<12.4f} "
               f"{class_report['weighted avg']['recall']:<12.4f} "
               f"{class_report['weighted avg']['f1-score']:<12.4f} "
               f"{int(class_report['weighted avg']['support']):<10}\n")
        f.write("-" * 60 + "\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Classes: {', '.join(label_encoder.classes_)}\n\n")
        f.write("Rows = Actual, Columns = Predicted\n\n")
        
        # Format confusion matrix
        f.write(" " * 15)
        for class_name in label_encoder.classes_:
            f.write(f"{class_name[:10]:<12}")
        f.write("\n")
        
        for i, class_name in enumerate(label_encoder.classes_):
            f.write(f"{class_name[:14]:<15}")
            for j in range(len(label_encoder.classes_)):
                f.write(f"{cm[i, j]:<12}")
            f.write("\n")
        
        f.write("\n" + "=" * 60 + "\n")
    
    print("   Metrics saved!")
    print("\nModel evaluation completed!")
    
    return accuracy, cm, class_report


if __name__ == "__main__":
    evaluate_model()

