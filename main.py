"""
Main Entry Point
Runs the complete ML pipeline: preprocessing, training, and evaluation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_preprocessing import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model


def main():
    """
    Execute the complete machine learning pipeline.
    """
    print("\n" + "=" * 70)
    print("NEWS ARTICLE CLASSIFICATION - MACHINE LEARNING PIPELINE")
    print("=" * 70 + "\n")
    
    try:
        # Step 1: Preprocess the data
        print("\n" + "=" * 70)
        print("STEP 1: DATA PREPROCESSING")
        print("=" * 70)
        preprocess_data()
        
        # Step 2: Train the model
        print("\n" + "=" * 70)
        print("STEP 2: MODEL TRAINING")
        print("=" * 70)
        train_model()
        
        # Step 3: Evaluate the model
        print("\n" + "=" * 70)
        print("STEP 3: MODEL EVALUATION")
        print("=" * 70)
        accuracy, cm, class_report = evaluate_model()
        
        # Final Summary
        print("\n" + "=" * 70)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\nFinal Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\nDetailed metrics saved to: results/metrics.txt")
        print(f"Trained model (with vectorizer) saved to: models/news_classifier.pkl")
        print("\n" + "=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

