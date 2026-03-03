"""
Inference script for URL maliciousness detection
Load trained model and make predictions on new URLs
"""

import numpy as np
import pandas as pd
from tensorflow import keras
from url_classifier import URLClassifier, AttentionLayer
import sys
import pickle
import os

def load_model_and_classifier(model_path):
    """
    Load trained model and initialize classifier with tokenizers
    
    Args:
        model_path: Path to saved model (.keras or .h5)
    
    Returns:
        classifier: URLClassifier instance with loaded model
    """
    # Initialize classifier
    classifier = URLClassifier()
    
    # Auto-detect model file if not specified
    if not os.path.exists(model_path):
        # Try .keras first (Keras 3.x native format)
        if os.path.exists('url_classifier_model.keras'):
            model_path = 'url_classifier_model.keras'
        elif os.path.exists('url_classifier_model.h5'):
            model_path = 'url_classifier_model.h5'
        else:
            print(f"Error: Model file not found!")
            print("Looking for: url_classifier_model.keras or url_classifier_model.h5")
            return None
    
    # Load the saved model with custom objects
    print(f"Loading model from {model_path}...")
    try:
        # Keras 3.x can load both .keras and .h5 files
        # Custom objects are handled automatically with get_config/from_config
        classifier.model = keras.models.load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTrying with explicit custom objects...")
        try:
            # Fallback: load with explicit custom objects
            classifier.model = keras.models.load_model(
                model_path,
                custom_objects={'AttentionLayer': AttentionLayer}
            )
            print("Model loaded successfully with custom objects!")
        except Exception as e2:
            print(f"Failed to load model: {e2}")
            return None
    
    # Try to load tokenizers if they exist
    char_tokenizer_path = 'char_tokenizer.pkl'
    word_tokenizer_path = 'word_tokenizer.pkl'
    
    if os.path.exists(char_tokenizer_path) and os.path.exists(word_tokenizer_path):
        print("Loading tokenizers...")
        try:
            with open(char_tokenizer_path, 'rb') as f:
                classifier.char_tokenizer = pickle.load(f)
            with open(word_tokenizer_path, 'rb') as f:
                classifier.word_tokenizer = pickle.load(f)
            print("Tokenizers loaded successfully!")
        except Exception as e:
            print(f"Error loading tokenizers: {e}")
            return None
    else:
        print("\nWarning: Tokenizer files not found.")
        print("Looking for: char_tokenizer.pkl and word_tokenizer.pkl")
        print("Please make sure these files are in the same directory.")
        print("\nIf you just trained the model, make sure main.py completed successfully.")
        return None
    
    return classifier


def main():
    """Main function with CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='URL Maliciousness Detector - Inference')
    parser.add_argument('--model', type=str, default='url_classifier_model.keras',
                       help='Path to trained model (.keras or .h5 format)')
    parser.add_argument('--mode', type=str, choices=['single'],
                       default='single', help='Prediction mode')
    parser.add_argument('--url', type=str, help='Single URL to predict')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        if not args.url:
            print("Error: --url required for single mode")
            return
        
        classifier = load_model_and_classifier(args.model)
        if classifier is None:
            print("Error: Could not load model and tokenizers.")
            return
        
        prediction = classifier.predict([args.url])[0][0]
        label = "MALICIOUS" if prediction > 0.5 else "BENIGN"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        print(f"\nURL: {args.url}")
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    # Example usage
    print("URL Maliciousness Detector - Inference Script")
    print("\nUsage examples:")
    print("  Single URL: python predict.py --mode single --url 'http://example.com'")
    
    main()