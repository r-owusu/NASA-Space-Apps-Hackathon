#!/usr/bin/env python3
"""
Quick test to verify analysis functionality works independently
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import pandas as pd
from model_loader import load_model_by_name, list_model_files
from inference import predict_df

def test_analysis():
    print("Testing analysis functionality...")
    
    # Test 1: Load model
    print("\n1. Testing model loading...")
    try:
        models = list_model_files()
        print(f"Available models: {models}")
        
        if models:
            model, scaler, metadata = load_model_by_name(models[0])
            print(f"Model loaded: {type(model)}")
            print(f"Scaler loaded: {type(scaler)}")
            print(f"Metadata: {metadata}")
        else:
            print("No models found!")
            return False
    except Exception as e:
        print(f"Model loading failed: {e}")
        return False
    
    # Test 2: Create sample data
    print("\n2. Creating sample data...")
    try:
        data = {
            'dt': [0.1, 0.5, 2.0, 0.2],
            'dtheta': [0.01, 0.05, 0.1, 0.02],
            'strength_ratio': [1.5, 2.0, 0.8, 3.0],
            'm1': ['GW', 'Gamma', 'Neutrino', 'GW'],
            'm2': ['Gamma', 'Neutrino', 'Optical', 'Optical']
        }
        df = pd.DataFrame(data)
        print(f"Sample data shape: {df.shape}")
        print(df.head())
    except Exception as e:
        print(f"Sample data creation failed: {e}")
        return False
    
    # Test 3: Run prediction
    print("\n3. Testing prediction...")
    try:
        df_pred = predict_df(df, model, scaler=scaler, threshold=0.5)
        print(f"Prediction successful! Result shape: {df_pred.shape}")
        print("Columns:", df_pred.columns.tolist())
        print("\nPrediction results:")
        print(df_pred[['dt', 'dtheta', 'pred_prob', 'pred_label']].head())
        
        # Check if we have meaningful results
        if 'pred_prob' in df_pred.columns:
            print(f"\nProbability range: {df_pred['pred_prob'].min():.3f} to {df_pred['pred_prob'].max():.3f}")
            print(f"Positive predictions: {df_pred['pred_label'].sum()}")
            return True
        else:
            print("Missing prediction columns!")
            return False
            
    except Exception as e:
        print(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_analysis()
    print(f"\n{'='*50}")
    print(f"Test result: {'PASS' if success else 'FAIL'}")
    print(f"{'='*50}")