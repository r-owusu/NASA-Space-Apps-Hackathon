#!/usr/bin/env python3
"""
Quick test to check model loading functionality
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from model_loader import list_model_files, load_model_by_name

def test_model_loading():
    print("Testing model loading functionality...")
    
    # Test 1: List model files
    print("\n1. Testing list_model_files()...")
    try:
        model_files = list_model_files()
        print(f"Found model files: {model_files}")
        
        if not model_files:
            print("❌ No model files found!")
            return False
        else:
            print(f"✅ Found {len(model_files)} model files")
    except Exception as e:
        print(f"❌ Error listing model files: {e}")
        return False
    
    # Test 2: Load first model
    print("\n2. Testing load_model_by_name()...")
    try:
        if model_files:
            model_name = model_files[0]
            print(f"Attempting to load: {model_name}")
            
            model, scaler, metadata = load_model_by_name(model_name)
            
            print(f"✅ Model loaded successfully!")
            print(f"   Model type: {type(model)}")
            print(f"   Scaler type: {type(scaler)}")
            print(f"   Metadata: {metadata}")
            
            return True
        else:
            print("❌ No model files to test")
            return False
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    print(f"\n{'='*50}")
    print(f"Model loading test: {'PASS' if success else 'FAIL'}")
    print(f"{'='*50}")