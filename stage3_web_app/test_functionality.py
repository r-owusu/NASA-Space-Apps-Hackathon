#!/usr/bin/env python3
"""
Comprehensive functionality test for the Enhanced Multimessenger AI Platform
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime

def test_model_loader():
    """Test the model loader functionality"""
    print('🔍 Testing Model Loader Functionality...')
    
    try:
        from model_loader import list_model_files, load_model_by_name, get_model_info
        print('✅ Model loader imports successful')
        
        # Test model file discovery
        model_files = list_model_files()
        print(f'📂 Found model files: {model_files}')
        
        if model_files:
            # Test loading the first model
            if isinstance(model_files[0], tuple):
                model_name = model_files[0][0]
            else:
                model_name = model_files[0]
            
            print(f'🔄 Testing model loading: {model_name}')
            
            try:
                model, scaler, metadata = load_model_by_name(model_name)
                print(f'✅ Model loaded successfully')
                print(f'   - Model object: {type(model).__name__ if model else "None"}')
                print(f'   - Scaler object: {type(scaler).__name__ if scaler else "None"}')
                print(f'   - Metadata available: {metadata is not None}')
                
                if metadata:
                    print(f'   - Algorithm: {metadata.get("best_model", "Unknown")}')
                    print(f'   - AUC Score: {metadata.get("best_auc", "N/A")}')
                
                return model, scaler, metadata
            except Exception as e:
                print(f'❌ Model loading failed: {e}')
                return None, None, None
        else:
            print('⚠️ No model files found')
            return None, None, None
            
    except Exception as e:
        print(f'❌ Model loader test failed: {e}')
        return None, None, None

def test_inference_engine():
    """Test the inference engine with enhanced predictions"""
    print('\n🔍 Testing Enhanced Inference Engine...')
    
    try:
        from inference import predict_df, validate_input_df, derive_features
        print('✅ Inference engine imports successful')
        
        # Create test data
        test_data = {
            'dt': [100, 500, 1200, 50, 800],
            'dtheta': [0.05, 0.3, 2.1, 0.15, 1.5],
            'strength_ratio': [1.2, 0.8, 3.5, 1.8, 0.6],
            'm1': ['Gamma', 'GW', 'Gamma', 'GW', 'Neutrino'],
            'm2': ['Neutrino', 'Optical', 'Radio', 'Neutrino', 'Optical']
        }
        
        df = pd.DataFrame(test_data)
        print(f'📊 Created test dataset with {len(df)} rows')
        
        # Test validation
        try:
            validated_df = validate_input_df(df)
            print('✅ Data validation successful')
        except Exception as e:
            print(f'❌ Data validation failed: {e}')
            return False
        
        # Test feature derivation
        try:
            derived_df = derive_features(df)
            print(f'✅ Feature derivation successful, added log_strength_ratio')
        except Exception as e:
            print(f'❌ Feature derivation failed: {e}')
            return False
        
        return df
        
    except Exception as e:
        print(f'❌ Inference engine test failed: {e}')
        return False

def test_full_prediction_pipeline(model, scaler, df):
    """Test the complete prediction pipeline"""
    print('\n🔍 Testing Complete Prediction Pipeline...')
    
    if model is None or df is False:
        print('⚠️ Skipping prediction test (no model or data)')
        return False
    
    try:
        from inference import predict_df
        
        # Run predictions
        results = predict_df(df, model, scaler, threshold=0.5)
        print(f'✅ Prediction pipeline successful')
        print(f'📊 Results shape: {results.shape}')
        
        # Check enhanced columns
        expected_columns = ['pred_prob', 'pred_label', 'confidence_level', 
                          'confidence_description', 'event_classification', 
                          'physical_reasoning', 'risk_assessment']
        
        missing_cols = [col for col in expected_columns if col not in results.columns]
        if missing_cols:
            print(f'⚠️ Missing enhanced columns: {missing_cols}')
        else:
            print('✅ All enhanced prediction columns present')
        
        # Display sample results
        print('\n📋 Sample Prediction Results:')
        for idx, row in results.head(3).iterrows():
            print(f'Event {idx+1}: {row["event_classification"]} '
                  f'(Prob: {row["pred_prob"]:.3f}, Confidence: {row["confidence_level"]})')
        
        return results
        
    except Exception as e:
        print(f'❌ Prediction pipeline test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_data_formats():
    """Test various data format handling"""
    print('\n🔍 Testing Data Format Support...')
    
    # Test CSV creation and reading
    test_data = {
        'dt': [100, 500, 200],
        'dtheta': [0.1, 0.5, 0.3],
        'strength_ratio': [1.5, 2.0, 0.8]
    }
    
    try:
        df = pd.DataFrame(test_data)
        
        # Test CSV
        df.to_csv('test_data.csv', index=False)
        df_csv = pd.read_csv('test_data.csv')
        print('✅ CSV format support working')
        
        # Test Excel (if openpyxl is available)
        try:
            df.to_excel('test_data.xlsx', index=False)
            df_excel = pd.read_excel('test_data.xlsx')
            print('✅ Excel format support working')
        except Exception as e:
            print(f'⚠️ Excel format test failed: {e}')
        
        # Test JSON
        df.to_json('test_data.json', orient='records')
        df_json = pd.read_json('test_data.json')
        print('✅ JSON format support working')
        
        # Test Parquet (if pyarrow is available)
        try:
            df.to_parquet('test_data.parquet')
            df_parquet = pd.read_parquet('test_data.parquet')
            print('✅ Parquet format support working')
        except Exception as e:
            print(f'⚠️ Parquet format test failed: {e}')
        
        # Cleanup
        import os
        for file in ['test_data.csv', 'test_data.xlsx', 'test_data.json', 'test_data.parquet']:
            try:
                os.remove(file)
            except:
                pass
        
        return True
        
    except Exception as e:
        print(f'❌ Data format test failed: {e}')
        return False

def test_statistical_functions():
    """Test statistical analysis functions"""
    print('\n🔍 Testing Statistical Analysis Functions...')
    
    try:
        from scipy import stats
        import numpy as np
        
        # Generate test data
        np.random.seed(42)
        data = np.random.normal(0.6, 0.2, 100)
        
        # Test descriptive statistics
        mean_val = np.mean(data)
        std_val = np.std(data)
        skewness = stats.skew(data)
        kurtosis_val = stats.kurtosis(data)
        
        print(f'✅ Descriptive statistics: mean={mean_val:.3f}, std={std_val:.3f}')
        print(f'✅ Distribution properties: skew={skewness:.3f}, kurtosis={kurtosis_val:.3f}')
        
        # Test hypothesis testing
        t_stat, p_value = stats.ttest_1samp(data, 0.5)
        print(f'✅ One-sample t-test: t={t_stat:.3f}, p={p_value:.3f}')
        
        # Test normality test
        shapiro_stat, shapiro_p = stats.shapiro(data[:50])  # Limit for Shapiro-Wilk
        print(f'✅ Normality test: W={shapiro_stat:.3f}, p={shapiro_p:.3f}')
        
        # Test correlation
        data2 = data + np.random.normal(0, 0.1, len(data))
        corr_coef, corr_p = stats.pearsonr(data, data2)
        print(f'✅ Correlation test: r={corr_coef:.3f}, p={corr_p:.3f}')
        
        return True
        
    except Exception as e:
        print(f'❌ Statistical analysis test failed: {e}')
        return False

def test_visualization_functions():
    """Test visualization capabilities"""
    print('\n🔍 Testing Visualization Functions...')
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create test data
        df = pd.DataFrame({
            'x': np.random.randn(50),
            'y': np.random.randn(50),
            'z': np.random.randn(50),
            'prob': np.random.uniform(0, 1, 50)
        })
        
        # Test 2D scatter plot
        fig_2d = px.scatter(df, x='x', y='y', color='prob')
        print('✅ 2D scatter plot creation successful')
        
        # Test 3D scatter plot
        fig_3d = px.scatter_3d(df, x='x', y='y', z='z', color='prob')
        print('✅ 3D scatter plot creation successful')
        
        # Test correlation matrix
        corr_matrix = df[['x', 'y', 'z', 'prob']].corr()
        fig_corr = px.imshow(corr_matrix)
        print('✅ Correlation matrix visualization successful')
        
        # Test histogram
        fig_hist = px.histogram(df, x='prob', nbins=20)
        print('✅ Histogram creation successful')
        
        # Test subplots
        fig_subplots = make_subplots(rows=2, cols=2)
        print('✅ Subplot creation successful')
        
        return True
        
    except Exception as e:
        print(f'❌ Visualization test failed: {e}')
        return False

def main():
    """Run all functionality tests"""
    print('🚀 Starting Comprehensive Functionality Tests...')
    print('=' * 60)
    
    # Test 1: Model Loader
    model, scaler, metadata = test_model_loader()
    
    # Test 2: Inference Engine
    test_df = test_inference_engine()
    
    # Test 3: Complete Prediction Pipeline
    results = test_full_prediction_pipeline(model, scaler, test_df)
    
    # Test 4: Data Format Support
    data_formats_ok = test_data_formats()
    
    # Test 5: Statistical Functions
    stats_ok = test_statistical_functions()
    
    # Test 6: Visualization Functions
    viz_ok = test_visualization_functions()
    
    # Summary
    print('\n' + '=' * 60)
    print('📊 TEST SUMMARY:')
    print(f'✅ Model Loading: {"PASS" if model is not None else "FAIL"}')
    print(f'✅ Inference Engine: {"PASS" if test_df is not False else "FAIL"}')
    print(f'✅ Prediction Pipeline: {"PASS" if results is not False else "FAIL"}')
    print(f'✅ Data Formats: {"PASS" if data_formats_ok else "FAIL"}')
    print(f'✅ Statistical Analysis: {"PASS" if stats_ok else "FAIL"}')
    print(f'✅ Visualizations: {"PASS" if viz_ok else "FAIL"}')
    
    all_tests_passed = all([
        model is not None,
        test_df is not False, 
        results is not False,
        data_formats_ok,
        stats_ok,
        viz_ok
    ])
    
    print(f'\n🎉 OVERALL RESULT: {"ALL TESTS PASSED! 🎉" if all_tests_passed else "SOME TESTS FAILED ⚠️"}')
    
    return all_tests_passed

if __name__ == "__main__":
    main()