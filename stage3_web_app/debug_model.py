#!/usr/bin/env python3
"""
Simplified debug app to test model selection
"""

import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(__file__))

from model_loader import list_model_files, load_model_by_name

st.set_page_config(page_title="Model Selection Debug", layout="wide")
st.title("🔧 Model Selection Debug Tool")

st.markdown("### Debug Information")

# Test model listing
st.markdown("#### 1. Testing model file listing...")
try:
    model_files = list_model_files()
    st.success(f"✅ Found {len(model_files)} model files: {model_files}")
    
    # Show file system info
    import os
    models_dir = "saved_models"
    st.info(f"Looking in directory: {os.path.abspath(models_dir)}")
    st.info(f"Directory exists: {os.path.exists(models_dir)}")
    
    if os.path.exists(models_dir):
        all_files = os.listdir(models_dir)
        st.info(f"All files in directory: {all_files}")
        pkl_files = [f for f in all_files if f.endswith('.pkl')]
        st.info(f"PKL files found: {pkl_files}")
    
except Exception as e:
    st.error(f"❌ Error listing model files: {e}")
    import traceback
    st.code(traceback.format_exc())

# Test model selection UI
st.markdown("#### 2. Testing model selection UI...")

if 'model_files' in locals() and model_files:
    st.markdown("**Available models:**")
    for i, model_file in enumerate(model_files):
        st.write(f"{i+1}. {model_file}")
    
    # Model selection dropdown
    model_choice = st.selectbox(
        "Select a model:",
        ["(none)"] + model_files,
        key="debug_model_selector"
    )
    
    st.write(f"Selected model: `{model_choice}`")
    
    # Test model loading
    if model_choice and model_choice != "(none)":
        st.markdown("#### 3. Testing model loading...")
        
        if st.button("🔍 Test Load Model", key="test_load"):
            try:
                with st.spinner("Loading model..."):
                    model, scaler, metadata = load_model_by_name(model_choice)
                
                st.success("✅ Model loaded successfully!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write("**Model:**")
                    st.write(f"Type: {type(model)}")
                    st.write(f"Details: {str(model)[:100]}...")
                
                with col2:
                    st.write("**Scaler:**")
                    st.write(f"Type: {type(scaler)}")
                    if scaler:
                        st.write(f"Features: {getattr(scaler, 'feature_names_in_', 'N/A')}")
                
                with col3:
                    st.write("**Metadata:**")
                    if metadata:
                        for key, value in metadata.items():
                            st.write(f"{key}: {value}")
                    else:
                        st.write("No metadata available")
                        
            except Exception as e:
                st.error(f"❌ Error loading model: {e}")
                import traceback
                st.code(traceback.format_exc())
    else:
        st.info("👆 Select a model above to test loading")
else:
    st.warning("⚠️ No model files found - cannot test model selection")

# Debug session state
st.markdown("---")
st.markdown("#### Session State Debug")
st.write("Current session state keys:", list(st.session_state.keys()))

if st.button("🧹 Clear Session State"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.success("✅ Session state cleared")
    st.experimental_rerun()