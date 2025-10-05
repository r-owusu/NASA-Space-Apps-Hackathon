#!/usr/bin/env python3
"""
Simplified debug version of the app to identify why analysis button isn't working
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from model_loader import list_model_files, load_model_by_name
from inference import predict_df

st.set_page_config(page_title="Debug Multimessenger AI", layout="wide")
st.title("🔍 Debug Multimessenger Analysis")

# Debug information
st.sidebar.header("🐛 Debug Info")
st.sidebar.write(f"Session State Keys: {list(st.session_state.keys())}")

# Model selection
model_files = list_model_files()
st.sidebar.subheader("🤖 Model Selection")
model_choice = st.sidebar.selectbox("Select Model:", ["(none)"] + model_files)

# Initialize variables
model = None
scaler = None

# Model loading
if model_choice and model_choice != "(none)":
    try:
        model, scaler, metadata = load_model_by_name(model_choice)
        st.sidebar.success(f"✅ Model loaded: {model_choice}")
        st.sidebar.write(f"Model type: {type(model)}")
        st.sidebar.write(f"Scaler type: {type(scaler)}")
    except Exception as e:
        st.sidebar.error(f"❌ Model load error: {e}")

# Sample data generation
st.header("📊 Generate Sample Data")
if st.button("Generate Sample Data", key="gen_data"):
    data = {
        'dt': [0.1, 0.5, 2.0, 0.2, 1.5],
        'dtheta': [0.01, 0.05, 0.1, 0.02, 0.08],
        'strength_ratio': [1.5, 2.0, 0.8, 3.0, 1.2],
        'm1': ['GW', 'Gamma', 'Neutrino', 'GW', 'Gamma'],
        'm2': ['Gamma', 'Neutrino', 'Optical', 'Optical', 'Radio']
    }
    df = pd.DataFrame(data)
    st.session_state.data = df
    st.success("✅ Sample data generated!")

# Display data if available
if 'data' in st.session_state:
    df = st.session_state.data
    st.header("📋 Current Data")
    st.dataframe(df)
    st.write(f"Data shape: {df.shape}")
    st.write(f"Columns: {df.columns.tolist()}")
    
    # Initialize session state for analysis
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = None
    
    st.header("🚀 Analysis")
    st.write(f"Model available: {model is not None}")
    st.write(f"Analysis complete: {st.session_state.analysis_complete}")
    st.write(f"Analysis data available: {st.session_state.analysis_data is not None}")
    
    # Analysis button
    if st.button('🔍 Run Analysis', key="analyze_btn", type="primary"):
        st.write("🔍 Button clicked!")
        
        if model is None:
            st.error('❌ Please select a model first')
        else:
            st.write("✅ Model is available, starting analysis...")
            try:
                with st.spinner('🔬 Analyzing...'):
                    st.write("📊 Calling predict_df...")
                    df_pred = predict_df(df, model, scaler=scaler, threshold=0.5)
                    st.write(f"🎯 Prediction completed! Shape: {df_pred.shape}")
                    
                # Store results in session state
                st.session_state.analysis_data = df_pred
                st.session_state.analysis_complete = True
                
                st.success('✅ Analysis completed successfully!')
                st.write(f"✅ Results stored in session state")
                
            except Exception as e:
                st.error(f'❌ Analysis error: {e}')
                import traceback
                st.code(traceback.format_exc())
    
    # Display results if analysis is complete
    if st.session_state.analysis_complete and st.session_state.analysis_data is not None:
        st.header("🎯 Analysis Results")
        df_pred = st.session_state.analysis_data
        
        st.write(f"Results shape: {df_pred.shape}")
        st.write(f"Results columns: {df_pred.columns.tolist()}")
        
        # Show predictions
        if 'pred_prob' in df_pred.columns:
            st.subheader("📈 Predictions")
            st.dataframe(df_pred[['dt', 'dtheta', 'strength_ratio', 'pred_prob', 'pred_label']])
            
            # Basic metrics
            total_events = len(df_pred)
            positive_associations = len(df_pred[df_pred['pred_label'] == 1])
            avg_confidence = df_pred['pred_prob'].mean()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Events", total_events)
            with col2:
                st.metric("Positive Associations", positive_associations)
            with col3:
                st.metric("Avg Confidence", f"{avg_confidence:.3f}")
        else:
            st.error("❌ Prediction columns missing!")
    else:
        st.info("ℹ️ No analysis results available. Click 'Run Analysis' to generate results.")

else:
    st.info("ℹ️ No data available. Click 'Generate Sample Data' to start.")