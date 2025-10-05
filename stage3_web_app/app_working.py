#!/usr/bin/env python3
"""
Simple working version of the multimessenger analysis app
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
from datetime import datetime
from model_loader import list_model_files, load_model_by_name
from inference import predict_df

# Page configuration
st.set_page_config(
    page_title="Multimessenger AI Analysis",
    page_icon="🌌",
    layout="wide"
)

st.title("🌌 Multimessenger AI Analysis Platform")
st.markdown("*AI-powered detection and analysis of multimessenger astronomical events*")

# Sidebar for controls
st.sidebar.header("🎛️ Analysis Controls")

# Model selection
st.sidebar.subheader("🤖 Select Model")
model_files = list_model_files()

if model_files:
    model_choice = st.sidebar.selectbox(
        "Choose trained model:",
        model_files,
        key="model_selector"
    )
else:
    model_choice = None
    st.sidebar.warning("No models found")

# Initialize session state
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'results' not in st.session_state:
    st.session_state.results = None

# Load model
model = None
scaler = None
metadata = None

if model_choice:
    try:
        model, scaler, metadata = load_model_by_name(model_choice)
        st.sidebar.success(f"✅ Model loaded: {model_choice}")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        model = None
else:
    st.sidebar.info("Please select a model")

# Analysis threshold
threshold = st.sidebar.slider("🎯 Association Threshold", 0.0, 1.0, 0.5, 0.05)

# Data input section
st.header("📊 Data Input")

# Data input method selection
input_method = st.radio(
    "Choose data input method:",
    ["📁 Load sample/demo data", "📂 Upload CSV file"],
    horizontal=True
)

df = None

if input_method == "📁 Load sample/demo data":
    # Demo data generation
    if st.button("🎲 Generate Demo Data", type="primary"):
        # Generate sample data
        np.random.seed(42)
        n_pairs = 100
        
        data = {
            'gamma_signal_strength': np.random.exponential(2, n_pairs),
            'neutrino_signal_strength': np.random.exponential(1.5, n_pairs),
            'position_error_gamma': np.random.exponential(0.5, n_pairs),
            'position_error_neutrino': np.random.exponential(0.8, n_pairs),
            'time_difference': np.abs(np.random.normal(0, 1000, n_pairs)),
            'angular_separation': np.random.exponential(1.0, n_pairs)
        }
        
        df = pd.DataFrame(data)
        st.session_state.current_data = df
        st.success(f"✅ Generated {len(df)} sample pairs")

elif input_method == "📂 Upload CSV file":
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file with multimessenger event pairs"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.current_data = df
            st.success(f"✅ Loaded {len(df)} rows from {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error reading file: {e}")

# Use data from session state if available
if st.session_state.current_data is not None:
    df = st.session_state.current_data

# Display data preview
if df is not None:
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(), use_container_width=True)
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Pairs", len(df))
    with col2:
        st.metric("Features", len(df.columns))
    with col3:
        st.metric("Mean Signal Strength", f"{df['gamma_signal_strength'].mean():.2f}")
    with col4:
        st.metric("Mean Time Diff", f"{df['time_difference'].mean():.0f}s")

# Analysis section
st.header("🔬 AI Analysis")

if df is not None and model is not None:
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        with st.spinner("Running AI analysis..."):
            try:
                # Run prediction
                results = predict_df(df, model, scaler, threshold)
                st.session_state.results = results
                
                # Display results
                if results is not None:
                    st.success("✅ Analysis completed successfully!")
                    
                    # Results summary
                    st.subheader("📈 Results Summary")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        associated = (results['probability'] > threshold).sum()
                        st.metric("Associated Pairs", associated)
                    
                    with col2:
                        max_prob = results['probability'].max()
                        st.metric("Max Probability", f"{max_prob:.3f}")
                    
                    with col3:
                        avg_prob = results['probability'].mean()
                        st.metric("Average Probability", f"{avg_prob:.3f}")
                    
                    # Detailed results
                    st.subheader("📊 Detailed Results")
                    
                    # Add association status
                    results_display = results.copy()
                    results_display['Associated'] = results_display['probability'] > threshold
                    results_display['Associated'] = results_display['Associated'].map({True: '✅ Yes', False: '❌ No'})
                    
                    st.dataframe(results_display, use_container_width=True)
                    
                    # Visualization
                    st.subheader("📈 Probability Distribution")
                    
                    fig = px.histogram(
                        results, 
                        x='probability',
                        nbins=20,
                        title="Distribution of Association Probabilities",
                        labels={'probability': 'Association Probability', 'count': 'Number of Pairs'}
                    )
                    fig.add_vline(x=threshold, line_dash="dash", line_color="red", 
                                annotation_text=f"Threshold ({threshold})")
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error("Analysis failed - no results returned")
                    
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                
elif df is None:
    st.info("👆 Please load data first")
elif model is None:
    st.info("👆 Please select a model first")

# Display previous results if available
if st.session_state.results is not None and df is not None:
    st.write("---")
    st.subheader("📋 Previous Results")
    
    results = st.session_state.results
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        associated = (results['probability'] > threshold).sum()
        st.metric("Associated Pairs", associated)
    
    with col2:
        max_prob = results['probability'].max()
        st.metric("Max Probability", f"{max_prob:.3f}")
    
    with col3:
        avg_prob = results['probability'].mean()
        st.metric("Average Probability", f"{avg_prob:.3f}")

# Footer
st.write("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
    🌌 Multimessenger AI Analysis Platform | Powered by Machine Learning
    </div>
    """, 
    unsafe_allow_html=True
)