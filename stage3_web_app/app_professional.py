#!/usr/bin/env python3
"""
Professional Multimessenger AI Analysis Platform
Clean, modern interface with reliable model loading and same/different event prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Multimessenger AI Platform",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header styling */
    .main-header {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 1rem;
    }
    
    /* Card styling */
    .info-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.2), rgba(255, 255, 255, 0.1));
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: white;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.2);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        font-weight: 500;
    }
    
    /* Status indicators */
    .status-success {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .status-info {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    /* Prediction results */
    .prediction-same {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
        box-shadow: 0 4px 16px rgba(17, 153, 142, 0.3);
    }
    
    .prediction-different {
        background: linear-gradient(135deg, #ff6b6b, #ffa500);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
        box-shadow: 0 4px 16px rgba(255, 107, 107, 0.3);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Data frame styling */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Fixed model loading functions
def get_model_files():
    """Get list of available model files with proper path handling"""
    current_dir = Path(__file__).parent
    models_dir = current_dir / "saved_models"
    
    # Also check stage2 models directory
    stage2_models_dir = current_dir.parent / "stage2_model_training" / "stage2_outputs" / "saved_models"
    
    model_files = []
    
    # Check local saved_models directory
    if models_dir.exists():
        model_files.extend([f.name for f in models_dir.glob("*.pkl")])
    
    # Check stage2 models directory
    if stage2_models_dir.exists():
        for pkl_file in stage2_models_dir.glob("*.pkl"):
            if pkl_file.name not in model_files:  # Avoid duplicates
                model_files.append(f"stage2/{pkl_file.name}")
    
    return sorted(model_files)

def load_selected_model(model_name):
    """Load model with proper error handling"""
    try:
        current_dir = Path(__file__).parent
        
        if model_name.startswith("stage2/"):
            # Load from stage2 directory
            model_file = model_name.replace("stage2/", "")
            model_path = current_dir.parent / "stage2_model_training" / "stage2_outputs" / "saved_models" / model_file
        else:
            # Load from local directory
            model_path = current_dir / "saved_models" / model_name
        
        if not model_path.exists():
            st.error(f"Model file not found: {model_path}")
            return None, None, None
        
        # Load the model
        model_obj = joblib.load(model_path)
        
        # Handle different model formats
        if isinstance(model_obj, dict):
            return model_obj.get('model'), model_obj.get('scaler'), model_obj.get('metadata')
        else:
            return model_obj, None, None
            
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def predict_events(df, model, scaler=None, threshold=0.5):
    """Enhanced prediction function with same/different event classification"""
    try:
        # Prepare features for prediction
        feature_columns = ['dt', 'dtheta', 'strength_ratio']
        available_columns = [col for col in feature_columns if col in df.columns]
        
        if len(available_columns) < 3:
            st.error(f"Missing required columns. Found: {available_columns}, Need: {feature_columns}")
            return None
        
        X = df[available_columns].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Apply scaling if scaler is available
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Make predictions
        predictions = model.predict_proba(X_scaled)[:, 1]  # Probability of positive class
        
        # Create results dataframe
        results = df.copy()
        results['association_probability'] = predictions
        
        # Enhanced classification logic
        results['event_classification'] = results['association_probability'].apply(
            lambda x: 'Same Astronomical Event' if x > threshold else 'Different Events'
        )
        
        # Add confidence levels
        results['confidence_level'] = results['association_probability'].apply(
            lambda x: 'High' if abs(x - 0.5) > 0.3 else 'Medium' if abs(x - 0.5) > 0.15 else 'Low'
        )
        
        # Add event status
        results['status'] = results.apply(
            lambda row: f"✅ Same Event ({row['confidence_level']} Confidence)" 
            if row['event_classification'] == 'Same Astronomical Event'
            else f"❌ Different Events ({row['confidence_level']} Confidence)", axis=1
        )
        
        return results
        
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

# Header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">🌌 Multimessenger AI Platform</h1>
    <p class="main-subtitle">Professional AI-powered analysis for multimessenger astronomical events</p>
    <p style="color: rgba(255, 255, 255, 0.8);">
        🎯 Same/Different Event Classification • 📊 Advanced Analytics • 🔬 Real-time Predictions
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'analysis_count' not in st.session_state:
    st.session_state.analysis_count = 0

# Sidebar
st.sidebar.markdown("## 🎛️ Control Panel")

# Model selection with fixed loading
st.sidebar.markdown("### 🤖 AI Model Selection")
model_files = get_model_files()

if model_files:
    model_choice = st.sidebar.selectbox(
        "Choose AI model:",
        model_files,
        help="Select a trained machine learning model for analysis"
    )
    
    # Load selected model
    model, scaler, metadata = load_selected_model(model_choice)
    
    if model is not None:
        st.sidebar.markdown(f"""
        <div class="status-success">
            ✅ <strong>Model Loaded Successfully</strong><br>
            File: {model_choice}
        </div>
        """, unsafe_allow_html=True)
        
        # Display model info if available
        if metadata:
            accuracy = metadata.get('best_auc', 'N/A')
            algorithm = metadata.get('best_model', 'Unknown')
            st.sidebar.markdown(f"""
            <div class="info-card">
                <h4 style="color: white; margin-bottom: 1rem;">📊 Model Information</h4>
                <p style="color: white; margin: 0.5rem 0;"><strong>Algorithm:</strong> {algorithm}</p>
                <p style="color: white; margin: 0.5rem 0;"><strong>AUC Score:</strong> {accuracy}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div class="status-warning">
            ⚠️ <strong>Model Loading Failed</strong><br>
            Check model file integrity
        </div>
        """, unsafe_allow_html=True)
        model = None
else:
    st.sidebar.markdown("""
    <div class="status-warning">
        ❌ <strong>No Models Found</strong><br>
        Please ensure model files (.pkl) are in the saved_models directory
    </div>
    """, unsafe_allow_html=True)
    model = None
    model_choice = None

# Analysis parameters
st.sidebar.markdown("### ⚙️ Analysis Settings")
threshold = st.sidebar.slider(
    "🎯 Association Threshold", 
    0.0, 1.0, 0.5, 0.05,
    help="Probability threshold for same event classification"
)

show_advanced = st.sidebar.checkbox("🔬 Advanced Options", False)
if show_advanced:
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.5, 0.3, 0.05)
    show_raw_data = st.sidebar.checkbox("Show Raw Data", True)

# System status
if model is not None:
    st.sidebar.markdown(f"""
    <div class="info-card">
        <h4 style="color: white; margin-bottom: 1rem;">🔥 System Status</h4>
        <p style="color: #38ef7d; margin: 0.3rem 0;">● AI Model: Online</p>
        <p style="color: #38ef7d; margin: 0.3rem 0;">● Analysis Engine: Ready</p>
        <p style="color: #38ef7d; margin: 0.3rem 0;">● Threshold: {threshold:.2f}</p>
        <p style="color: #38ef7d; margin: 0.3rem 0;">● Sessions: {st.session_state.analysis_count}</p>
    </div>
    """, unsafe_allow_html=True)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class="info-card">
        <h2 style="color: white; margin-bottom: 1rem;">📊 Data Input</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Data input options
    input_method = st.radio(
        "Select data source:",
        ["🎲 Generate Demo Data", "📂 Upload CSV File"],
        horizontal=True
    )
    
    df = None
    
    if input_method == "🎲 Generate Demo Data":
        col_demo1, col_demo2 = st.columns(2)
        
        with col_demo1:
            n_pairs = st.number_input("Number of event pairs:", 10, 500, 100, 10)
        
        with col_demo2:
            event_type = st.selectbox("Event type:", ["Gamma-Neutrino", "GW-Optical", "Mixed Events"])
        
        if st.button("🎲 Generate Demo Data", type="primary"):
            with st.spinner("Generating realistic multimessenger event data..."):
                np.random.seed(42)
                
                # Generate realistic demo data
                data = {
                    'dt': np.abs(np.random.normal(0, 1000, n_pairs)),  # time difference (seconds)
                    'dtheta': np.random.exponential(1.0, n_pairs),     # angular separation (degrees)
                    'strength_ratio': np.random.exponential(2, n_pairs), # signal strength ratio
                    'event_id': [f"Event_{i+1:03d}" for i in range(n_pairs)],
                    'detection_time': pd.date_range('2025-01-01', periods=n_pairs, freq='H')
                }
                
                # Add event-specific parameters
                if event_type == "Gamma-Neutrino":
                    data['gamma_energy_gev'] = np.random.lognormal(2, 1, n_pairs)
                    data['neutrino_energy_gev'] = np.random.lognormal(3, 1.5, n_pairs)
                elif event_type == "GW-Optical":
                    data['gw_strain'] = np.random.lognormal(-21, 0.5, n_pairs)
                    data['optical_magnitude'] = np.random.normal(20, 2, n_pairs)
                
                df = pd.DataFrame(data)
                st.session_state.current_data = df
                
                st.markdown(f"""
                <div class="status-success">
                    ✅ <strong>Demo Data Generated!</strong><br>
                    Created {len(df)} {event_type} event pairs
                </div>
                """, unsafe_allow_html=True)
    
    elif input_method == "📂 Upload CSV File":
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload CSV with columns: dt, dtheta, strength_ratio"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.current_data = df
                
                st.markdown(f"""
                <div class="status-success">
                    ✅ <strong>File Uploaded Successfully!</strong><br>
                    Loaded {len(df)} rows from {uploaded_file.name}
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error reading file: {e}")

# Use session state data
if st.session_state.current_data is not None:
    df = st.session_state.current_data

with col2:
    st.markdown("""
    <div class="info-card">
        <h3 style="color: white; margin-bottom: 1rem;">📈 Analytics</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if df is not None:
        # Display metrics
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">Event Pairs</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df.columns)}</div>
            <div class="metric-label">Features</div>
        </div>
        """, unsafe_allow_html=True)
        
        if 'dt' in df.columns:
            avg_time = df['dt'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_time:.0f}s</div>
                <div class="metric-label">Avg Time Diff</div>
            </div>
            """, unsafe_allow_html=True)

# Data preview
if df is not None:
    st.markdown("""
    <div class="info-card">
        <h3 style="color: white; margin-bottom: 1rem;">📋 Data Preview</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.dataframe(df.head(10), use_container_width=True)

# Analysis section
st.markdown("""
<div class="info-card">
    <h2 style="color: white; margin-bottom: 1rem;">🔬 AI Analysis & Event Classification</h2>
</div>
""", unsafe_allow_html=True)

if df is not None and model is not None:
    if st.button("🚀 Run Event Classification Analysis", type="primary", use_container_width=True):
        
        # Progress indicator
        progress_placeholder = st.empty()
        with progress_placeholder:
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
                if i < 50:
                    st.text("🔄 Processing multimessenger data...")
                else:
                    st.text("🧠 Classifying events...")
                import time
                time.sleep(0.02)
        
        # Clear progress
        progress_placeholder.empty()
        
        # Run analysis
        results = predict_events(df, model, scaler, threshold)
        
        if results is not None:
            st.session_state.results = results
            st.session_state.analysis_count += 1
            
            # Success message
            st.markdown("""
            <div class="status-success">
                ✅ <strong>Analysis Complete!</strong> Event classification successful
            </div>
            """, unsafe_allow_html=True)
            
            # Results summary
            same_events = (results['event_classification'] == 'Same Astronomical Event').sum()
            different_events = (results['event_classification'] == 'Different Events').sum()
            high_confidence = (results['confidence_level'] == 'High').sum()
            
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                st.markdown(f"""
                <div class="prediction-same">
                    <div style="font-size: 2rem; font-weight: bold; margin-bottom: 0.5rem;">{same_events}</div>
                    <div>🎯 Same Astronomical Events</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_res2:
                st.markdown(f"""
                <div class="prediction-different">
                    <div style="font-size: 2rem; font-weight: bold; margin-bottom: 0.5rem;">{different_events}</div>
                    <div>🎲 Different Events</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_res3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{high_confidence}</div>
                    <div class="metric-label">High Confidence</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed results
            st.markdown("### 📊 Classification Results")
            
            # Display results table
            display_df = results[['event_classification', 'association_probability', 'confidence_level', 'status']].round(3)
            st.dataframe(display_df, use_container_width=True)
            
            # Visualization
            st.markdown("### 📈 Results Visualization")
            
            # Probability distribution
            fig = px.histogram(
                results, 
                x='association_probability',
                color='event_classification',
                title="Event Classification Distribution",
                labels={'association_probability': 'Association Probability'},
                color_discrete_map={
                    'Same Astronomical Event': '#38ef7d',
                    'Different Events': '#ff6b6b'
                }
            )
            
            fig.add_vline(x=threshold, line_dash="dash", line_color="yellow", 
                         annotation_text=f"Threshold ({threshold})")
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)

elif df is None:
    st.markdown("""
    <div class="status-info">
        👆 <strong>Ready for Data</strong><br>
        Please load your multimessenger event data using the options above
    </div>
    """, unsafe_allow_html=True)

elif model is None:
    st.markdown("""
    <div class="status-warning">
        🤖 <strong>AI Model Required</strong><br>
        Please check that model files exist in the saved_models directory
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="margin-top: 2rem; text-align: center;">
    <div class="info-card">
        <p style="color: rgba(255, 255, 255, 0.8); margin: 0;">
            🌌 <strong>Professional Multimessenger AI Platform</strong> • 
            Same/Different Event Classification • 
            Real-time Analysis
        </p>
    </div>
</div>
""", unsafe_allow_html=True)