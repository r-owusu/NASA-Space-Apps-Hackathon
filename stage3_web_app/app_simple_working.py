#!/usr/bin/env python3
"""
Simple Working Multimessenger AI Analysis Platform
Reliable model loading and same/different event prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Multimessenger AI",
    page_icon="🌌",
    layout="wide"
)

# Simple, clean CSS
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    
    .header-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .result-success {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .result-different {
        background: linear-gradient(135deg, #ff6b6b, #ffa500);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .info-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    h1, h2, h3 { color: white; }
    .stSelectbox label, .stSlider label, .stRadio label { color: white !important; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-card">
    <h1>🌌 Multimessenger AI Platform</h1>
    <p style="color: rgba(255, 255, 255, 0.9); font-size: 1.1rem;">
        AI-powered Same/Different Event Classification for Multimessenger Astronomy
    </p>
</div>
""", unsafe_allow_html=True)

# Simple model loading
@st.cache_resource
def load_model():
    """Simple model loading with fallback options"""
    try:
        # Try multiple locations
        model_paths = [
            Path(__file__).parent / "saved_models" / "best_model.pkl",
            Path(__file__).parent.parent / "stage2_model_training" / "stage2_outputs" / "saved_models" / "best_model.pkl"
        ]
        
        for model_path in model_paths:
            if model_path.exists():
                st.success(f"✅ Found model at: {model_path}")
                model_obj = joblib.load(model_path)
                
                if isinstance(model_obj, dict):
                    return model_obj.get('model'), model_obj.get('scaler'), model_obj.get('metadata')
                else:
                    return model_obj, None, None
        
        st.error("❌ No model found in any location")
        return None, None, None
        
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None, None, None

def create_features(df):
    """Create features that match the model training"""
    try:
        # Basic features
        X = df[['dt', 'dtheta', 'strength_ratio']].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Add log transformation
        X['log_strength_ratio'] = np.sign(X['strength_ratio']) * np.log1p(np.abs(X['strength_ratio']))
        
        # Add ALL pair type dummies that the model was trained on
        pair_types = [
            'pair_GW_Neutrino', 'pair_GW_Optical', 'pair_GW_Radio', 
            'pair_Gamma_Neutrino', 'pair_Gamma_Optical', 'pair_Gamma_Radio',
            'pair_Neutrino_Optical', 'pair_Neutrino_Radio', 'pair_Optical_Radio'
        ]
        
        # If pair column exists, use it
        if 'pair' in df.columns:
            pair_dummies = pd.get_dummies(df['pair'], prefix='pair')
            for pair_type in pair_types:
                if pair_type in pair_dummies.columns:
                    X[pair_type] = pair_dummies[pair_type]
                else:
                    X[pair_type] = 0
        else:
            # Default values - all zeros except one
            for pair_type in pair_types:
                X[pair_type] = 0
            X['pair_Gamma_Neutrino'] = 1  # Default assumption
        
        # Ensure correct feature order
        feature_order = ['dt', 'dtheta', 'strength_ratio', 'log_strength_ratio'] + pair_types
        X = X[feature_order]
        
        return X
        
    except Exception as e:
        st.error(f"Feature creation failed: {e}")
        return None

# Load model
model, scaler, metadata = load_model()

# Sidebar
st.sidebar.markdown("## 🎛️ Controls")

if model is not None:
    st.sidebar.markdown("""
    <div class="result-success">
        ✅ <strong>Model Status: Online</strong><br>
        Ready for analysis
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    <div class="result-different">
        ❌ <strong>Model Status: Offline</strong><br>
        Check model files
    </div>
    """, unsafe_allow_html=True)

# Parameters
threshold = st.sidebar.slider("🎯 Classification Threshold", 0.0, 1.0, 0.5, 0.05)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'results' not in st.session_state:
    st.session_state.results = None

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class="info-card">
        <h2>📊 Data Input</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Data input
    input_method = st.radio(
        "Choose data source:",
        ["🎲 Generate Demo Data", "📂 Upload CSV"],
        horizontal=True
    )
    
    if input_method == "🎲 Generate Demo Data":
        col_demo1, col_demo2 = st.columns(2)
        
        with col_demo1:
            n_pairs = st.number_input("Number of pairs:", 10, 200, 50, 10)
        
        with col_demo2:
            event_type = st.selectbox("Event type:", [
                "Gamma-Neutrino", "GW-Optical", "GW-Neutrino", "GW-Radio",
                "Gamma-Optical", "Gamma-Radio", "Neutrino-Optical", 
                "Neutrino-Radio", "Optical-Radio"
            ])
        
        if st.button("🎲 Generate Data", type="primary"):
            np.random.seed(42)
            
            data = {
                'dt': np.abs(np.random.normal(0, 1000, n_pairs)),
                'dtheta': np.random.exponential(1.0, n_pairs),
                'strength_ratio': np.random.exponential(2, n_pairs),
                'event_id': [f"Event_{i+1:03d}" for i in range(n_pairs)]
            }
            
            # Add pair information with proper naming
            pair_mapping = {
                "Gamma-Neutrino": "Gamma_Neutrino",
                "GW-Optical": "GW_Optical", 
                "GW-Neutrino": "GW_Neutrino",
                "GW-Radio": "GW_Radio",
                "Gamma-Optical": "Gamma_Optical",
                "Gamma-Radio": "Gamma_Radio",
                "Neutrino-Optical": "Neutrino_Optical",
                "Neutrino-Radio": "Neutrino_Radio", 
                "Optical-Radio": "Optical_Radio"
            }
            
            data['pair'] = [pair_mapping[event_type]] * n_pairs
            
            df = pd.DataFrame(data)
            st.session_state.data = df
            
            st.markdown(f"""
            <div class="result-success">
                ✅ Generated {len(df)} {event_type} event pairs
            </div>
            """, unsafe_allow_html=True)
    
    elif input_method == "📂 Upload CSV":
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.data = df
                
                st.markdown(f"""
                <div class="result-success">
                    ✅ Loaded {len(df)} rows from file
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"File upload failed: {e}")

with col2:
    st.markdown("""
    <div class="info-card">
        <h3>📈 Statistics</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.data is not None:
        df = st.session_state.data
        st.metric("Event Pairs", len(df))
        st.metric("Features", len(df.columns))
        if 'dt' in df.columns:
            st.metric("Avg Time Diff", f"{df['dt'].mean():.0f}s")

# Data preview
if st.session_state.data is not None:
    df = st.session_state.data
    
    st.markdown("""
    <div class="info-card">
        <h3>📋 Data Preview</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.dataframe(df.head(), width=800)

# Analysis section
st.markdown("""
<div class="info-card">
    <h2>🔬 AI Analysis</h2>
</div>
""", unsafe_allow_html=True)

if st.session_state.data is not None and model is not None:
    if st.button("🚀 Run Same/Different Event Analysis", type="primary", use_container_width=True):
        
        with st.spinner("Running AI analysis..."):
            try:
                # Create features
                X = create_features(df)
                
                if X is not None:
                    # Apply scaling
                    if scaler is not None:
                        X_scaled = scaler.transform(X)
                    else:
                        X_scaled = X.values
                    
                    # Make predictions
                    predictions = model.predict_proba(X_scaled)[:, 1]
                    
                    # Create results
                    results = df.copy()
                    results['probability'] = predictions
                    results['classification'] = results['probability'].apply(
                        lambda x: 'Same Event' if x > threshold else 'Different Events'
                    )
                    results['confidence'] = results['probability'].apply(
                        lambda x: 'High' if abs(x - 0.5) > 0.3 else 'Medium' if abs(x - 0.5) > 0.15 else 'Low'
                    )
                    
                    st.session_state.results = results
                    
                    # Display results
                    same_events = (results['classification'] == 'Same Event').sum()
                    different_events = (results['classification'] == 'Different Events').sum()
                    
                    col_res1, col_res2, col_res3 = st.columns(3)
                    
                    with col_res1:
                        st.markdown(f"""
                        <div class="result-success">
                            <h3 style="margin: 0; color: white;">{same_events}</h3>
                            <p style="margin: 0;">🎯 Same Events</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_res2:
                        st.markdown(f"""
                        <div class="result-different">
                            <h3 style="margin: 0; color: white;">{different_events}</h3>
                            <p style="margin: 0;">🎲 Different Events</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_res3:
                        high_conf = (results['confidence'] == 'High').sum()
                        st.metric("High Confidence", high_conf)
                    
                    # Results table
                    st.markdown("### 📊 Detailed Results")
                    display_results = results[['classification', 'probability', 'confidence']].round(3)
                    st.dataframe(display_results, width=800)
                    
                    # Visualization
                    st.markdown("### 📈 Probability Distribution")
                    
                    fig = px.histogram(
                        results, 
                        x='probability',
                        color='classification',
                        title="Event Classification Distribution",
                        color_discrete_map={
                            'Same Event': '#38ef7d',
                            'Different Events': '#ff6b6b'
                        }
                    )
                    
                    fig.add_vline(x=threshold, line_dash="dash", line_color="yellow")
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                import traceback
                st.text(traceback.format_exc())

elif st.session_state.data is None:
    st.info("👆 Please generate or upload data first")
    
elif model is None:
    st.error("🤖 Model not available - check model files")

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 2rem;">
    <div class="info-card">
        <p style="color: white; margin: 0;">
            🌌 <strong>Multimessenger AI Platform</strong> • Same/Different Event Classification
        </p>
    </div>
</div>
""", unsafe_allow_html=True)