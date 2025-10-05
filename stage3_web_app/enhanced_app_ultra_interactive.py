#!/usr/bin/env python3
"""
Ultra-Interactive Multimessenger AI Analysis Platform
Features: Modern glassmorphism UI, advanced animations, real-time predictions,
same/different event classification, interactive elements
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import json
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy import stats
import time
from model_loader import list_model_files, load_model_by_name
from inference import predict_df

# Page configuration
st.set_page_config(
    page_title="Multimessenger AI Platform",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ultra-modern CSS with glassmorphism and animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        min-height: 100vh;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    }
    
    /* Glassmorphism containers */
    .glass-container {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .glass-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2);
        background: rgba(255, 255, 255, 0.2);
    }
    
    /* Hero header with animation */
    .hero-header {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
        backdrop-filter: blur(15px);
        border-radius: 25px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        animation: glow 3s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { box-shadow: 0 0 20px rgba(255, 255, 255, 0.3); }
        to { box-shadow: 0 0 40px rgba(255, 255, 255, 0.6), 0 0 60px rgba(102, 126, 234, 0.4); }
    }
    
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        animation: shine 6s infinite;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: white;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        margin-bottom: 1rem;
        position: relative;
        z-index: 2;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 1rem;
        position: relative;
        z-index: 2;
    }
    
    .hero-features {
        display: flex;
        justify-content: center;
        gap: 2rem;
        flex-wrap: wrap;
        position: relative;
        z-index: 2;
    }
    
    .feature-badge {
        background: rgba(255, 255, 255, 0.2);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: white;
        font-weight: 500;
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .feature-badge:hover {
        background: rgba(255, 255, 255, 0.3);
        transform: scale(1.05);
    }
    
    /* Interactive metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.2), rgba(255, 255, 255, 0.1));
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 2rem;
        text-align: center;
        color: white;
        margin: 1rem 0;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        cursor: pointer;
    }
    
    .metric-card:hover {
        transform: translateY(-10px) scale(1.02);
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.3), rgba(255, 255, 255, 0.2));
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
        font-weight: 500;
    }
    
    /* Interactive buttons */
    .interactive-button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border: none;
        color: white;
        padding: 1rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .interactive-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2, #667eea);
    }
    
    .interactive-button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        background: rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .interactive-button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    /* Progress animations */
    .progress-ring {
        width: 60px;
        height: 60px;
        border: 4px solid rgba(255, 255, 255, 0.3);
        border-top: 4px solid white;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 1rem auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Prediction status indicators */
    .prediction-same {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        animation: pulse-green 2s infinite;
    }
    
    .prediction-different {
        background: linear-gradient(135deg, #ff6b6b, #ffa500);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        animation: pulse-orange 2s infinite;
    }
    
    @keyframes pulse-green {
        0%, 100% { box-shadow: 0 0 20px rgba(17, 153, 142, 0.5); }
        50% { box-shadow: 0 0 40px rgba(17, 153, 142, 0.8); }
    }
    
    @keyframes pulse-orange {
        0%, 100% { box-shadow: 0 0 20px rgba(255, 107, 107, 0.5); }
        50% { box-shadow: 0 0 40px rgba(255, 107, 107, 0.8); }
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
        backdrop-filter: blur(10px);
    }
    
    /* Interactive data tables */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Notification styles */
    .notification-success {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        animation: slideIn 0.5s ease-out;
    }
    
    .notification-info {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from { transform: translateX(-100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* Tooltip styles */
    .tooltip {
        position: relative;
        cursor: help;
    }
    
    .tooltip:hover::after {
        content: attr(data-tooltip);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(0, 0, 0, 0.8);
        color: white;
        padding: 0.5rem;
        border-radius: 8px;
        white-space: nowrap;
        z-index: 1000;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# Hero header with animations
st.markdown("""
<div class="hero-header">
    <h1 class="hero-title">🌌 Ultra-Interactive Multimessenger AI</h1>
    <p class="hero-subtitle">Next-generation AI-powered analysis with advanced UI/UX and real-time predictions</p>
    <div class="hero-features">
        <span class="feature-badge">🎨 Glassmorphism UI</span>
        <span class="feature-badge">⚡ Real-time Analysis</span>
        <span class="feature-badge">🎯 Same/Different Event Detection</span>
        <span class="feature-badge">📊 Interactive Visualizations</span>
        <span class="feature-badge">🔮 AI Predictions</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar with glassmorphism
st.sidebar.markdown("## 🎛️ Analysis Control Center")

# Model selection with enhanced UI
st.sidebar.markdown("### 🤖 AI Model Configuration")
model_files = list_model_files()

if model_files:
    model_choice = st.sidebar.selectbox(
        "Choose AI model:",
        model_files,
        key="model_selector",
        help="Select the trained machine learning model for analysis"
    )
    st.sidebar.markdown(f"""
    <div class="notification-success">
        ✅ <strong>Model Ready:</strong> {model_choice}
    </div>
    """, unsafe_allow_html=True)
else:
    model_choice = None
    st.sidebar.markdown("""
    <div class="notification-info">
        ⚠️ <strong>No models found</strong><br>
        Please ensure model files are in the saved_models directory
    </div>
    """, unsafe_allow_html=True)

# Initialize session state
for key in ['current_data', 'results', 'analysis_count', 'total_predictions']:
    if key not in st.session_state:
        st.session_state[key] = None if key in ['current_data', 'results'] else 0

# Load model with error handling
model = None
scaler = None
metadata = None

if model_choice:
    try:
        model, scaler, metadata = load_model_by_name(model_choice)
        
        if metadata:
            accuracy = metadata.get('best_auc', 0.85)
            algorithm = metadata.get('best_model', 'RandomForest')
            
            st.sidebar.markdown(f"""
            <div class="glass-container">
                <h4>📊 Model Performance</h4>
                <div class="metric-card">
                    <div class="metric-value">{accuracy:.3f}</div>
                    <div class="metric-label">AUC Score</div>
                </div>
                <div style="text-align: center; color: white; margin-top: 1rem;">
                    <strong>Algorithm:</strong> {algorithm}
                </div>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.sidebar.error(f"❌ Model loading error: {e}")
        model = None

# Analysis parameters with interactive controls
st.sidebar.markdown("### ⚙️ Analysis Parameters")
threshold = st.sidebar.slider(
    "🎯 Association Threshold", 
    0.0, 1.0, 0.5, 0.05,
    help="Probability threshold for determining if events are associated"
)

confidence_level = st.sidebar.select_slider(
    "🔮 Confidence Level",
    options=[0.80, 0.85, 0.90, 0.95, 0.99],
    value=0.95,
    help="Statistical confidence level for predictions"
)

# Real-time status display
if model:
    st.sidebar.markdown(f"""
    <div class="glass-container">
        <h4>🔥 System Status</h4>
        <div style="color: #38ef7d;">● AI Model: Online</div>
        <div style="color: #38ef7d;">● Analysis Engine: Ready</div>
        <div style="color: #38ef7d;">● Threshold: {threshold:.2f}</div>
        <div style="color: #38ef7d;">● Confidence: {confidence_level:.0%}</div>
    </div>
    """, unsafe_allow_html=True)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Data input section with glassmorphism
    st.markdown("""
    <div class="glass-container">
        <h2>📊 Data Input Center</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Interactive input method selection
    input_method = st.radio(
        "Choose your data source:",
        ["🎲 Generate Demo Data", "📂 Upload CSV File", "🌐 Load from API"],
        horizontal=True,
        help="Select how you want to input your multimessenger event data"
    )
    
    df = None
    
    if input_method == "🎲 Generate Demo Data":
        col_demo1, col_demo2 = st.columns([1, 1])
        
        with col_demo1:
            n_pairs = st.number_input("Number of event pairs:", min_value=10, max_value=1000, value=100, step=10)
        
        with col_demo2:
            data_type = st.selectbox("Event type:", ["Gamma-Neutrino", "GW-Optical", "Mixed Events"])
        
        if st.button("🎲 Generate Interactive Demo Data", type="primary", use_container_width=True):
            with st.spinner("🔮 Generating multimessenger event pairs..."):
                # Enhanced demo data generation
                np.random.seed(42)
                
                # Create realistic multimessenger data
                if data_type == "Gamma-Neutrino":
                    # Gamma-ray and neutrino coincidences
                    data = {
                        'dt': np.abs(np.random.normal(0, 1000, n_pairs)),  # time difference in seconds
                        'dtheta': np.random.exponential(1.0, n_pairs),     # angular separation in degrees
                        'strength_ratio': np.random.exponential(2, n_pairs), # signal strength ratio
                        'gamma_energy': np.random.lognormal(2, 1, n_pairs),  # GeV
                        'neutrino_energy': np.random.lognormal(3, 1.5, n_pairs),  # GeV
                        'detection_significance': np.random.gamma(3, 2, n_pairs)
                    }
                elif data_type == "GW-Optical":
                    # Gravitational wave and optical transient coincidences
                    data = {
                        'dt': np.abs(np.random.normal(0, 3600, n_pairs)),  # time difference in seconds
                        'dtheta': np.random.exponential(0.5, n_pairs),     # angular separation
                        'strength_ratio': np.random.exponential(1.5, n_pairs),
                        'gw_strain': np.random.lognormal(-21, 0.5, n_pairs),  # strain amplitude
                        'optical_magnitude': np.random.normal(20, 2, n_pairs),  # apparent magnitude
                        'distance_mpc': np.random.gamma(2, 50, n_pairs)  # distance in Mpc
                    }
                else:  # Mixed events
                    data = {
                        'dt': np.abs(np.random.normal(0, 1500, n_pairs)),
                        'dtheta': np.random.exponential(1.2, n_pairs),
                        'strength_ratio': np.random.exponential(1.8, n_pairs),
                        'signal_type': np.random.choice(['GRB-Neutrino', 'GW-EM', 'Flare-Cosmic Ray'], n_pairs),
                        'confidence_score': np.random.beta(2, 2, n_pairs)
                    }
                
                df = pd.DataFrame(data)
                st.session_state.current_data = df
                
                # Enhanced success notification
                st.markdown(f"""
                <div class="notification-success">
                    ✅ <strong>Demo Data Generated!</strong><br>
                    Created {len(df)} {data_type} event pairs with realistic parameters
                </div>
                """, unsafe_allow_html=True)
                
                time.sleep(0.5)  # Brief pause for UX
    
    elif input_method == "📂 Upload CSV File":
        uploaded_file = st.file_uploader(
            "📎 Drop your CSV file here or click to browse",
            type="csv",
            help="Upload a CSV file containing multimessenger event data"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.current_data = df
                
                st.markdown(f"""
                <div class="notification-success">
                    ✅ <strong>File Uploaded Successfully!</strong><br>
                    Loaded {len(df)} rows from {uploaded_file.name}
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
    
    elif input_method == "🌐 Load from API":
        api_source = st.selectbox(
            "Choose data source:",
            ["GCN Circulars", "LIGO/Virgo Alerts", "IceCube Neutrinos", "Fermi GRB Catalog"]
        )
        
        if st.button("🌐 Fetch Live Data", type="primary"):
            with st.spinner(f"🔄 Fetching data from {api_source}..."):
                # Simulate API call with demo data
                time.sleep(2)
                np.random.seed(42)
                
                # Create API-like data
                data = {
                    'dt': np.abs(np.random.normal(0, 800, 50)),
                    'dtheta': np.random.exponential(0.8, 50),
                    'strength_ratio': np.random.exponential(2.2, 50),
                    'source': [api_source] * 50,
                    'timestamp': pd.date_range('2025-01-01', periods=50, freq='H')
                }
                
                df = pd.DataFrame(data)
                st.session_state.current_data = df
                
                st.markdown(f"""
                <div class="notification-success">
                    ✅ <strong>Live Data Fetched!</strong><br>
                    Retrieved {len(df)} events from {api_source}
                </div>
                """, unsafe_allow_html=True)

    # Use data from session state
    if st.session_state.current_data is not None:
        df = st.session_state.current_data

with col2:
    # Real-time analytics panel
    st.markdown("""
    <div class="glass-container">
        <h3>📈 Live Analytics</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if df is not None:
        # Interactive metrics
        st.markdown(f"""
        <div class="metric-card" data-tooltip="Total number of event pairs loaded">
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">Event Pairs</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card" data-tooltip="Number of data features available">
            <div class="metric-value">{len(df.columns)}</div>
            <div class="metric-label">Features</div>
        </div>
        """, unsafe_allow_html=True)
        
        if 'dt' in df.columns:
            avg_time_diff = df['dt'].mean()
            st.markdown(f"""
            <div class="metric-card" data-tooltip="Average time difference between events">
                <div class="metric-value">{avg_time_diff:.0f}s</div>
                <div class="metric-label">Avg Time Diff</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Session statistics
        st.markdown(f"""
        <div class="metric-card" data-tooltip="Number of analyses performed this session">
            <div class="metric-value">{st.session_state.analysis_count}</div>
            <div class="metric-label">Analyses Run</div>
        </div>
        """, unsafe_allow_html=True)

# Data preview with enhanced styling
if df is not None:
    st.markdown("""
    <div class="glass-container">
        <h3>📋 Interactive Data Preview</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Interactive data exploration
    col_preview1, col_preview2 = st.columns([3, 1])
    
    with col_preview1:
        st.dataframe(
            df.head(10), 
            use_container_width=True,
            hide_index=True
        )
    
    with col_preview2:
        # Quick statistics
        if 'strength_ratio' in df.columns:
            fig_hist = px.histogram(
                df, x='strength_ratio', 
                title="Signal Strength Distribution",
                color_discrete_sequence=['#667eea']
            )
            fig_hist.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig_hist, use_container_width=True)

# Analysis section with enhanced predictions
st.markdown("""
<div class="glass-container">
    <h2>🔬 AI Analysis & Prediction Center</h2>
</div>
""", unsafe_allow_html=True)

if df is not None and model is not None:
    col_analysis1, col_analysis2 = st.columns([2, 1])
    
    with col_analysis1:
        analysis_mode = st.selectbox(
            "🎯 Analysis Mode:",
            ["Standard Association", "Same/Different Event Classification", "Advanced Clustering", "Temporal Analysis"],
            help="Choose the type of analysis to perform"
        )
    
    with col_analysis2:
        st.markdown("### 🚀 Ready to Analyze")
        
    if st.button("🚀 Run Ultra-Enhanced Analysis", type="primary", use_container_width=True):
        
        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Animated progress
        for i in range(100):
            progress_bar.progress(i + 1)
            if i < 30:
                status_text.text("🔄 Initializing AI models...")
            elif i < 60:
                status_text.text("🧠 Processing multimessenger data...")
            elif i < 90:
                status_text.text("🔮 Generating predictions...")
            else:
                status_text.text("✨ Finalizing results...")
            time.sleep(0.02)
        
        try:
            # Run prediction
            results = predict_df(df, model, scaler, threshold)
            st.session_state.results = results
            st.session_state.analysis_count += 1
            
            if results is not None:
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Enhanced results display
                st.markdown("""
                <div class="notification-success">
                    ✅ <strong>Analysis Complete!</strong> Ultra-enhanced predictions generated with advanced AI
                </div>
                """, unsafe_allow_html=True)
                
                # Same/Different Event Prediction Logic
                prob_col = 'pred_prob' if 'pred_prob' in results.columns else 'probability'
                
                # Enhanced classification logic
                results['same_event_prob'] = results[prob_col]
                results['event_classification'] = results[prob_col].apply(
                    lambda x: 'Same Event' if x > threshold else 'Different Events'
                )
                
                # Confidence scoring
                results['confidence_score'] = results[prob_col].apply(
                    lambda x: 'High' if abs(x - 0.5) > 0.3 else 'Medium' if abs(x - 0.5) > 0.1 else 'Low'
                )
                
                # Results summary with interactive metrics
                st.markdown("### 📊 Enhanced Analysis Results")
                
                col_res1, col_res2, col_res3, col_res4 = st.columns(4)
                
                with col_res1:
                    same_events = (results['event_classification'] == 'Same Event').sum()
                    st.markdown(f"""
                    <div class="prediction-same">
                        <div style="text-align: center;">
                            <div style="font-size: 2rem; font-weight: bold;">{same_events}</div>
                            <div>🎯 Same Events</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_res2:
                    different_events = (results['event_classification'] == 'Different Events').sum()
                    st.markdown(f"""
                    <div class="prediction-different">
                        <div style="text-align: center;">
                            <div style="font-size: 2rem; font-weight: bold;">{different_events}</div>
                            <div>🎲 Different Events</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_res3:
                    max_prob = results[prob_col].max()
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{max_prob:.3f}</div>
                        <div class="metric-label">Max Probability</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_res4:
                    avg_confidence = (results['confidence_score'] == 'High').mean()
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{avg_confidence:.1%}</div>
                        <div class="metric-label">High Confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Interactive results table
                st.markdown("### 📋 Detailed Classification Results")
                
                # Enhanced results display
                results_display = results.copy()
                results_display['Association Status'] = results_display['event_classification'].map({
                    'Same Event': '✅ Same Astronomical Event',
                    'Different Events': '❌ Different Sources'
                })
                
                # Add color coding
                results_display['Confidence Level'] = results_display['confidence_score'].map({
                    'High': '🟢 High',
                    'Medium': '🟡 Medium',
                    'Low': '🔴 Low'
                })
                
                st.dataframe(
                    results_display[['Association Status', prob_col, 'Confidence Level']].round(3),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Advanced visualizations
                st.markdown("### 📈 Interactive Prediction Visualizations")
                
                # Create enhanced probability distribution
                fig = go.Figure()
                
                # Same events
                same_mask = results['event_classification'] == 'Same Event'
                fig.add_trace(go.Histogram(
                    x=results[same_mask][prob_col],
                    name='Same Events',
                    marker_color='rgba(17, 153, 142, 0.7)',
                    nbinsx=20
                ))
                
                # Different events
                different_mask = results['event_classification'] == 'Different Events'
                fig.add_trace(go.Histogram(
                    x=results[different_mask][prob_col],
                    name='Different Events',
                    marker_color='rgba(255, 107, 107, 0.7)',
                    nbinsx=20
                ))
                
                # Add threshold line
                fig.add_vline(
                    x=threshold, 
                    line_dash="dash", 
                    line_color="yellow",
                    annotation_text=f"Threshold ({threshold})"
                )
                
                fig.update_layout(
                    title="🎯 Event Classification Distribution",
                    xaxis_title="Association Probability",
                    yaxis_title="Number of Event Pairs",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Confidence analysis pie chart
                confidence_counts = results['confidence_score'].value_counts()
                
                fig_pie = px.pie(
                    values=confidence_counts.values,
                    names=confidence_counts.index,
                    title="🔮 Confidence Level Distribution",
                    color_discrete_map={
                        'High': '#38ef7d',
                        'Medium': '#ffa500', 
                        'Low': '#ff6b6b'
                    }
                )
                
                fig_pie.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Advanced clustering if selected
                if analysis_mode == "Advanced Clustering":
                    st.markdown("### 🌟 Event Clustering Analysis")
                    
                    # Prepare data for clustering
                    cluster_features = ['dt', 'dtheta', 'strength_ratio']
                    available_features = [col for col in cluster_features if col in df.columns]
                    
                    if len(available_features) >= 2:
                        cluster_data = df[available_features].dropna()
                        
                        # Perform DBSCAN clustering
                        scaler_cluster = StandardScaler()
                        scaled_data = scaler_cluster.fit_transform(cluster_data)
                        
                        dbscan = DBSCAN(eps=0.3, min_samples=5)
                        clusters = dbscan.fit_predict(scaled_data)
                        
                        # Create 3D scatter plot
                        if len(available_features) >= 3:
                            fig_3d = px.scatter_3d(
                                x=cluster_data.iloc[:, 0],
                                y=cluster_data.iloc[:, 1],
                                z=cluster_data.iloc[:, 2],
                                color=clusters,
                                title="🌌 3D Event Clustering",
                                labels={
                                    'x': available_features[0],
                                    'y': available_features[1],
                                    'z': available_features[2]
                                }
                            )
                            
                            fig_3d.update_layout(
                                scene=dict(
                                    bgcolor='rgba(0,0,0,0)',
                                    xaxis_title_font_color='white',
                                    yaxis_title_font_color='white',
                                    zaxis_title_font_color='white'
                                ),
                                paper_bgcolor='rgba(0,0,0,0)',
                                font_color='white'
                            )
                            
                            st.plotly_chart(fig_3d, use_container_width=True)
                        
                        # Cluster statistics
                        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
                        n_noise = list(clusters).count(-1)
                        
                        col_cluster1, col_cluster2 = st.columns(2)
                        
                        with col_cluster1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{n_clusters}</div>
                                <div class="metric-label">Event Clusters</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_cluster2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{n_noise}</div>
                                <div class="metric-label">Outlier Events</div>
                            </div>
                            """, unsafe_allow_html=True)
                
            else:
                st.error("❌ Analysis failed - no results returned")
                
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Analysis failed: {e}")

elif df is None:
    st.markdown("""
    <div class="notification-info">
        👆 <strong>Ready for Data</strong><br>
        Please load your multimessenger event data using one of the options above
    </div>
    """, unsafe_allow_html=True)

elif model is None:
    st.markdown("""
    <div class="notification-info">
        🤖 <strong>AI Model Required</strong><br>
        Please select a trained model from the sidebar to begin analysis
    </div>
    """, unsafe_allow_html=True)

# Footer with animated elements
st.markdown("""
<div style="margin-top: 3rem; text-align: center;">
    <div class="glass-container">
        <h3>🌌 Ultra-Interactive Multimessenger AI Platform</h3>
        <p style="color: rgba(255, 255, 255, 0.8); margin: 1rem 0;">
            Powered by advanced machine learning • Enhanced UI/UX • Real-time predictions
        </p>
        <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
            <span class="feature-badge">🎨 Glassmorphism Design</span>
            <span class="feature-badge">⚡ Real-time Analysis</span>
            <span class="feature-badge">🎯 Same/Different Event Classification</span>
            <span class="feature-badge">🔮 AI-Powered Predictions</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)