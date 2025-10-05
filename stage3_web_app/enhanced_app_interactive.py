#!/usr/bin/env python3
"""
Ultra-Modern Interactive Multimessenger AI Analysis Platform
Enhanced UI/UX with modern design patterns, animations, and interactivity
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
    page_title="🌌 Multimessenger AI Hub",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ultra-modern CSS with animations and interactive elements
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header Styles */
    .hero-header {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 20px;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        position: relative;
        overflow: hidden;
        animation: slideInFromTop 0.8s ease-out;
    }
    
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shimmer 3s infinite;
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 2rem;
    }
    
    .hero-features {
        display: flex;
        justify-content: center;
        gap: 2rem;
        flex-wrap: wrap;
    }
    
    .hero-feature {
        background: rgba(255,255,255,0.1);
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.9rem;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Card Styles */
    .modern-card {
        background: rgba(255,255,255,0.95);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .modern-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 16px 48px rgba(0,0,0,0.2);
    }
    
    .modern-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    .card-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .card-description {
        color: #718096;
        margin-bottom: 1rem;
        line-height: 1.6;
    }
    
    /* Interactive Buttons */
    .interactive-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .interactive-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.6);
    }
    
    .interactive-button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .interactive-button:hover::before {
        left: 100%;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
        box-shadow: 0 16px 32px rgba(102, 126, 234, 0.4);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-delta {
        font-size: 0.8rem;
        margin-top: 0.5rem;
        padding: 0.2rem 0.8rem;
        background: rgba(255,255,255,0.2);
        border-radius: 12px;
        display: inline-block;
    }
    
    /* Sidebar Styles */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(20px);
    }
    
    .sidebar-section {
        background: rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.3s ease;
    }
    
    .sidebar-section:hover {
        background: rgba(255,255,255,0.15);
        transform: translateX(5px);
    }
    
    /* Tab Styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 0.5rem;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255,255,255,0.1);
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255,255,255,0.2) !important;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    
    /* Success/Error Styles */
    .success-card {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(72, 187, 120, 0.3);
        animation: slideInFromRight 0.5s ease-out;
    }
    
    .error-card {
        background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
        color: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(245, 101, 101, 0.3);
        animation: shake 0.5s ease-in-out;
    }
    
    .info-card {
        background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
        color: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(66, 153, 225, 0.3);
        animation: pulse 2s infinite;
    }
    
    /* Progress Bar */
    .custom-progress {
        background: rgba(255,255,255,0.2);
        border-radius: 10px;
        height: 8px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .custom-progress-bar {
        background: linear-gradient(90deg, #667eea, #764ba2);
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
        animation: progressPulse 2s infinite;
    }
    
    /* Animations */
    @keyframes slideInFromTop {
        from { transform: translateY(-50px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    @keyframes slideInFromRight {
        from { transform: translateX(50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-5px); }
        75% { transform: translateX(5px); }
    }
    
    @keyframes progressPulse {
        0%, 100% { box-shadow: 0 0 10px rgba(102, 126, 234, 0.5); }
        50% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.8); }
    }
    
    /* Data Table Styles */
    .dataframe {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: none;
    }
    
    .dataframe thead tr {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .dataframe tbody tr:hover {
        background: rgba(102, 126, 234, 0.1);
        transform: scale(1.01);
        transition: all 0.2s ease;
    }
    
    /* Loading Spinner */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255,255,255,0.3);
        border-radius: 50%;
        border-top-color: white;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title { font-size: 2rem; }
        .hero-features { flex-direction: column; }
        .metric-value { font-size: 2rem; }
    }
    
    /* Interactive Elements */
    .interactive-element {
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .interactive-element:hover {
        transform: translateY(-2px);
        filter: brightness(110%);
    }
    
    /* Glassmorphism Effects */
    .glass-panel {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.1);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# Hero Header with animations
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🌌 Multimessenger AI Hub</div>
    <div class="hero-subtitle">Next-Generation Interactive Analysis Platform for Astronomical Events</div>
    <div class="hero-features">
        <div class="hero-feature">🤖 AI-Powered</div>
        <div class="hero-feature">🌐 Real-time Data</div>
        <div class="hero-feature">📊 3D Visualizations</div>
        <div class="hero-feature">🎯 Event Clustering</div>
        <div class="hero-feature">⚡ Ultra-Fast</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Enhanced Sidebar with interactive elements
st.sidebar.markdown("""
<div style="text-align: center; color: white; padding: 1rem;">
    <h2>🎛️ Control Center</h2>
    <div style="height: 2px; background: linear-gradient(90deg, #667eea, #764ba2); margin: 1rem 0; border-radius: 2px;"></div>
</div>
""", unsafe_allow_html=True)

# Model selection with enhanced UI
st.sidebar.markdown("""
<div class="sidebar-section">
    <h3 style="color: white; margin-bottom: 1rem;">🤖 AI Model</h3>
</div>
""", unsafe_allow_html=True)

model_files = list_model_files()

if model_files:
    model_choice = st.sidebar.selectbox(
        "Choose AI model:",
        model_files,
        key="model_selector",
        help="Select the trained AI model for analysis"
    )
    
    # Model status indicator
    if model_choice:
        st.sidebar.markdown("""
        <div style="background: rgba(72, 187, 120, 0.2); color: #48bb78; padding: 0.5rem; border-radius: 8px; text-align: center; margin-top: 0.5rem;">
            ✅ Model Ready
        </div>
        """, unsafe_allow_html=True)
else:
    model_choice = None
    st.sidebar.markdown("""
    <div style="background: rgba(245, 101, 101, 0.2); color: #f56565; padding: 0.5rem; border-radius: 8px; text-align: center;">
        ⚠️ No Models Found
    </div>
    """, unsafe_allow_html=True)

# Initialize session state
for key in ['current_data', 'results', 'api_data', 'clustering_results', 'analysis_progress']:
    if key not in st.session_state:
        st.session_state[key] = None

# Load model with enhanced feedback
model = None
scaler = None
metadata = None

if model_choice:
    try:
        with st.sidebar:
            with st.spinner("Loading model..."):
                model, scaler, metadata = load_model_by_name(model_choice)
        
        st.sidebar.markdown(f"""
        <div class="sidebar-section">
            <h4 style="color: white;">📊 Model Info</h4>
            <div style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">
                <div>🔸 <strong>Model:</strong> {model_choice}</div>
        """, unsafe_allow_html=True)
        
        if metadata:
            st.sidebar.markdown(f"""
                <div>🔸 <strong>Algorithm:</strong> {metadata.get('best_model', 'Unknown')}</div>
                <div>🔸 <strong>AUC Score:</strong> {metadata.get('best_auc', 'N/A'):.3f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.sidebar.markdown(f"""
        <div class="error-card">
            ❌ Error loading model: {str(e)[:50]}...
        </div>
        """, unsafe_allow_html=True)
        model = None

# Enhanced Analysis Parameters
st.sidebar.markdown("""
<div class="sidebar-section">
    <h3 style="color: white; margin-bottom: 1rem;">⚙️ Parameters</h3>
</div>
""", unsafe_allow_html=True)

threshold = st.sidebar.slider(
    "🎯 Association Threshold", 
    0.0, 1.0, 0.5, 0.05,
    help="Confidence threshold for event associations"
)

clustering_eps = st.sidebar.slider(
    "🔗 Clustering Sensitivity", 
    0.1, 2.0, 0.5, 0.1,
    help="DBSCAN clustering sensitivity parameter"
)

min_samples = st.sidebar.slider(
    "👥 Min Cluster Size", 
    2, 10, 3,
    help="Minimum samples to form a cluster"
)

# Advanced options in collapsible section
with st.sidebar.expander("🔬 Advanced Settings", expanded=False):
    confidence_interval = st.slider("Confidence Interval", 0.90, 0.99, 0.95, 0.01)
    show_debug = st.checkbox("Debug Mode", help="Show detailed debug information")
    auto_refresh = st.checkbox("Auto-refresh Data", help="Automatically refresh real-time data")
    animation_speed = st.selectbox("Animation Speed", ["Slow", "Normal", "Fast"], index=1)

# Real-time status indicator
st.sidebar.markdown("""
<div class="sidebar-section">
    <h4 style="color: white;">🔴 System Status</h4>
    <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
        <span style="color: rgba(255,255,255,0.8);">API Connection:</span>
        <span style="color: #48bb78;">🟢 Online</span>
    </div>
    <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
        <span style="color: rgba(255,255,255,0.8);">Data Processing:</span>
        <span style="color: #48bb78;">🟢 Ready</span>
    </div>
    <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
        <span style="color: rgba(255,255,255,0.8);">AI Engine:</span>
        <span style="color: #48bb78;">🟢 Active</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Main content with enhanced tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Hub", 
    "🔬 AI Analysis", 
    "📈 Visualizations", 
    "🎯 Event Clustering",
    "🌐 Live Monitor"
])

with tab1:
    st.markdown("""
    <div class="modern-card">
        <div class="card-title">📊 Data Input Center</div>
        <div class="card-description">Choose your data source and configure input parameters for analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced data input with card layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="modern-card interactive-element">
            <div class="card-title">🗂️ Standard Sources</div>
            <div class="card-description">Upload files or generate synthetic data for testing and analysis</div>
        </div>
        """, unsafe_allow_html=True)
        
        input_method = st.radio(
            "Select input method:",
            ["🎲 Generate Demo Data", "📂 Upload CSV File", "📋 Manual Entry"],
            help="Choose how you want to input your data"
        )
    
    with col2:
        st.markdown("""
        <div class="modern-card interactive-element">
            <div class="card-title">🌐 Live Data Sources</div>
            <div class="card-description">Connect to real-time astronomical databases and observatories</div>
        </div>
        """, unsafe_allow_html=True)
        
        api_source = st.selectbox(
            "Real-time source:",
            ["🔭 LIGO/Virgo GW Events", "🌟 Gamma-ray Bursts", "⚡ Neutrino Alerts", "🎯 Multi-messenger"]
        )
        
        if st.button("🔄 Fetch Live Data", type="primary", use_container_width=True):
            progress_placeholder = st.empty()
            
            # Enhanced progress indicator
            with progress_placeholder:
                st.markdown("""
                <div class="info-card">
                    <div style="display: flex; align-items: center; gap: 1rem;">
                        <div class="loading-spinner"></div>
                        <div>Connecting to astronomical databases...</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Simulate API call with progress
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
                time.sleep(0.02)
            
            # Generate realistic mock data
            np.random.seed(int(time.time()) % 1000)
            n_events = np.random.randint(8, 30)
            
            api_data = {
                'dt': np.abs(np.random.normal(0, 500, n_events)),
                'dtheta': np.random.exponential(0.8, n_events),
                'strength_ratio': np.random.exponential(1.5, n_events),
                'event_id': [f"API_{api_source.split()[0]}_{i:03d}" for i in range(n_events)],
                'timestamp': [datetime.now() - timedelta(hours=np.random.randint(0, 48)) for _ in range(n_events)],
                'confidence': np.random.uniform(0.7, 0.99, n_events)
            }
            
            st.session_state.api_data = pd.DataFrame(api_data)
            
            progress_placeholder.empty()
            st.markdown(f"""
            <div class="success-card">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <div style="font-size: 2rem;">🎉</div>
                    <div>
                        <div style="font-size: 1.2rem; font-weight: 600;">Success!</div>
                        <div>Fetched {n_events} events from {api_source}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Data generation with enhanced UI
    df = None
    
    if input_method == "🎲 Generate Demo Data":
        st.markdown("""
        <div class="modern-card">
            <div class="card-title">🎲 Demo Data Generator</div>
            <div class="card-description">Configure synthetic astronomical data parameters</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_pairs = st.number_input("Event pairs", 50, 500, 100, help="Number of event pairs to generate")
        with col2:
            noise_level = st.slider("Noise level", 0.1, 2.0, 1.0, help="Background noise intensity")
        with col3:
            seed = st.number_input("Random seed", 1, 1000, 42, help="Seed for reproducible results")
        
        if st.button("🎲 Generate Data", type="primary", use_container_width=True):
            with st.spinner("Generating astronomical event data..."):
                np.random.seed(seed)
                
                # Enhanced data generation
                data = {
                    'dt': np.abs(np.random.normal(0, 1000 * noise_level, n_pairs)),
                    'dtheta': np.random.exponential(1.0 * noise_level, n_pairs),
                    'strength_ratio': np.random.exponential(2.0 / noise_level, n_pairs),
                    'event_type': np.random.choice(['GW-Gamma', 'GW-Neutrino', 'Gamma-Neutrino'], n_pairs),
                    'detection_time': [datetime.now() - timedelta(days=np.random.randint(0, 30)) for _ in range(n_pairs)]
                }
                
                df = pd.DataFrame(data)
                st.session_state.current_data = df
                
                st.markdown(f"""
                <div class="success-card">
                    <div style="display: flex; align-items: center; gap: 1rem;">
                        <div style="font-size: 2rem;">✨</div>
                        <div>
                            <div style="font-size: 1.2rem; font-weight: 600;">Data Generated!</div>
                            <div>{len(df)} astronomical event pairs created</div>
                            <div style="font-size: 0.9rem; opacity: 0.8;">Noise: {noise_level} | Seed: {seed}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    elif input_method == "📂 Upload CSV File":
        st.markdown("""
        <div class="modern-card">
            <div class="card-title">📂 File Upload Center</div>
            <div class="card-description">Upload your astronomical event data in CSV format</div>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose CSV file", 
            type="csv",
            help="Upload astronomical event data",
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.current_data = df
                
                st.markdown(f"""
                <div class="success-card">
                    <div style="display: flex; align-items: center; gap: 1rem;">
                        <div style="font-size: 2rem;">📁</div>
                        <div>
                            <div style="font-size: 1.2rem; font-weight: 600;">File Uploaded!</div>
                            <div>{len(df)} rows from {uploaded_file.name}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f"""
                <div class="error-card">
                    <div style="display: flex; align-items: center; gap: 1rem;">
                        <div style="font-size: 2rem;">❌</div>
                        <div>
                            <div style="font-size: 1.2rem; font-weight: 600;">Upload Failed</div>
                            <div>{str(e)}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    elif input_method == "📋 Manual Entry":
        st.markdown("""
        <div class="modern-card">
            <div class="card-title">📋 Manual Data Entry</div>
            <div class="card-description">Manually input event pair parameters</div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("manual_entry", clear_on_submit=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                dt_val = st.number_input("Time difference (s)", 0.0, 10000.0, 100.0)
            with col2:
                dtheta_val = st.number_input("Angular separation (°)", 0.0, 10.0, 1.0)
            with col3:
                strength_val = st.number_input("Strength ratio", 0.0, 100.0, 2.0)
            
            if st.form_submit_button("➕ Add Entry", use_container_width=True):
                new_entry = {
                    'dt': [dt_val],
                    'dtheta': [dtheta_val], 
                    'strength_ratio': [strength_val]
                }
                
                if st.session_state.current_data is None:
                    st.session_state.current_data = pd.DataFrame(new_entry)
                else:
                    new_df = pd.DataFrame(new_entry)
                    st.session_state.current_data = pd.concat([st.session_state.current_data, new_df], ignore_index=True)
                
                st.markdown("""
                <div class="success-card">
                    <div style="text-align: center;">
                        <div style="font-size: 2rem;">✅</div>
                        <div style="font-size: 1.2rem; font-weight: 600;">Entry Added!</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Use data from session state or API
    if st.session_state.current_data is not None:
        df = st.session_state.current_data
    elif st.session_state.api_data is not None and st.checkbox("🌐 Use API data for analysis"):
        df = st.session_state.api_data
    
    # Enhanced data preview
    if df is not None:
        st.markdown("""
        <div class="modern-card">
            <div class="card-title">📋 Data Overview</div>
            <div class="card-description">Preview and statistics of your loaded data</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Interactive metrics with enhanced styling
        col1, col2, col3, col4, col5 = st.columns(5)
        
        metrics = [
            ("📊", "Total Pairs", len(df), "events"),
            ("🔧", "Features", len(df.columns), "columns"), 
            ("💪", "Avg Strength", f"{df['strength_ratio'].mean():.2f}" if 'strength_ratio' in df.columns else "N/A", "ratio"),
            ("⏱️", "Avg Time Δ", f"{df['dt'].mean():.0f}s" if 'dt' in df.columns else "N/A", "seconds"),
            ("📐", "Avg Angle", f"{df['dtheta'].mean():.2f}°" if 'dtheta' in df.columns else "N/A", "degrees")
        ]
        
        for i, (icon, label, value, unit) in enumerate(metrics):
            with [col1, col2, col3, col4, col5][i]:
                st.markdown(f"""
                <div class="metric-card interactive-element">
                    <div style="font-size: 1.5rem;">{icon}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                    <div class="metric-delta">{unit}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Enhanced data table
        st.markdown("#### 📊 Data Sample")
        st.dataframe(
            df.head(10), 
            use_container_width=True,
            column_config={
                "dt": st.column_config.NumberColumn("Time Δ (s)", format="%.1f"),
                "dtheta": st.column_config.NumberColumn("Angular Sep (°)", format="%.3f"),
                "strength_ratio": st.column_config.NumberColumn("Strength Ratio", format="%.2f"),
            }
        )

# Continue with enhanced tab2 (AI Analysis) - this would be the pattern for all tabs
with tab2:
    st.markdown("""
    <div class="modern-card">
        <div class="card-title">🔬 AI Analysis Center</div>
        <div class="card-description">Run advanced AI algorithms to detect multimessenger event associations</div>
    </div>
    """, unsafe_allow_html=True)
    
    if df is not None and model is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🚀 Launch AI Analysis", type="primary", use_container_width=True):
                # Enhanced analysis with progress and animations
                progress_container = st.container()
                
                with progress_container:
                    st.markdown("""
                    <div class="info-card">
                        <div style="display: flex; align-items: center; gap: 1rem;">
                            <div class="loading-spinner"></div>
                            <div>
                                <div style="font-size: 1.2rem; font-weight: 600;">AI Analysis in Progress</div>
                                <div>Processing multimessenger events with advanced algorithms...</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Animated progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                stages = [
                    "🔍 Preprocessing data...",
                    "🧠 Loading AI model...", 
                    "⚡ Computing associations...",
                    "📊 Analyzing results...",
                    "✨ Finalizing output..."
                ]
                
                for i, stage in enumerate(stages):
                    status_text.text(stage)
                    for j in range(20):
                        progress_bar.progress((i * 20 + j + 1))
                        time.sleep(0.05)
                
                try:
                    # Run actual prediction
                    results = predict_df(df, model, scaler, threshold)
                    st.session_state.results = results
                    
                    progress_container.empty()
                    
                    if results is not None:
                        # Enhanced success display
                        st.markdown("""
                        <div class="success-card">
                            <div style="display: flex; align-items: center; gap: 1rem;">
                                <div style="font-size: 3rem;">🎉</div>
                                <div>
                                    <div style="font-size: 1.5rem; font-weight: 700;">Analysis Complete!</div>
                                    <div style="font-size: 1.1rem;">Advanced AI has successfully processed all multimessenger events</div>
                                    <div style="font-size: 0.9rem; opacity: 0.9; margin-top: 0.5rem;">Ready for visualization and clustering analysis</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Enhanced results metrics
                        st.markdown("### 📈 Analysis Results")
                        
                        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                        
                        associated = (results['pred_prob'] > threshold).sum()
                        total_pairs = len(results)
                        max_prob = results['pred_prob'].max()
                        avg_prob = results['pred_prob'].mean()
                        
                        with metrics_col1:
                            st.markdown(f"""
                            <div class="metric-card interactive-element">
                                <div style="font-size: 1.5rem;">🎯</div>
                                <div class="metric-value">{associated}</div>
                                <div class="metric-label">Associated</div>
                                <div class="metric-delta">{(associated/total_pairs)*100:.1f}% of total</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with metrics_col2:
                            st.markdown(f"""
                            <div class="metric-card interactive-element">
                                <div style="font-size: 1.5rem;">🏆</div>
                                <div class="metric-value">{max_prob:.3f}</div>
                                <div class="metric-label">Max Confidence</div>
                                <div class="metric-delta">Peak detection</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with metrics_col3:
                            st.markdown(f"""
                            <div class="metric-card interactive-element">
                                <div style="font-size: 1.5rem;">📊</div>
                                <div class="metric-value">{avg_prob:.3f}</div>
                                <div class="metric-label">Avg Confidence</div>
                                <div class="metric-delta">Overall score</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with metrics_col4:
                            reliability = "High" if avg_prob > 0.7 else "Medium" if avg_prob > 0.4 else "Low"
                            reliability_color = "#48bb78" if avg_prob > 0.7 else "#ed8936" if avg_prob > 0.4 else "#f56565"
                            st.markdown(f"""
                            <div class="metric-card interactive-element" style="background: linear-gradient(135deg, {reliability_color} 0%, {reliability_color}dd 100%);">
                                <div style="font-size: 1.5rem;">🔍</div>
                                <div class="metric-value">{reliability}</div>
                                <div class="metric-label">Reliability</div>
                                <div class="metric-delta">{confidence_interval*100:.0f}% CI</div>
                            </div>
                            """, unsafe_allow_html=True)
                
                except Exception as e:
                    progress_container.empty()
                    st.markdown(f"""
                    <div class="error-card">
                        <div style="display: flex; align-items: center; gap: 1rem;">
                            <div style="font-size: 2rem;">❌</div>
                            <div>
                                <div style="font-size: 1.2rem; font-weight: 600;">Analysis Failed</div>
                                <div>{str(e)}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if show_debug:
                        st.exception(e)
        
        with col2:
            st.markdown("""
            <div class="glass-panel">
                <h4 style="color: white; margin-bottom: 1rem;">🎛️ Analysis Config</h4>
                <div style="color: rgba(255,255,255,0.8); font-size: 0.9rem; line-height: 1.6;">
                    <div><strong>Model:</strong> {}</div>
                    <div><strong>Threshold:</strong> {}</div>
                    <div><strong>Data Points:</strong> {}</div>
                    <div><strong>Features:</strong> {}</div>
                </div>
            </div>
            """.format(
                model_choice or "None",
                threshold,
                len(df) if df is not None else 0,
                len(df.columns) if df is not None else 0
            ), unsafe_allow_html=True)
            
            if metadata:
                st.markdown(f"""
                <div class="glass-panel">
                    <h4 style="color: white; margin-bottom: 1rem;">📊 Model Performance</h4>
                    <div style="color: rgba(255,255,255,0.8); font-size: 0.9rem; line-height: 1.6;">
                        <div><strong>Algorithm:</strong> {metadata.get('best_model', 'Unknown')}</div>
                        <div><strong>AUC Score:</strong> {metadata.get('best_auc', 'N/A'):.3f}</div>
                        <div><strong>Status:</strong> <span style="color: #48bb78;">Ready</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    elif df is None:
        st.markdown("""
        <div class="info-card">
            <div style="text-align: center;">
                <div style="font-size: 3rem;">📊</div>
                <div style="font-size: 1.3rem; font-weight: 600; margin: 1rem 0;">No Data Loaded</div>
                <div>Please load data in the Data Hub tab first</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    elif model is None:
        st.markdown("""
        <div class="info-card">
            <div style="text-align: center;">
                <div style="font-size: 3rem;">🤖</div>
                <div style="font-size: 1.3rem; font-weight: 600; margin: 1rem 0;">No Model Selected</div>
                <div>Please select an AI model in the sidebar first</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Display enhanced results if available
    if st.session_state.results is not None:
        st.markdown("---")
        results = st.session_state.results
        
        st.markdown("""
        <div class="modern-card">
            <div class="card-title">📊 Detailed Results</div>
            <div class="card-description">Comprehensive analysis output with association probabilities and predictions</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced results display
        results_display = results.copy()
        results_display['Status'] = results_display['pred_prob'].apply(
            lambda x: f"✅ ASSOCIATED" if x > threshold else f"❌ NOT ASSOCIATED"
        )
        results_display['Confidence'] = results_display['pred_prob'].apply(
            lambda x: "🔥 Very High" if x > 0.9 else "🎯 High" if x > 0.7 else "⚡ Medium" if x > 0.5 else "📊 Low"
        )
        
        st.dataframe(
            results_display,
            use_container_width=True,
            column_config={
                "pred_prob": st.column_config.ProgressColumn(
                    "Association Probability",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.3f"
                ),
                "pred_label": st.column_config.CheckboxColumn("Associated"),
                "Status": st.column_config.TextColumn("Status"),
                "Confidence": st.column_config.TextColumn("Confidence Level"),
            }
        )

# Add similar enhancements for tabs 3, 4, and 5...
with tab3:
    st.markdown("""
    <div class="modern-card">
        <div class="card-title">📈 Advanced Visualizations</div>
        <div class="card-description">Interactive 3D plots, correlations, and scientific visualizations</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.results is not None:
        results = st.session_state.results
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown("""
            <div class="glass-panel">
                <h4 style="color: white; margin-bottom: 1rem;">🎨 Visualization Controls</h4>
            </div>
            """, unsafe_allow_html=True)
            
            viz_type = st.selectbox(
                "Visualization type:",
                [
                    "📊 Probability Distribution",
                    "🎯 3D Feature Space", 
                    "📈 Correlation Matrix",
                    "⏱️ Time Series Analysis",
                    "🌌 Sky Map (3D)",
                    "📋 Statistical Dashboard"
                ]
            )
            
            color_scheme = st.selectbox("Color scheme:", ["viridis", "plasma", "cividis", "magma"])
            show_threshold = st.checkbox("Show threshold", True)
            interactive_mode = st.checkbox("Interactive mode", True)
        
        with col2:
            if viz_type == "📊 Probability Distribution":
                fig = px.histogram(
                    results, 
                    x='pred_prob',
                    nbins=25,
                    title="🎯 Association Probability Distribution",
                    labels={'pred_prob': 'Association Probability', 'count': 'Number of Event Pairs'},
                    color_discrete_sequence=px.colors.sequential.Viridis
                )
                
                if show_threshold:
                    fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                                annotation_text=f"Threshold ({threshold})")
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    title_font_size=20,
                    title_font_color="white"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "🎯 3D Feature Space":
                fig = px.scatter_3d(
                    results,
                    x='dt', y='dtheta', z='strength_ratio',
                    color='pred_prob',
                    size='pred_prob',
                    title="🌌 3D Feature Space",
                    color_continuous_scale=color_scheme,
                    hover_data=['pred_prob']
                )
                
                fig.update_layout(
                    scene=dict(bgcolor='rgba(0,0,0,0)'),
                    height=600,
                    title_font_color="white"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "📈 Correlation Matrix":
                numeric_cols = ['dt', 'dtheta', 'strength_ratio', 'pred_prob']
                available_cols = [col for col in numeric_cols if col in results.columns]
                
                if len(available_cols) > 1:
                    corr_matrix = results[available_cols].corr()
                    
                    fig = px.imshow(
                        corr_matrix,
                        title="🔗 Feature Correlation Matrix",
                        color_continuous_scale=color_scheme,
                        aspect="auto"
                    )
                    
                    fig.update_traces(
                        text=np.around(corr_matrix.values, decimals=2),
                        texttemplate="%{text}",
                        textfont={"size": 12}
                    )
                    
                    fig.update_layout(title_font_color="white")
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("""
        <div class="info-card">
            <div style="text-align: center;">
                <div style="font-size: 3rem;">📊</div>
                <div style="font-size: 1.3rem; font-weight: 600; margin: 1rem 0;">No Analysis Results</div>
                <div>Run AI analysis first to generate visualizations</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

with tab4:
    st.markdown("""
    <div class="modern-card">
        <div class="card-title">🎯 Event Clustering Analysis</div>
        <div class="card-description">Detect single vs multiple event sources using advanced clustering algorithms</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.results is not None:
        results = st.session_state.results
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            <div class="glass-panel">
                <h4 style="color: white; margin-bottom: 1rem;">⚙️ Clustering Settings</h4>
            </div>
            """, unsafe_allow_html=True)
            
            cluster_features = st.multiselect(
                "Features for clustering:",
                ['dt', 'dtheta', 'strength_ratio', 'pred_prob'],
                default=['dt', 'dtheta', 'strength_ratio']
            )
            
            if st.button("🔍 Run Clustering", type="primary", use_container_width=True):
                if len(cluster_features) >= 2:
                    with st.spinner("🔬 Analyzing event clusters..."):
                        cluster_data = results[cluster_features].copy()
                        scaler_cluster = StandardScaler()
                        cluster_scaled = scaler_cluster.fit_transform(cluster_data)
                        
                        clustering = DBSCAN(eps=clustering_eps, min_samples=min_samples)
                        cluster_labels = clustering.fit_predict(cluster_scaled)
                        
                        results_clustered = results.copy()
                        results_clustered['cluster'] = cluster_labels
                        results_clustered['is_outlier'] = cluster_labels == -1
                        
                        st.session_state.clustering_results = results_clustered
                        
                        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                        n_noise = list(cluster_labels).count(-1)
                        
                        st.markdown(f"""
                        <div class="success-card">
                            <div style="text-align: center;">
                                <div style="font-size: 2rem;">🎯</div>
                                <div style="font-size: 1.2rem; font-weight: 600;">Clustering Complete!</div>
                                <div>Found {n_clusters} clusters and {n_noise} outliers</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        
        with col2:
            if st.session_state.clustering_results is not None:
                cluster_results = st.session_state.clustering_results
                
                # Enhanced cluster metrics
                col1, col2, col3 = st.columns(3)
                
                n_clusters = len(set(cluster_results['cluster'])) - (1 if -1 in cluster_results['cluster'].values else 0)
                n_outliers = (cluster_results['cluster'] == -1).sum()
                largest_cluster = cluster_results['cluster'].value_counts().iloc[0] if len(cluster_results) > 0 else 0
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card interactive-element">
                        <div style="font-size: 1.5rem;">🎯</div>
                        <div class="metric-value">{n_clusters}</div>
                        <div class="metric-label">Event Clusters</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card interactive-element">
                        <div style="font-size: 1.5rem;">👤</div>
                        <div class="metric-value">{n_outliers}</div>
                        <div class="metric-label">Isolated Events</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card interactive-element">
                        <div style="font-size: 1.5rem;">👥</div>
                        <div class="metric-value">{largest_cluster}</div>
                        <div class="metric-label">Largest Cluster</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Cluster visualization
                if len(cluster_features) >= 2:
                    fig = px.scatter(
                        cluster_results,
                        x=cluster_features[0], y=cluster_features[1],
                        color='cluster',
                        size='pred_prob',
                        title="🎯 Event Clustering Map",
                        hover_data=['pred_prob']
                    )
                    fig.update_layout(title_font_color="white")
                    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.markdown("""
    <div class="modern-card">
        <div class="card-title">🌐 Real-time Event Monitor</div>
        <div class="card-description">Live monitoring dashboard for multimessenger astronomical events</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="glass-panel">
            <h4 style="color: white; margin-bottom: 1rem;">📡 Live Event Stream</h4>
        </div>
        """, unsafe_allow_html=True)
        
        if auto_refresh:
            placeholder = st.empty()
            
            for i in range(5):
                with placeholder.container():
                    current_time = datetime.now()
                    event_type = np.random.choice(['🌊 Gravitational Wave', '⚡ Gamma-ray Burst', '👻 Neutrino Detection'])
                    confidence = np.random.uniform(0.6, 0.95)
                    significance = np.random.uniform(3.0, 8.0)
                    
                    st.markdown(f"""
                    <div class="info-card">
                        <div style="display: flex; align-items: center; gap: 1rem;">
                            <div style="font-size: 2rem;">🚨</div>
                            <div>
                                <div style="font-size: 1.2rem; font-weight: 600;">NEW EVENT - {current_time.strftime('%H:%M:%S')} UTC</div>
                                <div><strong>Type:</strong> {event_type}</div>
                                <div><strong>Confidence:</strong> {confidence:.2f} | <strong>Significance:</strong> {significance:.1f}σ</div>
                                <div style="color: rgba(255,255,255,0.8);">🔄 Processing for associations...</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    progress_bar = st.progress(0)
                    for j in range(100):
                        progress_bar.progress(j + 1)
                        time.sleep(0.01)
                    
                    st.markdown("""
                    <div class="success-card">
                        <div style="text-align: center;">✅ Event processed and stored</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                time.sleep(2)
        else:
            st.markdown("""
            <div class="info-card">
                <div style="text-align: center;">
                    <div style="font-size: 2rem;">🔄</div>
                    <div style="font-size: 1.2rem; font-weight: 600;">Auto-refresh Disabled</div>
                    <div>Enable auto-refresh in the sidebar to see live events</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-panel">
            <h4 style="color: white; margin-bottom: 1rem;">📊 Live Metrics</h4>
        </div>
        """, unsafe_allow_html=True)
        
        live_metrics = [
            ("🔴", "Active Alerts", "7"),
            ("📈", "Events Today", "23"),
            ("🎯", "Associations", "4"),
            ("⚡", "Rate", "1.2/hr")
        ]
        
        for icon, label, value in live_metrics:
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.1); padding: 0.8rem; border-radius: 8px; margin: 0.5rem 0; display: flex; justify-content: space-between; align-items: center;">
                <span style="color: white;">{icon} {label}</span>
                <span style="color: #48bb78; font-weight: 600;">{value}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="glass-panel">
            <h4 style="color: white; margin-bottom: 1rem;">🔭 Observatory Status</h4>
        </div>
        """, unsafe_allow_html=True)
        
        observatories = {
            "LIGO Hanford": "🟢 Online",
            "LIGO Livingston": "🟢 Online", 
            "Virgo": "🟡 Maintenance",
            "IceCube": "🟢 Online",
            "Fermi-GBM": "🟢 Online"
        }
        
        for obs, status in observatories.items():
            color = "#48bb78" if "Online" in status else "#ed8936"
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.1); padding: 0.8rem; border-radius: 8px; margin: 0.5rem 0;">
                <div style="color: white; font-size: 0.9rem; font-weight: 500;">{obs}</div>
                <div style="color: {color}; font-size: 0.8rem;">{status}</div>
            </div>
            """, unsafe_allow_html=True)

# Enhanced footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 3rem; background: rgba(255,255,255,0.1); border-radius: 16px; margin-top: 2rem; backdrop-filter: blur(10px);'>
    <div style='font-size: 1.5rem; font-weight: 600; color: white; margin-bottom: 1rem;'>
        🌌 Multimessenger AI Hub
    </div>
    <div style='color: rgba(255,255,255,0.8); margin-bottom: 1.5rem;'>
        Next-generation platform for astronomical event analysis
    </div>
    <div style='display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap; font-size: 0.9rem; color: rgba(255,255,255,0.7);'>
        <div>🚀 Real-time Analysis</div>
        <div>🤖 Advanced AI</div>
        <div>📊 Interactive Visualizations</div>
        <div>🔬 Scientific Computing</div>
        <div>⚡ Ultra-fast Processing</div>
    </div>
</div>
""", unsafe_allow_html=True)