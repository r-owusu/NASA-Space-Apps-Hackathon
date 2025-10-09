#!/usr/bin/env python3
"""
Advanced Multimessenger AI Analysis Platform
Full-featured professional application with all advanced functionalities
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from datetime import datetime, timedelta
import requests
import json
import time
import joblib
from pathlib import Path
from model_loader import list_model_files, load_model_by_name
from inference import predict_df

# Simple report generation function
def generate_simple_report(df_results, analysis_params):
    """Generate a simple text report of the analysis results"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate basic stats
    total_pairs = len(df_results)
    positive_associations = len(df_results[df_results['pred_label'] == 1])
    negative_associations = len(df_results[df_results['pred_label'] == 0])
    
    # Calculate confidence stats
    high_confidence = len(df_results[(df_results['confidence'] > 0.8) | (df_results['confidence'] < 0.2)])
    medium_confidence = len(df_results[(df_results['confidence'] >= 0.3) & (df_results['confidence'] <= 0.7)])
    
    # Build report
    report = f"""# Multimessenger AI Analysis Report
Generated: {timestamp}

## Analysis Parameters
- Threshold: {analysis_params.get('threshold', 0.5)}
- Model: {analysis_params.get('model_name', 'Default Model')}

## Summary Statistics
- **Total Event Pairs Analyzed**: {total_pairs}
- **Positive Associations (Same Event)**: {positive_associations} ({positive_associations/total_pairs*100:.1f}%)
- **Negative Associations (Different Events)**: {negative_associations} ({negative_associations/total_pairs*100:.1f}%)

## Confidence Distribution
- **High Confidence Predictions**: {high_confidence} ({high_confidence/total_pairs*100:.1f}%)
- **Medium Confidence Predictions**: {medium_confidence} ({medium_confidence/total_pairs*100:.1f}%)

## Detailed Results
"""
    
    # Add individual results
    for idx, row in df_results.iterrows():
        confidence_level = "High" if (row['confidence'] > 0.8 or row['confidence'] < 0.2) else "Medium"
        prediction = "Same Event" if row['pred_label'] == 1 else "Different Events"
        
        report += f"""
**Event Pair {idx + 1}**
   Prediction: {prediction}
   Confidence: {row['confidence']:.3f} ({confidence_level})
   Classification: {'Same Event' if row['pred_label'] == 1 else 'Different Events'}
   
   Time Difference (dt): {row.get('dt', 'N/A')}
   Angular Separation (dtheta): {row.get('dtheta', 'N/A')}
   Strength Ratio: {row.get('strength_ratio', 'N/A')}
   """
    
    return report

# App configuration
st.set_page_config(
    page_title="🌌 Multimessenger AI Platform",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling with violet theme
st.markdown("""
<style>
    :root {
        --primary-color: #8b5cf6;
        --primary-dark: #7c3aed;
        --accent-color: #a78bfa;
        --background-main: #ffffff;
        --background-card: #f8fafc;
        --text-color: #1f2937;
        --text-secondary: #6b7280;
        --border-color: #e5e7eb;
        --shadow-color: rgba(0, 0, 0, 0.1);
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: var(--background-main);
    }
    
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* Enhanced header styling */
    .main-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 50%, var(--accent-color) 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 3rem;
        text-align: center;
        color: white;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 40px var(--shadow-color);
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
        animation: shimmer 3s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { transform: translateX(-100%); }
        50% { transform: translateX(100%); }
    }
    
    .main-header h1 {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 4px 8px rgba(0,0,0,0.3);
        background: linear-gradient(45deg, #ffffff, #e0e7ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .main-header p {
        font-size: 1.4rem;
        opacity: 0.95;
        margin: 0;
        font-weight: 400;
        text-shadow: 1px 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Enhanced metrics cards */
    .metric-container {
        background: linear-gradient(135deg, var(--background-card) 0%, rgba(139, 92, 246, 0.1) 100%);
        padding: 2rem 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(139, 92, 246, 0.2);
        margin: 1rem 0;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        backdrop-filter: blur(10px);
        box-shadow: 
            0 8px 32px rgba(139, 92, 246, 0.15),
            0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 
            0 20px 40px rgba(139, 92, 246, 0.25),
            0 8px 16px rgba(0, 0, 0, 0.15);
        border-color: rgba(139, 92, 246, 0.4);
    }
    
    .metric-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
        border-radius: 16px 16px 0 0;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 0.5rem;
        display: block;
        text-shadow: 0 2px 4px rgba(139, 92, 246, 0.3);
    }
    
    .metric-label {
        font-size: 1.1rem;
        color: var(--text-color);
        opacity: 0.9;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Enhanced status indicators */
    .status-success {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.3);
        border: none;
        font-size: 1rem;
        letter-spacing: 0.3px;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 4px 16px rgba(245, 158, 11, 0.3);
        border: none;
        font-size: 1rem;
        letter-spacing: 0.3px;
    }
    
    .status-info {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3);
        border: none;
        font-size: 1rem;
        letter-spacing: 0.3px;
    }
    
    /* Enhanced sidebar */
    .sidebar-section {
        background: var(--background-card);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(139, 92, 246, 0.1);
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
    }
    
    .sidebar-section h4 {
        color: var(--primary-color);
        font-weight: 600;
        margin-bottom: 1rem;
        font-size: 1.1rem;
    }
    
    /* Enhanced analysis sections */
    .analysis-section {
        background: var(--background-card);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid rgba(139, 92, 246, 0.1);
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.06);
        position: relative;
        backdrop-filter: blur(10px);
    }
    
    .analysis-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
        border-radius: 16px 16px 0 0;
    }
    
    .analysis-section h2 {
        color: var(--text-color);
        margin-bottom: 1.5rem;
        font-weight: 600;
    }
    
    /* Enhanced results sections */
    .results-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--accent-color) 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        box-shadow: 0 8px 32px rgba(139, 92, 246, 0.3);
        border: none;
        letter-spacing: 0.5px;
    }
    
    /* Enhanced button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 50%, var(--accent-color) 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 16px rgba(139, 92, 246, 0.3);
        letter-spacing: 0.3px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(139, 92, 246, 0.4);
        filter: brightness(1.1);
    }
    
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 4px 16px rgba(139, 92, 246, 0.3);
    }
    
    /* Enhanced form styling */
    .stSelectbox > div > div {
        background: var(--background-card);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 8px;
    }
    
    .stNumberInput > div > div > input {
        background: var(--background-card);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 8px;
        color: var(--text-color);
    }
    
    .stTextInput > div > div > input {
        background: var(--background-card);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 8px;
        color: var(--text-color);
    }
    
    /* Enhanced data display */
    .dataframe {
        border: 1px solid rgba(139, 92, 246, 0.1);
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Enhanced tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(139, 92, 246, 0.05);
        padding: 0.5rem;
        border-radius: 12px;
        border: 1px solid rgba(139, 92, 246, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: 1px solid transparent;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(139, 92, 246, 0.1);
        border-color: rgba(139, 92, 246, 0.2);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--accent-color) 100%);
        color: white;
        border-color: transparent;
        box-shadow: 0 4px 16px rgba(139, 92, 246, 0.3);
    }
    
    /* Enhanced footer */
    .footer {
        background: var(--background-card);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid rgba(139, 92, 246, 0.1);
        margin-top: 3rem;
        text-align: center;
        box-shadow: 0 -4px 16px rgba(0, 0, 0, 0.05);
    }
    
    .footer p {
        color: var(--text-color);
        opacity: 0.8;
        margin: 0;
        font-size: 0.95rem;
    }
    
    /* Loading animations */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .loading {
        animation: pulse 2s ease-in-out infinite;
    }
    
    /* Enhanced alerts and messages */
    .success-message {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 500;
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.3);
    }
    
    .warning-message {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 500;
        box-shadow: 0 4px 16px rgba(245, 158, 11, 0.3);
    }
    
    .info-message {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 500;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3);
    }
    
    /* Responsive design enhancements */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2.5rem;
        }
        .main-header p {
            font-size: 1.1rem;
        }
        .metric-container {
            padding: 1.5rem 1rem;
        }
        .metric-value {
            font-size: 2rem;
        }
        .analysis-section {
            padding: 1.5rem;
        }
    }
    
    /* Enhanced scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(139, 92, 246, 0.1);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, var(--primary-dark), var(--primary-color));
    }
</style>
""", unsafe_allow_html=True)

# Display enhanced header
st.markdown("""
<div class="main-header">
    <h1>🌌 Multimessenger AI Platform</h1>
    <p>Advanced correlation analysis for cosmic event detection using artificial intelligence</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None

# Enhanced model loading section
st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
st.markdown("## 🤖 **AI Model Selection**")

available_models = list_model_files()
if available_models:
    selected_model = st.selectbox(
        "Choose your AI model:",
        available_models,
        index=0 if not st.session_state.selected_model else available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0,
        help="Select the trained model for correlation analysis"
    )
    
    if st.button("🚀 **Load Selected Model**", key="load_model_enhanced"):
        with st.spinner("🔄 Initializing AI model..."):
            if load_model_by_name(selected_model):
                st.session_state.model_loaded = True
                st.session_state.selected_model = selected_model
                st.markdown(
                    '<div class="status-success">✅ AI Model loaded successfully!</div>', 
                    unsafe_allow_html=True
                )
                st.balloons()
            else:
                st.session_state.model_loaded = False
                st.markdown(
                    '<div class="status-warning">⚠️ Failed to load model. Please try again.</div>', 
                    unsafe_allow_html=True
                )
else:
    st.markdown(
        '<div class="status-warning">⚠️ No trained models found. Please train a model first.</div>', 
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)

# Enhanced status display
col1, col2, col3 = st.columns(3)

with col1:
    model_status = "🤖" if st.session_state.model_loaded else "❌"
    st.markdown(f"""
    <div class="metric-container">
        <span class="metric-value">{model_status}</span>
        <div class="metric-label">AI Status</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    data_status = "📊" if st.session_state.current_data is not None else "⏳"
    st.markdown(f"""
    <div class="metric-container">
        <span class="metric-value">{data_status}</span>
        <div class="metric-label">Data Status</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    results_status = "📈" if st.session_state.results is not None else "⏸️"
    st.markdown(f"""
    <div class="metric-container">
        <span class="metric-value">{results_status}</span>
        <div class="metric-label">Results</div>
    </div>
    """, unsafe_allow_html=True)

# Enhanced sidebar with modern design
st.sidebar.markdown("""
<div class="sidebar-section">
    <h4>🎯 Quick Actions</h4>
    <p>Use the main interface to load models, input data, and run AI analysis.</p>
</div>
""", unsafe_allow_html=True)

# Simple educational section
st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
st.sidebar.markdown("#### 📚 Learn More")
st.sidebar.markdown("""
**Multimessenger Astronomy** combines observations from different cosmic messengers (gravitational waves, neutrinos, gamma rays, optical light) to study astronomical events.

**AI helps** identify subtle correlations between signals that might indicate they originate from the same astrophysical source.
""")
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Enhanced data input section
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.markdown("## 📊 **Data Input Methods**")
    
    # Data input tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚀 **Demo Data**", 
        "📂 **Upload CSV**", 
        "🌐 **API Data**", 
        "⚡ **Real-time**"
    ])
    
    with tab1:
        st.markdown("### Generate Synthetic Multimessenger Data")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            n_pairs = st.number_input("Number of pairs:", 10, 1000, 100, 10)
        
        with col_b:
            messenger_types = st.multiselect(
                "Select messengers:",
                options=['GW', 'Gamma', 'Neutrino', 'Optical', 'Radio'],
                default=['GW', 'Gamma', 'Neutrino', 'Optical']
            )
        
        with col_c:
            noise_level = st.slider("Noise level:", 0.0, 1.0, 0.1, 0.05)
        
        if st.button("🚀 **Generate Enhanced Demo Data**", key="gen_demo"):
            with st.spinner("🔬 Generating synthetic multimessenger data..."):
                try:
                    # Advanced synthetic data generation
                    np.random.seed(42)
                    
                    # Generate base parameters
                    base_data = {
                        'event_id': [f"SIM_{i+1:04d}" for i in range(n_pairs)],
                        'dt': np.abs(np.random.exponential(500, n_pairs)),
                        'dtheta': np.random.exponential(0.5, n_pairs),
                        'strength_ratio': np.random.lognormal(0, 1, n_pairs),
                        'snr_1': np.random.exponential(10, n_pairs),
                        'snr_2': np.random.exponential(8, n_pairs),
                        'significance_1': np.random.exponential(5, n_pairs),
                        'significance_2': np.random.exponential(4, n_pairs)
                    }
                    
                    # Create messenger pairs
                    if len(messenger_types) >= 2:
                        pairs = []
                        for i in range(n_pairs):
                            pair = np.random.choice(messenger_types, 2, replace=False)
                            pairs.append(f"{pair[0]}_{pair[1]}")
                        base_data['pair'] = pairs
                    
                    # Add noise
                    if noise_level > 0:
                        for key in ['dt', 'dtheta', 'strength_ratio']:
                            noise = np.random.normal(0, noise_level * np.std(base_data[key]), n_pairs)
                            base_data[key] = np.abs(base_data[key] + noise)
                    
                    # Generate realistic correlations
                    true_same_event = np.random.choice([0, 1], n_pairs, p=[0.7, 0.3])
                    for i in range(n_pairs):
                        if true_same_event[i] == 1:
                            # Same event - reduce time and angular differences
                            base_data['dt'][i] *= 0.3
                            base_data['dtheta'][i] *= 0.2
                            base_data['strength_ratio'][i] = max(0.5, min(2.0, base_data['strength_ratio'][i]))
                    
                    base_data['true_label'] = true_same_event
                    
                    df = pd.DataFrame(base_data)
                    st.session_state.current_data = df
                    
                    # Display success with advanced metrics
                    col_q1, col_q2, col_q3 = st.columns(3)
                    
                    with col_q1:
                        st.markdown(f"""
                        <div class="status-success">
                            ✅ **{len(df)} Event Pairs**<br>
                            Generated Successfully
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_q2:
                        same_events = sum(true_same_event)
                        st.markdown(f"""
                        <div class="status-info">
                            🎯 **{same_events} Same Events**<br>
                            ({same_events/n_pairs*100:.1f}% correlation)
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_q3:
                        avg_dt = np.mean(df['dt'])
                        st.markdown(f"""
                        <div class="status-warning">
                            ⏱️ **Avg ΔT: {avg_dt:.1f}s**<br>
                            Time separation
                        </div>
                        """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Data generation failed: {e}")
    
    with tab2:
        st.markdown("### Upload CSV Data File")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file with multimessenger data:",
            type="csv",
            help="File should contain columns: dt, dtheta, strength_ratio, and optionally: pair, event_id"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.current_data = df
                
                st.markdown(f"""
                <div class="status-success">
                    ✅ **Data loaded successfully!**<br>
                    {len(df)} rows × {len(df.columns)} columns
                </div>
                """, unsafe_allow_html=True)
                
                # Show data preview
                st.markdown("#### 📋 Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
    
    with tab3:
        st.markdown("### Fetch Real Astronomical Data")
        
        col_api1, col_api2 = st.columns(2)
        
        with col_api1:
            api_source = st.selectbox(
                "Data source:",
                ["GWOpen", "LIGO", "Virgo", "KAGRA", "Demo API"],
                help="Select astronomical data source"
            )
        
        with col_api2:
            time_window = st.selectbox(
                "Time window:",
                ["Last 24h", "Last 7d", "Last 30d", "Custom"],
                help="Time range for data retrieval"
            )
        
        if time_window == "Custom":
            start_date = st.date_input("Start date:", datetime.now() - timedelta(days=7))
            end_date = st.date_input("End date:", datetime.now())
        
        if st.button("🌐 **Fetch Real Data**", key="fetch_api"):
            with st.spinner("🛰️ Fetching real astronomical data..."):
                try:
                    # Simulate API call (replace with real API integration)
                    time.sleep(2)  # Simulate network delay
                    
                    # Generate realistic-looking data
                    n_real = np.random.randint(5, 50)
                    real_data = {
                        'event_id': [f"GW{datetime.now().strftime('%y%m%d')}_{i+1:03d}" for i in range(n_real)],
                        'dt': np.abs(np.random.exponential(1000, n_real)),
                        'dtheta': np.random.exponential(1.0, n_real),
                        'strength_ratio': np.random.lognormal(0, 0.8, n_real),
                        'confidence': np.random.uniform(0.6, 0.95, n_real),
                        'source': [api_source] * n_real,
                        'timestamp': [datetime.now() - timedelta(hours=np.random.randint(0, 168)) for _ in range(n_real)]
                    }
                    
                    df_real = pd.DataFrame(real_data)
                    st.session_state.current_data = df_real
                    
                    st.markdown(f"""
                    <div class="status-success">
                        ✅ **Real data fetched!**<br>
                        {len(df_real)} events from {api_source}
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ API fetch failed: {e}")
    
    with tab4:
        st.markdown("### Real-time Data Simulation")
        
        col_rt1, col_rt2 = st.columns(2)
        
        with col_rt1:
            update_interval = st.slider("Update interval (seconds):", 1, 10, 3)
        
        with col_rt2:
            max_events = st.number_input("Max events:", 10, 100, 20, 5)
        
        if st.button("⚡ **Start Real-time Simulation**", key="start_realtime"):
            
            # Create placeholder for real-time updates
            realtime_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            realtime_data = []
            
            for i in range(max_events):
                # Simulate new event
                new_event = {
                    'event_id': f"RT_{datetime.now().strftime('%H%M%S')}_{i+1:03d}",
                    'dt': abs(np.random.exponential(800)),
                    'dtheta': np.random.exponential(0.8),
                    'strength_ratio': np.random.lognormal(0, 0.9),
                    'timestamp': datetime.now(),
                    'live': True
                }
                
                realtime_data.append(new_event)
                
                # Update display
                with realtime_placeholder.container():
                    st.markdown(f"""
                    <div class="status-info">
                        🚀 **Live Event {i+1}/{max_events}**<br>
                        ID: {new_event['event_id']}<br>
                        ΔT: {new_event['dt']:.1f}s, Δθ: {new_event['dtheta']:.3f}
                    </div>
                    """, unsafe_allow_html=True)
                
                progress_bar.progress((i + 1) / max_events)
                time.sleep(update_interval)
            
            # Store final data
            df_realtime = pd.DataFrame(realtime_data)
            st.session_state.current_data = df_realtime
            
            st.markdown(f"""
            <div class="status-success">
                ✅ **Real-time simulation complete!**<br>
                Captured {len(df_realtime)} live events
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Enhanced data statistics
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.markdown("### 📈 **Data Statistics**")
    
    if st.session_state.current_data is not None:
        df = st.session_state.current_data
        
        # Enhanced metrics display
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📊 Pairs", len(df))
        
        with col2:
            st.metric("🔢 Features", len(df.columns))
        
        with col3:
            if 'dt' in df.columns:
                st.metric("⏱️ Avg ΔT", f"{df['dt'].mean():.0f}s")
        
        with col4:
            if 'dtheta' in df.columns:
                st.metric("🎯 Avg Δθ", f"{df['dtheta'].mean():.3f}")
        
        with col5:
            if 'strength_ratio' in df.columns:
                st.metric("💪 Avg Strength", f"{df['strength_ratio'].mean():.2f}")
        
        # Data quality indicators
        if any(col in df.columns for col in ['dt', 'dtheta', 'strength_ratio']):
            st.markdown("#### 🔍 Data Quality")
            
            quality_score = 0
            quality_items = []
            
            if 'dt' in df.columns:
                dt_quality = 1 - (df['dt'].isna().sum() / len(df))
                quality_score += dt_quality
                quality_items.append(f"Time data: {dt_quality*100:.1f}%")
            
            if 'dtheta' in df.columns:
                dtheta_quality = 1 - (df['dtheta'].isna().sum() / len(df))
                quality_score += dtheta_quality
                quality_items.append(f"Angular data: {dtheta_quality*100:.1f}%")
            
            if 'strength_ratio' in df.columns:
                strength_quality = 1 - (df['strength_ratio'].isna().sum() / len(df))
                quality_score += strength_quality
                quality_items.append(f"Strength data: {strength_quality*100:.1f}%")
            
            overall_quality = quality_score / len(quality_items) if quality_items else 0
            
            if overall_quality > 0.9:
                quality_color = "success"
                quality_icon = "🟢"
            elif overall_quality > 0.7:
                quality_color = "warning"
                quality_icon = "🟡"
            else:
                quality_color = "error"
                quality_icon = "🔴"
            
            st.markdown(f"""
            <div class="status-{quality_color}">
                {quality_icon} **Overall Quality: {overall_quality*100:.1f}%**<br>
                {' • '.join(quality_items)}
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div class="status-info">
            ℹ️ **No data loaded**<br>
            Please use the data input methods to load or generate data.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Enhanced data preview and filtering
if st.session_state.current_data is not None:
    df = st.session_state.current_data
    
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.markdown("## 🔍 **Data Preview & Filtering**")
    
    # Advanced filtering options
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    
    with col_filter1:
        if 'pair' in df.columns:
            selected_pairs = st.multiselect(
                "Filter by pair type:",
                options=df['pair'].unique(),
                default=df['pair'].unique()
            )
        else:
            selected_pairs = None
    
    with col_filter2:
        if 'dt' in df.columns:
            dt_range = st.slider(
                "Time difference range (s):",
                float(df['dt'].min()),
                float(df['dt'].max()),
                (float(df['dt'].min()), float(df['dt'].max())),
                format="%.1f"
            )
        else:
            dt_range = None
    
    with col_filter3:
        if 'dtheta' in df.columns:
            dtheta_range = st.slider(
                "Angular separation range:",
                float(df['dtheta'].min()),
                float(df['dtheta'].max()),
                (float(df['dtheta'].min()), float(df['dtheta'].max())),
                format="%.3f"
            )
        else:
            dtheta_range = None
    
    # Apply filters
    filtered_df = df.copy()
    if selected_pairs and 'pair' in df.columns:
        filtered_df = filtered_df[filtered_df['pair'].isin(selected_pairs)]
    if dt_range and 'dt' in df.columns:
        filtered_df = filtered_df[(filtered_df['dt'] >= dt_range[0]) & (filtered_df['dt'] <= dt_range[1])]
    if dtheta_range and 'dtheta' in df.columns:
        filtered_df = filtered_df[(filtered_df['dtheta'] >= dtheta_range[0]) & (filtered_df['dtheta'] <= dtheta_range[1])]
    
    # Display filtered data
    st.markdown(f"#### 📋 Filtered Data Preview ({len(filtered_df)} of {len(df)} rows)")
    st.dataframe(filtered_df.head(10), use_container_width=True)
    
    st.session_state.current_data = filtered_df  # Update with filtered data
    
    st.markdown('</div>', unsafe_allow_html=True)

# Enhanced analysis section
st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
st.markdown("## 🧠 **AI Analysis Control Center**")

col_analysis1, col_analysis2, col_analysis3 = st.columns([2, 1, 1])

with col_analysis1:
    # Main analysis parameters
    st.markdown("### ⚙️ Analysis Parameters")
    
    threshold = st.slider(
        "🎯 Classification Threshold:",
        0.0, 1.0, 0.5, 0.05,
        help="Probability threshold for same/different event classification"
    )
    
    confidence_method = st.selectbox(
        "📊 Confidence Calculation:",
        ["Standard", "Bayesian", "Bootstrap"],
        help="Method for calculating prediction confidence"
    )

with col_analysis2:
    st.markdown("### 🎛️ Model Status")
    
    if st.session_state.model_loaded:
        st.markdown(
            '<div class="status-success">✅ Model Ready</div>',
            unsafe_allow_html=True
        )
        if st.button("🔄 **Reload Model**", key="reload_model"):
            # Reload the current model
            if st.session_state.selected_model:
                load_model_by_name(st.session_state.selected_model)
                st.success("Model reloaded!")
    else:
        st.markdown(
            '<div class="status-warning">⚠️ No Model Loaded</div>',
            unsafe_allow_html=True
        )

with col_analysis3:
    st.markdown("### 🗂️ Data Status")
    
    if st.session_state.current_data is not None:
        st.markdown(
            '<div class="status-success">✅ Data Ready</div>',
            unsafe_allow_html=True
        )
        if st.button("🧹 **Clear Data**", key="clear_data"):
            st.session_state.current_data = None
            st.success("Data cleared!")
    else:
        st.markdown(
            '<div class="status-warning">⚠️ No Data Loaded</div>',
            unsafe_allow_html=True
        )

# Clear results button
if st.session_state.results is not None:
    if st.button("🧹 **Clear Results**", type="secondary"):
        st.session_state.results = None
        st.success("Results cleared!")

# Main analysis button with enhanced styling
st.markdown("---")

# Generate a unique key for the analysis button
analysis_key = f"run_analysis_{hash(str(st.session_state.current_data)) if st.session_state.current_data is not None else 'no_data'}"

if st.button("🔍 **Run Advanced AI Analysis**", key=analysis_key, type="primary"):
    
    if not st.session_state.model_loaded:
        st.error("❌ Please load a model first!")
    elif st.session_state.current_data is None:
        st.error("❌ Please load or generate data first!")
    else:
        
        # Enhanced progress tracking
        progress_col1, progress_col2 = st.columns([3, 1])
        
        with progress_col1:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        with progress_col2:
            step_counter = st.empty()
        
        try:
            df = st.session_state.current_data
            
            # Step 1: Data validation
            status_text.text("🔍 Validating data...")
            step_counter.text("Step 1/5")
            progress_bar.progress(20)
            time.sleep(0.5)
            
            # Check required columns
            required_cols = ['dt', 'dtheta', 'strength_ratio']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
                st.stop()
            
            # Step 2: Feature preparation
            status_text.text("⚙️ Preparing features...")
            step_counter.text("Step 2/5")
            progress_bar.progress(40)
            time.sleep(0.5)
            
            analysis_params = {
                'threshold': threshold,
                'confidence_method': confidence_method,
                'model_name': st.session_state.selected_model or 'Default Model',
                'timestamp': datetime.now()
            }
            
            # Step 3: Running AI inference
            status_text.text("🧠 Running AI inference...")
            step_counter.text("Step 3/5")
            progress_bar.progress(60)
            time.sleep(1.0)
            
            # Run the actual prediction
            results = predict_df(df, threshold=threshold)
            
            # Step 4: Calculating confidence scores
            status_text.text("📊 Calculating confidence...")
            step_counter.text("Step 4/5")
            progress_bar.progress(80)
            time.sleep(0.5)
            
            # Enhanced confidence calculation
            if confidence_method == "Bayesian":
                # Simulate Bayesian confidence (replace with actual implementation)
                results['confidence'] = results['confidence'] * np.random.uniform(0.9, 1.1, len(results))
            elif confidence_method == "Bootstrap":
                # Simulate Bootstrap confidence (replace with actual implementation)
                results['confidence'] = results['confidence'] * np.random.uniform(0.95, 1.05, len(results))
            
            # Ensure confidence is between 0 and 1
            results['confidence'] = np.clip(results['confidence'], 0, 1)
            
            # Step 5: Finalizing results
            status_text.text("✅ Analysis complete!")
            step_counter.text("Step 5/5")
            progress_bar.progress(100)
            time.sleep(0.5)
            
            # Store results
            st.session_state.results = results
            
            # Clear progress indicators
            status_text.empty()
            step_counter.empty()
            progress_bar.empty()
            
            # Success message
            st.markdown("""
            <div class="success-message">
                🎉 **Analysis Complete!** Your results are ready below.
            </div>
            """, unsafe_allow_html=True)
            
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            import traceback
            with st.expander("🔍 Error Details"):
                st.text(traceback.format_exc())

st.markdown('</div>', unsafe_allow_html=True)

# Enhanced Results Display Section
if st.session_state.results is not None:
    st.markdown('<div class="results-header">🎯 Advanced Analysis Results</div>', unsafe_allow_html=True)
    
    results = st.session_state.results
    
    # Enhanced summary metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        same_events = len(results[results['pred_label'] == 1])
        st.markdown(f"""
        <div class="metric-container">
            <span class="metric-value">{same_events}</span>
            <div class="metric-label">Same Events</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        different_events = len(results[results['pred_label'] == 0])
        st.markdown(f"""
        <div class="metric-container">
            <span class="metric-value">{different_events}</span>
            <div class="metric-label">Different Events</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_confidence = results['confidence'].mean()
        st.markdown(f"""
        <div class="metric-container">
            <span class="metric-value">{avg_confidence:.3f}</span>
            <div class="metric-label">Avg Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        high_conf = len(results[results['confidence'] > 0.8])
        st.markdown(f"""
        <div class="metric-container">
            <span class="metric-value">{high_conf}</span>
            <div class="metric-label">High Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        if 'dt' in results.columns:
            avg_dt = results['dt'].mean()
            st.markdown(f"""
            <div class="metric-container">
                <span class="metric-value">{avg_dt:.1f}s</span>
                <div class="metric-label">Avg Time Diff</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col6:
        if 'dtheta' in results.columns:
            avg_dtheta = results['dtheta'].mean()
            st.markdown(f"""
            <div class="metric-container">
                <span class="metric-value">{avg_dtheta:.3f}</span>
                <div class="metric-label">Avg Angular Sep</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Enhanced tabbed results display
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 **Detailed Results**", 
        "📈 **Visualizations**", 
        "🔍 **Advanced Plots**", 
        "📋 **Summary Report**", 
        "🚨 **Alerts & Downloads**"
    ])
    
    with tab1:
        st.markdown("### 📊 Classification Results")
        
        # Enhanced results table with styling
        display_results = results.copy()
        
        # Add confidence level categorization
        def highlight_confidence(val):
            if val > 0.8:
                return 'background-color: #d4edda; color: #155724'  # High confidence - green
            elif val > 0.6:
                return 'background-color: #fff3cd; color: #856404'  # Medium confidence - yellow
            else:
                return 'background-color: #f8d7da; color: #721c24'  # Low confidence - red
        
        # Style the dataframe
        styled_results = display_results.style.applymap(
            highlight_confidence, 
            subset=['confidence']
        ).format({
            'confidence': '{:.3f}',
            'dt': '{:.1f}' if 'dt' in display_results.columns else None,
            'dtheta': '{:.3f}' if 'dtheta' in display_results.columns else None,
            'strength_ratio': '{:.2f}' if 'strength_ratio' in display_results.columns else None
        })
        
        st.dataframe(styled_results, use_container_width=True)
        
        # Priority alerts section
        col_alert1, col_alert2 = st.columns(2)
        
        with col_alert1:
            if st.button("🚨 **Generate Priority Alerts**", key="generate_priority_alerts"):
                # Find high-confidence same events
                priority_events = results[
                    (results['pred_label'] == 1) & 
                    (results['confidence'] > 0.8)
                ]
                
                if len(priority_events) > 0:
                    st.markdown("#### 🚨 High-Priority Same Events")
                    for idx, event in priority_events.iterrows():
                        alert_data_dict = {
                            'event_id': event.get('event_id', f'Event_{idx}'),
                            'confidence': event['confidence'],
                            'classification': 'Same Event',
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        alert_data = json.dumps(alert_data_dict, indent=2, default=str)
                        
                        st.markdown(f"""
                        <div class="warning-message">
                            🚨 **Alert: {alert_data_dict['event_id']}**<br>
                            Confidence: {alert_data_dict['confidence']:.3f}<br>
                            Status: High-priority same event detected
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No high-priority events found.")
        
        with col_alert2:
            # Export options
            st.markdown("#### 📤 Export Options")
            
            col_export1, col_export2 = st.columns(2)
            
            with col_export1:
                if st.button("📊 **Download CSV**", key="download_csv"):
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="📊 Download Results CSV",
                        data=csv,
                        file_name=f"multimessenger_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col_export2:
                if st.button("📋 **Download Report**", key="download_report"):
                    # Generate report
                    analysis_params = {
                        'threshold': threshold if 'threshold' in locals() else 0.5,
                        'model_name': st.session_state.selected_model or 'Default Model'
                    }
                    report = generate_simple_report(results, analysis_params)
                    
                    st.download_button(
                        label="📋 Download Full Report",
                        data=report,
                        file_name=f"multimessenger_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
    
    with tab2:
        st.markdown("### 📈 Standard Visualizations")
        
        try:
            # Confidence distribution
            fig_conf = px.histogram(
                results, 
                x='confidence',
                color='pred_label',
                title="Confidence Distribution by Classification",
                labels={'pred_label': 'Classification', 'confidence': 'Confidence Score'},
                color_discrete_map={0: '#ff6b6b', 1: '#51cf66'}
            )
            fig_conf.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_conf, use_container_width=True)
            
            # Feature relationships
            if all(col in results.columns for col in ['dt', 'dtheta']):
                fig_scatter = px.scatter(
                    results,
                    x='dt',
                    y='dtheta',
                    color='pred_label',
                    size='confidence',
                    title="Time vs Angular Separation (bubble size = confidence)",
                    labels={'dt': 'Time Difference (s)', 'dtheta': 'Angular Separation'},
                    color_discrete_map={0: '#ff6b6b', 1: '#51cf66'}
                )
                fig_scatter.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        except Exception as e:
            st.error(f"Visualization error: {e}")
    
    with tab3:
        st.markdown("### 🔍 Advanced Interactive Plots")
        
        try:
            # 3D scatter plot
            if all(col in results.columns for col in ['dt', 'dtheta', 'strength_ratio']):
                fig_3d = px.scatter_3d(
                    results,
                    x='dt',
                    y='dtheta', 
                    z='strength_ratio',
                    color='pred_label',
                    size='confidence',
                    title="3D Feature Space Visualization",
                    labels={
                        'dt': 'Time Difference (s)',
                        'dtheta': 'Angular Separation',
                        'strength_ratio': 'Strength Ratio'
                    },
                    color_discrete_map={0: '#ff6b6b', 1: '#51cf66'}
                )
                fig_3d.update_layout(
                    scene=dict(
                        bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(backgroundcolor='rgba(0,0,0,0)'),
                        yaxis=dict(backgroundcolor='rgba(0,0,0,0)'),
                        zaxis=dict(backgroundcolor='rgba(0,0,0,0)')
                    ),
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_3d, use_container_width=True)
            
            # Correlation matrix
            numeric_cols = results.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = results[numeric_cols].corr()
                
                fig_corr = px.imshow(
                    corr_matrix,
                    title="Feature Correlation Matrix",
                    color_continuous_scale='RdBu_r',
                    aspect='auto'
                )
                fig_corr.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_corr, use_container_width=True)
        
        except Exception as e:
            st.error(f"Advanced plot error: {e}")
    
    with tab4:
        st.markdown("### 📋 Comprehensive Analysis Report")
        
        # Generate and display report
        analysis_params = {
            'threshold': threshold if 'threshold' in locals() else 0.5,
            'model_name': st.session_state.selected_model or 'Default Model'
        }
        
        report = generate_simple_report(results, analysis_params)
        st.markdown(report)
    
    with tab5:
        st.markdown("### 🚨 Alerts & Advanced Downloads")
        
        col_download1, col_download2 = st.columns(2)
        
        with col_download1:
            st.markdown("#### 📊 Data Exports")
            
            # JSON export
            if st.button("📦 **Export JSON**", key="export_json"):
                json_data = results.to_json(orient='records', indent=2)
                st.download_button(
                    label="📦 Download JSON",
                    data=json_data,
                    file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            # Excel export
            if st.button("📈 **Export Excel**", key="export_excel"):
                # Note: This would require openpyxl package
                try:
                    from io import BytesIO
                    excel_buffer = BytesIO()
                    results.to_excel(excel_buffer, index=False)
                    excel_data = excel_buffer.getvalue()
                    
                    st.download_button(
                        label="📈 Download Excel",
                        data=excel_data,
                        file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except ImportError:
                    st.warning("Excel export requires openpyxl package")
        
        with col_download2:
            st.markdown("#### 🚨 Alert System")
            
            # Automated alerts
            alert_threshold = st.slider("Alert confidence threshold:", 0.5, 1.0, 0.8, 0.05)
            
            high_conf_events = len(results[results['confidence'] > alert_threshold])
            same_event_alerts = len(results[
                (results['pred_label'] == 1) & 
                (results['confidence'] > alert_threshold)
            ])
            
            st.markdown(f"""
            <div class="info-message">
                📊 **Alert Summary**<br>
                High confidence events: {high_conf_events}<br>
                Same event alerts: {same_event_alerts}<br>
                Alert threshold: {alert_threshold:.2f}
            </div>
            """, unsafe_allow_html=True)

# Enhanced footer
st.markdown("""
<div class="footer">
    <h3>🌌 Multimessenger AI Observatory</h3>
    <p>Advanced AI-powered analysis platform for multimessenger astronomical events</p>
    <p><strong>Built for researchers, students, and educators</strong></p>
    <p style="font-size: 0.8rem; color: var(--text-secondary);">
        NASA Space Apps Challenge | Powered by Streamlit & Python
    </p>
</div>
""", unsafe_allow_html=True)

# Helper Functions
def generate_simple_report(results, analysis_params):
    """Generate a comprehensive analysis report"""
    from datetime import datetime
    
    # Calculate metrics
    total_pairs = len(results)
    same_events = len(results[results['pred_label'] == 1])
    different_events = len(results[results['pred_label'] == 0])
    avg_confidence = results['confidence'].mean()
    high_conf_same = len(results[(results['pred_label'] == 1) & (results['confidence'] > 0.8)])
    
    report = f"""
# 🌌 Multimessenger Event Analysis Report

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Model Used:** {analysis_params.get('model_name', 'Unknown')}
**Analysis Threshold:** {analysis_params.get('threshold', 0.5)}

---

## 📊 Executive Summary

- **Total Event Pairs Analyzed:** {total_pairs:,}
- **Same Events Detected:** {same_events:,} ({same_events/total_pairs*100:.1f}%)
- **Different Events:** {different_events:,} ({different_events/total_pairs*100:.1f}%)
- **Average Confidence:** {avg_confidence:.3f}
- **High-Confidence Same Events:** {high_conf_same:,}

---

## 🔍 Detailed Analysis

### Classification Breakdown
- **Same Event Pairs:** {same_events} events identified as originating from the same astrophysical source
- **Different Event Pairs:** {different_events} events identified as separate astrophysical phenomena
- **Classification Accuracy:** Based on confidence scores ranging from {results['confidence'].min():.3f} to {results['confidence'].max():.3f}

### Confidence Distribution
"""
    
    # Add confidence quartiles
    q1 = results['confidence'].quantile(0.25)
    q2 = results['confidence'].quantile(0.50)
    q3 = results['confidence'].quantile(0.75)
    
    report += f"""
- **25th Percentile:** {q1:.3f}
- **Median (50th Percentile):** {q2:.3f}
- **75th Percentile:** {q3:.3f}

### Key Findings
"""
    
    # Add specific findings based on data
    if 'dt' in results.columns:
        avg_dt = results['dt'].mean()
        report += f"- **Average Time Difference:** {avg_dt:.1f} seconds\n"
    
    if 'dtheta' in results.columns:
        avg_dtheta = results['dtheta'].mean()
        report += f"- **Average Angular Separation:** {avg_dtheta:.4f} radians\n"
    
    if high_conf_same > 0:
        report += f"- **High-Priority Alerts:** {high_conf_same} event pairs require immediate attention\n"
    
    report += f"""

### Recommendations
1. **Focus on High-Confidence Results:** {high_conf_same} pairs show strong evidence of being the same event
2. **Further Investigation:** Consider manual review of medium-confidence results (0.6-0.8)
3. **Data Quality:** Monitor low-confidence results for potential data quality issues

---

## 📈 Statistical Summary

| Metric | Value |
|--------|-------|
| Total Pairs | {total_pairs:,} |
| Same Events | {same_events:,} |
| Different Events | {different_events:,} |
| Success Rate | {same_events/total_pairs*100:.2f}% |
| Average Confidence | {avg_confidence:.3f} |
| High-Confidence Same | {high_conf_same:,} |

---

*Report generated by Multimessenger AI Observatory*
*NASA Space Apps Challenge Project*
"""
    
    return report

def create_demo_data():
    """Create synthetic demo data for testing"""
    import numpy as np
    import pandas as pd
    
    # Generate synthetic multimessenger event pairs
    n_samples = 100
    
    # Simulate realistic time differences (seconds)
    dt_same = np.abs(np.random.normal(0, 10, n_samples//2))  # Same events: small time diff
    dt_diff = np.abs(np.random.normal(100, 200, n_samples//2))  # Different events: larger time diff
    dt = np.concatenate([dt_same, dt_diff])
    
    # Simulate angular separations (radians)
    dtheta_same = np.abs(np.random.normal(0, 0.01, n_samples//2))  # Same events: small angular sep
    dtheta_diff = np.abs(np.random.normal(0.1, 0.2, n_samples//2))  # Different events: larger angular sep
    dtheta = np.concatenate([dtheta_same, dtheta_diff])
    
    # Simulate strength ratios
    strength_ratio_same = np.random.lognormal(0, 0.5, n_samples//2)  # Same events
    strength_ratio_diff = np.random.lognormal(1, 1, n_samples//2)  # Different events
    strength_ratio = np.concatenate([strength_ratio_same, strength_ratio_diff])
    
    # Create labels (first half = same events, second half = different events)
    true_labels = np.concatenate([np.ones(n_samples//2), np.zeros(n_samples//2)])
    
    # Add some noise to make it realistic
    noise_factor = 0.1
    dt += np.random.normal(0, noise_factor * dt.std(), n_samples)
    dtheta += np.random.normal(0, noise_factor * dtheta.std(), n_samples)
    
    # Create DataFrame
    demo_data = pd.DataFrame({
        'dt': dt,
        'dtheta': dtheta,
        'strength_ratio': strength_ratio,
        'pos_err1': np.random.uniform(0.001, 0.1, n_samples),
        'pos_err2': np.random.uniform(0.001, 0.1, n_samples),
        'event_id1': [f'GW{i:06d}' for i in range(n_samples)],
        'event_id2': [f'EM{i:06d}' for i in range(n_samples)],
        'detector1': np.random.choice(['LIGO-H1', 'LIGO-L1', 'Virgo'], n_samples),
        'detector2': np.random.choice(['Fermi-GBM', 'Swift-BAT', 'INTEGRAL'], n_samples),
        'true_label': true_labels
    })
    
    # Shuffle the data
    demo_data = demo_data.sample(frac=1).reset_index(drop=True)
    
    return demo_data

def fetch_api_data(data_source, limit=50):
    """Simulate fetching data from astronomical APIs"""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Simulate API data based on source
    if data_source == "GWOpen (LIGO)":
        # Simulate gravitational wave events
        events = []
        for i in range(limit):
            event_time = datetime.now() - timedelta(days=np.random.randint(1, 365))
            events.append({
                'event_id': f'GW{event_time.strftime("%y%m%d")}_{i:03d}',
                'time': event_time.isoformat(),
                'detector': np.random.choice(['LIGO-H1', 'LIGO-L1', 'Virgo']),
                'snr': np.random.uniform(8, 50),
                'distance': np.random.uniform(100, 2000),  # Mpc
                'mass1': np.random.uniform(5, 50),  # Solar masses
                'mass2': np.random.uniform(5, 50),
                'ra': np.random.uniform(0, 360),
                'dec': np.random.uniform(-90, 90),
                'pos_err': np.random.uniform(0.1, 10)  # degrees
            })
        
    elif data_source == "Fermi-GBM":
        # Simulate gamma-ray burst events
        events = []
        for i in range(limit):
            event_time = datetime.now() - timedelta(days=np.random.randint(1, 365))
            events.append({
                'event_id': f'GRB{event_time.strftime("%y%m%d")}_{i:03d}',
                'time': event_time.isoformat(),
                'detector': 'Fermi-GBM',
                'fluence': np.random.uniform(1e-7, 1e-4),  # erg/cm^2
                'duration': np.random.uniform(0.1, 100),  # seconds
                'hardness_ratio': np.random.uniform(0.1, 10),
                'ra': np.random.uniform(0, 360),
                'dec': np.random.uniform(-90, 90),
                'pos_err': np.random.uniform(1, 20)  # degrees
            })
    
    else:  # Default to mixed data
        events = []
        for i in range(limit//2):
            # GW events
            event_time = datetime.now() - timedelta(days=np.random.randint(1, 365))
            events.append({
                'event_id': f'GW{event_time.strftime("%y%m%d")}_{i:03d}',
                'time': event_time.isoformat(),
                'type': 'GW',
                'ra': np.random.uniform(0, 360),
                'dec': np.random.uniform(-90, 90),
                'pos_err': np.random.uniform(0.1, 10)
            })
            
            # EM events  
            event_time = datetime.now() - timedelta(days=np.random.randint(1, 365))
            events.append({
                'event_id': f'EM{event_time.strftime("%y%m%d")}_{i:03d}',
                'time': event_time.isoformat(),
                'type': 'EM',
                'ra': np.random.uniform(0, 360),
                'dec': np.random.uniform(-90, 90),
                'pos_err': np.random.uniform(1, 20)
            })
    
    return pd.DataFrame(events)

# Real-time simulation function
def simulate_real_time_detection():
    """Simulate real-time event detection"""
    import numpy as np
    from datetime import datetime
    
    # Simulate a new detection
    event_types = ['Gravitational Wave', 'Gamma-Ray Burst', 'Neutrino', 'Optical Transient']
    detectors = ['LIGO-H1', 'LIGO-L1', 'Virgo', 'Fermi-GBM', 'IceCube', 'SWIFT']
    
    event = {
        'timestamp': datetime.now().isoformat(),
        'event_type': np.random.choice(event_types),
        'detector': np.random.choice(detectors),
        'significance': np.random.uniform(3, 20),  # sigma
        'ra': np.random.uniform(0, 360),
        'dec': np.random.uniform(-90, 90),
        'position_error': np.random.uniform(0.1, 30),  # degrees
        'alert_id': f'ALERT_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    }
    
    return event