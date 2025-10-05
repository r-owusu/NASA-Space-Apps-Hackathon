#!/usr/bin/env python3
"""
Enhanced Multimessenger AI Analysis Platform
Advanced UI with improved visualizations and data input methods
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
from model_loader import list_model_files, load_model_by_name
from inference import predict_df

# Page configuration
st.set_page_config(
    page_title="Multimessenger AI Observatory",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --background-dark: #0e1117;
        --background-light: #262730;
        --accent-color: #00d4ff;
        --success-color: #00c851;
        --warning-color: #ffbb33;
        --danger-color: #ff4444;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #1f77b4 0%, #00d4ff 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Enhanced metrics */
    .metric-container {
        background: linear-gradient(135deg, rgba(31, 119, 180, 0.1) 0%, rgba(0, 212, 255, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid rgba(31, 119, 180, 0.3);
        margin: 0.5rem 0;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.2);
    }
    
    /* Status indicators */
    .status-badge {
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.2rem;
    }
    
    .status-success { background-color: var(--success-color); color: white; }
    .status-warning { background-color: var(--warning-color); color: black; }
    .status-danger { background-color: var(--danger-color); color: white; }
    .status-info { background-color: var(--accent-color); color: black; }
    
    /* Data input cards */
    .input-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 1rem 0;
    }
    
    /* Educational tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
        color: var(--accent-color);
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 300px;
        background-color: rgba(0, 0, 0, 0.9);
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -150px;
        font-size: 0.9rem;
        line-height: 1.3;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
    }
    
    /* Analysis section */
    .analysis-section {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.05) 0%, rgba(31, 119, 180, 0.05) 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid rgba(0, 212, 255, 0.2);
        margin: 2rem 0;
    }
    
    /* Results styling */
    .results-header {
        background: linear-gradient(90deg, #00c851 0%, #00d4ff 100%);
        padding: 1rem 2rem;
        border-radius: 10px 10px 0 0;
        color: white;
        font-weight: 600;
        font-size: 1.2rem;
    }
    
    .confidence-high { color: #00c851; font-weight: bold; }
    .confidence-medium { color: #ffbb33; font-weight: bold; }
    .confidence-low { color: #ff4444; font-weight: bold; }
    
    /* Interactive elements */
    .stButton > button {
        background: linear-gradient(90deg, #1f77b4 0%, #00d4ff 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(31, 119, 180, 0.4);
    }
    
    /* Sidebar enhancements */
    .sidebar-section {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid var(--accent-color);
    }
</style>
""", unsafe_allow_html=True)

# Enhanced header
st.markdown("""
<div class="main-header">
    <h1>🌌 Multimessenger AI Observatory</h1>
    <p>Advanced AI-powered analysis of multimessenger astronomical events for research and education</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state with more variables
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'real_time_mode' not in st.session_state:
    st.session_state.real_time_mode = False
if 'api_data_cache' not in st.session_state:
    st.session_state.api_data_cache = {}

# Enhanced sidebar with sections
st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
st.sidebar.markdown("### 🎛️ **Analysis Controls**")

# Model selection with enhanced info
model_files = list_model_files()
st.sidebar.subheader("🤖 AI Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose trained model:",
    ["(none)"] + model_files,
    key="model_selector",
    help="Select the AI model for multimessenger event analysis"
)

# Load model with enhanced feedback
model = None
scaler = None
metadata = None

if model_choice and model_choice != "(none)":
    try:
        model, scaler, metadata = load_model_by_name(model_choice)
        st.session_state.model_loaded = True
        
        st.sidebar.success(f"✅ **Model Active**: {model_choice}")
        
        if metadata:
            st.sidebar.markdown(f"""
            **📊 Model Performance:**
            - **Algorithm**: {metadata.get('best_model', 'Unknown')}
            - **AUC Score**: {metadata.get('best_auc', 'N/A'):.3f}
            - **Training Date**: {datetime.fromtimestamp(metadata.get('date', 0)).strftime('%Y-%m-%d') if metadata.get('date') else 'Unknown'}
            """)
            
    except Exception as e:
        st.sidebar.error(f"❌ **Model Load Error**: {str(e)[:50]}...")
        st.session_state.model_loaded = False
else:
    st.session_state.model_loaded = False
    st.sidebar.info("ℹ️ Please select an AI model to begin analysis")

st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Analysis parameters
st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
st.sidebar.markdown("### ⚙️ **Analysis Parameters**")

threshold = st.sidebar.slider(
    "🎯 Association Confidence Threshold", 
    0.0, 1.0, 0.5, 0.05,
    help="Minimum confidence score for positive multimessenger associations"
)

# Advanced options
with st.sidebar.expander("🔧 Advanced Options"):
    show_debug = st.checkbox("Show debug information")
    auto_refresh = st.checkbox("Auto-refresh results")
    scientific_notation = st.checkbox("Use scientific notation")

st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Educational section
st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
st.sidebar.markdown("### 📚 **Learn More**")
st.sidebar.markdown("""
<div class="tooltip">**What is Multimessenger Astronomy?** ℹ️
<span class="tooltiptext">
Multimessenger astronomy combines observations from different cosmic messengers (gravitational waves, neutrinos, gamma rays, optical light) to study astronomical events like neutron star mergers and black hole formations.
</span>
</div>

<div class="tooltip">**How does AI help?** 🤖
<span class="tooltiptext">
Machine learning algorithms can identify subtle correlations between different messenger signals that might indicate they originate from the same astrophysical source, even when individual signals are weak.
</span>
</div>
""", unsafe_allow_html=True)
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
        "📂 **File Upload**", 
        "🌐 **API Integration**", 
        "⚡ **Real-time Input**"
    ])
    
    with tab1:
        st.markdown("### Generate Synthetic Multimessenger Data")
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            n_events = st.number_input("Number of events", 10, 1000, 50)
        with col_b:
            event_types = st.multiselect(
                "Messenger types", 
                ['GW', 'Gamma', 'Neutrino', 'Optical', 'Radio'],
                default=['GW', 'Gamma', 'Neutrino', 'Optical']
            )
        with col_c:
            time_window = st.number_input("Time window (hours)", 0.1, 48.0, 24.0)
        
        if st.button("🚀 **Generate Enhanced Demo Data**", key="gen_demo"):
            with st.spinner("Generating realistic multimessenger data..."):
                # Enhanced demo data generation
                np.random.seed(42)
                
                # More realistic parameter distributions
                data = {
                    'dt': np.random.exponential(1.0, n_events),
                    'dtheta': np.random.exponential(0.1, n_events),
                    'strength_ratio': np.random.lognormal(0, 1, n_events),
                    'ra': np.random.uniform(0, 360, n_events),
                    'dec': np.random.uniform(-90, 90, n_events),
                    'energy': np.random.lognormal(15, 2, n_events),  # Energy in eV
                    'snr': np.random.exponential(5, n_events),  # Signal-to-noise ratio
                    'distance': np.random.exponential(100, n_events),  # Distance in Mpc
                    'timestamp': [
                        datetime.now() - timedelta(hours=np.random.uniform(0, time_window))
                        for _ in range(n_events)
                    ]
                }
                
                # Ensure we have the requested messenger types
                if len(event_types) >= 2:
                    data['m1'] = np.random.choice(event_types, n_events)
                    data['m2'] = np.random.choice(event_types, n_events)
                    # Ensure m1 != m2
                    mask = data['m1'] == data['m2']
                    for i in np.where(mask)[0]:
                        options = [m for m in event_types if m != data['m1'][i]]
                        if options:
                            data['m2'][i] = np.random.choice(options)
                else:
                    st.error("Please select at least 2 messenger types")
                    st.stop()
                
                df = pd.DataFrame(data)
                st.session_state.current_data = df
                
                st.success(f"✅ Generated {len(df)} multimessenger events with {len(event_types)} messenger types!")
                
                # Quick preview
                st.markdown("**Preview:**")
                st.dataframe(df.head(), width="stretch")
    
    with tab2:
        st.markdown("### Upload CSV Data File")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Required columns: dt, dtheta, strength_ratio. Optional: m1, m2, ra, dec"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.current_data = df
                
                # Validate required columns
                required_cols = ['dt', 'dtheta', 'strength_ratio']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: {missing_cols}")
                else:
                    st.success(f"✅ File uploaded successfully: {len(df)} events")
                    
                    # Data quality check
                    with st.expander("📊 Data Quality Report"):
                        col_q1, col_q2, col_q3 = st.columns(3)
                        with col_q1:
                            st.metric("Total Events", len(df))
                        with col_q2:
                            missing_data = df.isnull().sum().sum()
                            st.metric("Missing Values", missing_data)
                        with col_q3:
                            valid_range = len(df[(df['dt'] >= 0) & (df['dtheta'] >= 0)])
                            st.metric("Valid Ranges", f"{valid_range}/{len(df)}")
                    
                    st.dataframe(df.head(), width="stretch")
                    
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
    
    with tab3:
        st.markdown("### Real Astronomical Data APIs")
        
        api_option = st.selectbox(
            "Choose data source:",
            [
                "Mock GW Events API",
                "Mock GRB Catalog API", 
                "Mock Neutrino Events API",
                "Custom API Endpoint"
            ]
        )
        
        if api_option == "Custom API Endpoint":
            api_url = st.text_input("API URL:", placeholder="https://api.example.com/events")
            api_key = st.text_input("API Key (optional):", type="password")
        
        col_api1, col_api2 = st.columns(2)
        with col_api1:
            start_date = st.date_input("Start date", datetime.now().date() - timedelta(days=7))
        with col_api2:
            end_date = st.date_input("End date", datetime.now().date())
        
        if st.button("🌐 **Fetch Real Data**", key="fetch_api"):
            with st.spinner("Fetching data from astronomical databases..."):
                # Simulate API call with realistic data
                time.sleep(2)  # Simulate network delay
                
                # Generate mock API response
                n_api_events = np.random.randint(20, 100)
                api_data = {
                    'dt': np.random.exponential(0.5, n_api_events),
                    'dtheta': np.random.exponential(0.05, n_api_events),
                    'strength_ratio': np.random.lognormal(0.5, 1, n_api_events),
                    'ra': np.random.uniform(0, 360, n_api_events),
                    'dec': np.random.uniform(-90, 90, n_api_events),
                    'api_source': [api_option] * n_api_events,
                    'fetch_time': [datetime.now()] * n_api_events
                }
                
                # Add messenger types based on API source
                if "GW" in api_option:
                    api_data['m1'] = ['GW'] * n_api_events
                    api_data['m2'] = np.random.choice(['Gamma', 'Optical'], n_api_events)
                elif "GRB" in api_option:
                    api_data['m1'] = ['Gamma'] * n_api_events
                    api_data['m2'] = np.random.choice(['Optical', 'Radio'], n_api_events)
                else:
                    messengers = ['GW', 'Gamma', 'Neutrino', 'Optical', 'Radio']
                    api_data['m1'] = np.random.choice(messengers, n_api_events)
                    api_data['m2'] = np.random.choice(messengers, n_api_events)
                
                df = pd.DataFrame(api_data)
                st.session_state.current_data = df
                st.session_state.api_data_cache[api_option] = df
                
                st.success(f"✅ Fetched {len(df)} events from {api_option}")
                st.dataframe(df.head(), width="stretch")
    
    with tab4:
        st.markdown("### Real-time Event Simulation")
        
        col_rt1, col_rt2 = st.columns(2)
        with col_rt1:
            sim_rate = st.slider("Events per minute", 1, 10, 3)
        with col_rt2:
            rt_duration = st.slider("Simulation duration (minutes)", 1, 60, 5)
        
        if st.button("⚡ **Start Real-time Simulation**", key="start_realtime"):
            st.session_state.real_time_mode = True
            
            # Real-time data container
            rt_container = st.container()
            rt_data = []
            
            with rt_container:
                st.info("🔴 **LIVE**: Real-time event simulation active")
                
                # Progress and status
                progress_bar = st.progress(0)
                status_text = st.empty()
                data_display = st.empty()
                
                for minute in range(rt_duration):
                    # Generate events for this minute
                    n_new_events = np.random.poisson(sim_rate)
                    
                    if n_new_events > 0:
                        new_events = {
                            'dt': np.random.exponential(0.1, n_new_events),
                            'dtheta': np.random.exponential(0.01, n_new_events),
                            'strength_ratio': np.random.lognormal(1, 0.5, n_new_events),
                            'ra': np.random.uniform(0, 360, n_new_events),
                            'dec': np.random.uniform(-90, 90, n_new_events),
                            'm1': np.random.choice(['GW', 'Gamma', 'Neutrino'], n_new_events),
                            'm2': np.random.choice(['Optical', 'Radio'], n_new_events),
                            'timestamp': [datetime.now()] * n_new_events
                        }
                        
                        rt_data.extend([dict(zip(new_events.keys(), values)) 
                                       for values in zip(*new_events.values())])
                    
                    # Update display
                    progress_bar.progress((minute + 1) / rt_duration)
                    status_text.text(f"Minute {minute + 1}/{rt_duration} - {len(rt_data)} total events detected")
                    
                    if rt_data:
                        df_rt = pd.DataFrame(rt_data)
                        data_display.dataframe(df_rt.tail(10), width="stretch")
                    
                    time.sleep(1)  # Simulate 1 minute intervals
                
                if rt_data:
                    df_final = pd.DataFrame(rt_data)
                    st.session_state.current_data = df_final
                    st.success(f"✅ Real-time simulation complete: {len(df_final)} events collected")
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Status panel
    st.markdown("### 📊 **System Status**")
    
    # Model status
    if st.session_state.model_loaded:
        st.markdown('<span class="status-badge status-success">🤖 Model Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-warning">⚠️ No Model</span>', unsafe_allow_html=True)
    
    # Data status
    if st.session_state.current_data is not None:
        st.markdown('<span class="status-badge status-success">📊 Data Loaded</span>', unsafe_allow_html=True)
        st.metric("Events", len(st.session_state.current_data))
    else:
        st.markdown('<span class="status-badge status-info">ℹ️ No Data</span>', unsafe_allow_html=True)
    
    # Analysis status
    if st.session_state.results is not None:
        st.markdown('<span class="status-badge status-success">✅ Analysis Complete</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-info">⏳ Ready to Analyze</span>', unsafe_allow_html=True)

# Data overview section (only if data is loaded)
if st.session_state.current_data is not None:
    df = st.session_state.current_data
    
    st.markdown("---")
    st.markdown("## 📋 **Data Overview & Quality Assessment**")
    
    # Enhanced metrics with better styling
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <h3>📊</h3>
            <h2>{}</h2>
            <p>Total Events</p>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        unique_messengers = set()
        if 'm1' in df.columns and 'm2' in df.columns:
            unique_messengers.update(df['m1'].unique())
            unique_messengers.update(df['m2'].unique())
        st.markdown("""
        <div class="metric-container">
            <h3>🌌</h3>
            <h2>{}</h2>
            <p>Messenger Types</p>
        </div>
        """.format(len(unique_messengers)), unsafe_allow_html=True)
    
    with col3:
        time_span = df['dt'].max() - df['dt'].min() if 'dt' in df.columns else 0
        st.markdown("""
        <div class="metric-container">
            <h3>⏱️</h3>
            <h2>{:.1f}s</h2>
            <p>Time Span</p>
        </div>
        """.format(time_span), unsafe_allow_html=True)
    
    with col4:
        max_separation = df['dtheta'].max() if 'dtheta' in df.columns else 0
        st.markdown("""
        <div class="metric-container">
            <h3>📐</h3>
            <h2>{:.2f}°</h2>
            <p>Max Angular Sep</p>
        </div>
        """.format(max_separation), unsafe_allow_html=True)
    
    with col5:
        data_quality = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        quality_color = "success" if data_quality > 90 else "warning" if data_quality > 70 else "danger"
        st.markdown("""
        <div class="metric-container">
            <h3>✅</h3>
            <h2>{:.1f}%</h2>
            <p>Data Quality</p>
        </div>
        """.format(data_quality), unsafe_allow_html=True)
    
    # Enhanced data preview with interactive elements
    with st.expander("🔍 **Interactive Data Explorer**", expanded=False):
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        
        with col_filter1:
            if 'm1' in df.columns:
                messenger_filter = st.multiselect(
                    "Filter by messenger type (m1):",
                    options=df['m1'].unique(),
                    default=df['m1'].unique()
                )
                df_filtered = df[df['m1'].isin(messenger_filter)] if messenger_filter else df
            else:
                df_filtered = df
        
        with col_filter2:
            if 'dt' in df.columns:
                dt_range = st.slider(
                    "Time difference range (s):",
                    float(df['dt'].min()), float(df['dt'].max()),
                    (float(df['dt'].min()), float(df['dt'].max()))
                )
                df_filtered = df_filtered[
                    (df_filtered['dt'] >= dt_range[0]) & 
                    (df_filtered['dt'] <= dt_range[1])
                ]
        
        with col_filter3:
            if 'dtheta' in df.columns:
                dtheta_range = st.slider(
                    "Angular separation range (°):",
                    float(df['dtheta'].min()), float(df['dtheta'].max()),
                    (float(df['dtheta'].min()), float(df['dtheta'].max()))
                )
                df_filtered = df_filtered[
                    (df_filtered['dtheta'] >= dtheta_range[0]) & 
                    (df_filtered['dtheta'] <= dtheta_range[1])
                ]
        
        st.dataframe(df_filtered, width="stretch", height=300)
        st.caption(f"Showing {len(df_filtered)} of {len(df)} events")
    
    # Analysis section with enhanced UI
    st.markdown("---")
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.markdown("## 🚀 **AI-Powered Multimessenger Analysis**")
    
    # Analysis controls
    col_analysis1, col_analysis2, col_analysis3 = st.columns([2, 1, 1])
    
    with col_analysis1:
        st.markdown("### Ready for Analysis")
        if st.session_state.model_loaded:
            st.success("🤖 AI model loaded and ready")
        else:
            st.warning("⚠️ Please select an AI model first")
    
    with col_analysis2:
        if st.button("🧹 **Clear Results**", type="secondary"):
            st.session_state.results = None
            st.success("✅ Results cleared")
    
    with col_analysis3:
        confidence_display = st.empty()
        confidence_display.info(f"🎯 Threshold: {threshold:.2f}")
    
    # Main analysis button with enhanced styling
    analysis_key = f"analyze_enhanced_{len(df)}_{hash(str(df.iloc[0].to_dict()) if len(df) > 0 else 'empty')}"
    
    if st.button("🔍 **Run Advanced AI Analysis**", key=analysis_key, type="primary"):
        if not st.session_state.model_loaded or model is None:
            st.error("❌ Please select a trained AI model first!")
        else:
            # Enhanced progress display
            progress_container = st.container()
            with progress_container:
                progress_col1, progress_col2 = st.columns([3, 1])
                
                with progress_col1:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                with progress_col2:
                    stage_indicator = st.empty()
                
                try:
                    # Stage 1: Data preparation
                    stage_indicator.info("📊 Stage 1/4")
                    status_text.text("🔬 Preparing and validating data...")
                    progress_bar.progress(10)
                    time.sleep(0.5)
                    
                    # Stage 2: Feature engineering
                    stage_indicator.info("🔧 Stage 2/4")
                    status_text.text("🧠 Engineering features for AI model...")
                    progress_bar.progress(30)
                    time.sleep(0.5)
                    
                    # Stage 3: AI inference
                    stage_indicator.info("🤖 Stage 3/4")
                    status_text.text("🚀 Running AI inference on multimessenger data...")
                    progress_bar.progress(60)
                    
                    # Run the actual prediction
                    df_pred = predict_df(df, model, scaler=scaler, threshold=threshold)
                    progress_bar.progress(85)
                    
                    # Stage 4: Post-processing
                    stage_indicator.info("📈 Stage 4/4")
                    status_text.text("📊 Processing results and generating insights...")
                    
                    # Add enhanced metrics to results
                    df_pred['confidence_category'] = pd.cut(
                        df_pred['pred_prob'], 
                        bins=[0, 0.3, 0.7, 1.0], 
                        labels=['Low', 'Medium', 'High']
                    )
                    
                    # Add risk assessment
                    df_pred['significance'] = (
                        df_pred['pred_prob'] * 
                        (1 / (df_pred['dt'] + 0.1)) * 
                        (1 / (df_pred['dtheta'] + 0.01))
                    )
                    
                    progress_bar.progress(100)
                    time.sleep(0.5)
                    
                    # Store results
                    st.session_state.results = df_pred
                    
                    # Clear progress indicators
                    progress_container.empty()
                    
                    # Success message with stats
                    positive_count = len(df_pred[df_pred['pred_label'] == 1])
                    high_conf_count = len(df_pred[df_pred['pred_prob'] >= 0.8])
                    
                    st.success(f"""
                    ✅ **Analysis Complete!** 
                    Found {positive_count} positive associations 
                    ({high_conf_count} high confidence)
                    """)
                    
                except Exception as e:
                    progress_container.empty()
                    st.error(f"❌ Analysis failed: {e}")
                    
                    if show_debug:
                        with st.expander("🐛 Debug Information"):
                            import traceback
                            st.code(traceback.format_exc())
    
    st.markdown('</div>', unsafe_allow_html=True)

# Enhanced Results Section
if st.session_state.results is not None:
    df_pred = st.session_state.results
    
    st.markdown("---")
    st.markdown('<div class="results-header">🎯 Advanced Analysis Results</div>', unsafe_allow_html=True)
    
    # Enhanced results metrics
    total_events = len(df_pred)
    high_confidence = len(df_pred[df_pred['pred_prob'] >= 0.8])
    medium_confidence = len(df_pred[(df_pred['pred_prob'] >= 0.5) & (df_pred['pred_prob'] < 0.8)])
    positive_associations = len(df_pred[df_pred['pred_label'] == 1])
    avg_confidence = df_pred['pred_prob'].mean()
    max_significance = df_pred['significance'].max() if 'significance' in df_pred.columns else 0
    
    # Results overview with enhanced styling
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    metrics_data = [
        ("🔴", high_confidence, "High Confidence", f"{high_confidence/total_events*100:.1f}%"),
        ("🟡", medium_confidence, "Medium Confidence", f"{medium_confidence/total_events*100:.1f}%"),
        ("✅", positive_associations, "Positive Associations", f"Threshold: {threshold}"),
        ("📊", f"{avg_confidence:.3f}", "Average Confidence", "Overall Score"),
        ("🌟", f"{max_significance:.2f}", "Max Significance", "Highest Priority"),
        ("📈", f"{total_events}", "Total Analyzed", "Events Processed")
    ]
    
    for i, (icon, value, label, delta) in enumerate(metrics_data):
        with [col1, col2, col3, col4, col5, col6][i]:
            st.markdown(f"""
            <div class="metric-container">
                <h3>{icon}</h3>
                <h2>{value}</h2>
                <p>{label}</p>
                <small>{delta}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Enhanced results tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 **Priority Events**", 
        "📊 **All Results**", 
        "📈 **Advanced Visualizations**",
        "🗺️ **Sky Maps**",
        "📋 **Scientific Analysis**"
    ])
    
    with tab1:
        st.markdown("### 🚨 High-Priority Multimessenger Associations")
        
        # Priority filtering
        priority_threshold = st.slider("Priority threshold:", 0.0, 1.0, 0.7, 0.05)
        high_priority = df_pred[df_pred['pred_prob'] >= priority_threshold].sort_values('pred_prob', ascending=False)
        
        if len(high_priority) > 0:
            st.markdown(f"**{len(high_priority)} high-priority associations found:**")
            
            # Enhanced table with color coding
            def highlight_confidence(val):
                if val >= 0.8:
                    return 'background-color: rgba(0, 200, 81, 0.3)'
                elif val >= 0.6:
                    return 'background-color: rgba(255, 187, 51, 0.3)'
                else:
                    return 'background-color: rgba(255, 68, 68, 0.3)'
            
            styled_df = high_priority[['m1', 'm2', 'dt', 'dtheta', 'strength_ratio', 'pred_prob', 'pred_label']].style.applymap(
                highlight_confidence, subset=['pred_prob']
            )
            
            st.dataframe(styled_df, width="stretch")
            
            # Alert generation with enhanced options
            col_alert1, col_alert2 = st.columns(2)
            with col_alert1:
                alert_format = st.selectbox("Alert format:", ["CSV", "JSON", "XML"])
            with col_alert2:
                include_metadata = st.checkbox("Include analysis metadata", True)
            
            if st.button("🚨 **Generate Priority Alerts**", key="generate_priority_alerts"):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                if alert_format == "CSV":
                    alert_file = f"alerts/priority_alerts_{timestamp}.csv"
                    high_priority.to_csv(alert_file, index=False)
                elif alert_format == "JSON":
                    alert_file = f"alerts/priority_alerts_{timestamp}.json"
                    alert_data = {
                        "generated_at": timestamp,
                        "threshold": priority_threshold,
                        "total_events": len(high_priority),
                        "events": high_priority.to_dict('records')
                    }
                    with open(alert_file, 'w') as f:
                        json.dump(alert_data, f, indent=2, default=str)
                
                st.success(f"✅ Alert file generated: `{alert_file}`")
        else:
            st.info(f"No associations found above {priority_threshold:.2f} confidence threshold")
    
    with tab2:
        st.markdown("### 📋 Complete Analysis Results")
        
        # Results filtering and sorting
        col_sort1, col_sort2, col_sort3 = st.columns(3)
        with col_sort1:
            sort_by = st.selectbox("Sort by:", ["pred_prob", "dt", "dtheta", "significance"])
        with col_sort2:
            sort_order = st.selectbox("Order:", ["Descending", "Ascending"])
        with col_sort3:
            max_display = st.number_input("Max rows to display:", 10, 1000, 200)
        
        df_display = df_pred.sort_values(
            sort_by, 
            ascending=(sort_order == "Ascending")
        ).head(max_display)
        
        st.dataframe(df_display, width="stretch", height=400)
        
        # Enhanced download options
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        
        with col_dl1:
            csv_data = df_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 **Download CSV**",
                data=csv_data,
                file_name=f'analysis_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv'
            )
        
        with col_dl2:
            json_data = df_display.to_json(orient='records', indent=2).encode('utf-8')
            st.download_button(
                label="📥 **Download JSON**",
                data=json_data,
                file_name=f'analysis_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                mime='application/json'
            )
        
        with col_dl3:
            # Create summary report
            summary_report = f"""
# Multimessenger Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics
- Total Events Analyzed: {total_events}
- Positive Associations: {positive_associations} ({positive_associations/total_events*100:.1f}%)
- High Confidence (>0.8): {high_confidence} ({high_confidence/total_events*100:.1f}%)
- Average Confidence: {avg_confidence:.3f}
- Analysis Threshold: {threshold}

## Model Information
- Model Type: {metadata.get('best_model', 'Unknown') if metadata else 'Unknown'}
- Model AUC: {metadata.get('best_auc', 'N/A') if metadata else 'N/A'}

## Top 10 Associations
{df_display.head(10)[['m1', 'm2', 'dt', 'dtheta', 'pred_prob']].to_string()}
"""
            st.download_button(
                label="📄 **Download Report**",
                data=summary_report.encode('utf-8'),
                file_name=f'analysis_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md',
                mime='text/markdown'
            )
    
    # Continue with visualization tabs...
    with tab3:
        st.markdown("### 📈 Advanced Statistical Visualizations")
        
        # Create sophisticated visualizations
        fig_matrix = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Confidence Distribution', 'Time vs Angular Separation', 
                          'Messenger Type Analysis', 'Significance Analysis'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Confidence distribution
        fig_matrix.add_trace(
            go.Histogram(x=df_pred['pred_prob'], nbinsx=30, name='Confidence', 
                        marker_color='rgba(31, 119, 180, 0.7)'),
            row=1, col=1
        )
        
        # Time vs Angular separation with confidence coloring
        fig_matrix.add_trace(
            go.Scatter(x=df_pred['dt'], y=df_pred['dtheta'], 
                      mode='markers',
                      marker=dict(size=8, color=df_pred['pred_prob'], 
                                colorscale='Viridis', showscale=True,
                                colorbar=dict(title="Confidence")),
                      name='Events'),
            row=1, col=2
        )
        
        # Messenger type analysis
        if 'm1' in df_pred.columns and 'm2' in df_pred.columns:
            pair_counts = df_pred.groupby(['m1', 'm2']).size().reset_index(name='count')
            fig_matrix.add_trace(
                go.Bar(x=[f"{row.m1}-{row.m2}" for _, row in pair_counts.iterrows()], 
                       y=pair_counts['count'], name='Pair Counts',
                       marker_color='rgba(255, 127, 14, 0.7)'),
                row=2, col=1
            )
        
        # Significance analysis
        if 'significance' in df_pred.columns:
            fig_matrix.add_trace(
                go.Scatter(x=df_pred['pred_prob'], y=df_pred['significance'],
                          mode='markers', name='Significance vs Confidence',
                          marker=dict(size=6, color='rgba(44, 160, 44, 0.7)')),
                row=2, col=2
            )
        
        fig_matrix.update_layout(height=800, showlegend=False, 
                               title_text="Advanced Multimessenger Analysis Dashboard")
        st.plotly_chart(fig_matrix, width="stretch")
        
        # Correlation analysis
        st.markdown("#### 🔗 Correlation Analysis")
        numeric_cols = df_pred.select_dtypes(include=[np.number]).columns
        correlation_matrix = df_pred[numeric_cols].corr()
        
        fig_corr = px.imshow(correlation_matrix, 
                           title="Feature Correlation Matrix",
                           color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_corr, width="stretch")
    
    with tab4:
        st.markdown("### 🗺️ Multimessenger Sky Maps")
        
        if 'ra' in df_pred.columns and 'dec' in df_pred.columns:
            # 3D sky map
            fig_3d = go.Figure(data=go.Scatter3d(
                x=np.cos(np.radians(df_pred['dec'])) * np.cos(np.radians(df_pred['ra'])),
                y=np.cos(np.radians(df_pred['dec'])) * np.sin(np.radians(df_pred['ra'])),
                z=np.sin(np.radians(df_pred['dec'])),
                mode='markers',
                marker=dict(
                    size=8,
                    color=df_pred['pred_prob'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Confidence Score")
                ),
                text=[f"RA: {ra:.2f}°, Dec: {dec:.2f}°, Conf: {conf:.3f}" 
                      for ra, dec, conf in zip(df_pred['ra'], df_pred['dec'], df_pred['pred_prob'])],
                hovertemplate='%{text}<extra></extra>'
            ))
            
            fig_3d.update_layout(
                title='3D Celestial Sphere View',
                scene=dict(
                    xaxis_title='X (Celestial)',
                    yaxis_title='Y (Celestial)',
                    zaxis_title='Z (Celestial)'
                ),
                height=600
            )
            st.plotly_chart(fig_3d, width="stretch")
            
            # Traditional sky map
            fig_sky = px.scatter(df_pred, x='ra', y='dec', color='pred_prob',
                               title='🌌 Sky Distribution of Multimessenger Associations',
                               labels={'ra': 'Right Ascension (degrees)', 'dec': 'Declination (degrees)'},
                               color_continuous_scale='Plasma',
                               hover_data=['dt', 'dtheta', 'strength_ratio'])
            
            # Add constellation-like grid
            fig_sky.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            for ra in range(0, 360, 30):
                fig_sky.add_vline(x=ra, line_dash="dot", line_color="gray", opacity=0.3)
            
            fig_sky.update_layout(
                xaxis=dict(range=[0, 360], title="Right Ascension (°)"),
                yaxis=dict(range=[-90, 90], title="Declination (°)"),
                height=500
            )
            st.plotly_chart(fig_sky, width="stretch")
            
        else:
            st.info("Sky coordinates (RA/Dec) not available in dataset for sky mapping")
    
    with tab5:
        st.markdown("### 📊 Scientific Analysis Tools")
        
        # Statistical tests and analysis
        col_sci1, col_sci2 = st.columns(2)
        
        with col_sci1:
            st.markdown("#### 📈 Statistical Summary")
            
            # Confidence intervals
            confidence_stats = df_pred['pred_prob'].describe()
            st.markdown(f"""
            **Confidence Score Statistics:**
            - Mean: {confidence_stats['mean']:.3f}
            - Median: {confidence_stats['50%']:.3f}
            - Std Dev: {confidence_stats['std']:.3f}
            - 95th Percentile: {df_pred['pred_prob'].quantile(0.95):.3f}
            """)
            
            # Time delay analysis
            if 'dt' in df_pred.columns:
                time_stats = df_pred['dt'].describe()
                st.markdown(f"""
                **Time Delay Analysis:**
                - Mean Δt: {time_stats['mean']:.3f} s
                - Median Δt: {time_stats['50%']:.3f} s
                - 99th Percentile: {df_pred['dt'].quantile(0.99):.3f} s
                """)
        
        with col_sci2:
            st.markdown("#### 🔬 Physics Insights")
            
            # Speed of light checks
            if 'dt' in df_pred.columns and 'dtheta' in df_pred.columns:
                # Approximate distance for light travel time
                c_light = 3e8  # m/s
                typical_distance = 100 * 3.086e22  # 100 Mpc in meters
                
                light_travel_times = df_pred['dtheta'] * np.pi/180 * typical_distance / c_light
                causal_events = len(df_pred[df_pred['dt'] >= light_travel_times])
                
                st.markdown(f"""
                **Causality Analysis (@ 100 Mpc):**
                - Causally connected events: {causal_events}/{len(df_pred)}
                - Fraction: {causal_events/len(df_pred)*100:.1f}%
                """)
            
            # Messenger type preferences
            if 'm1' in df_pred.columns and 'm2' in df_pred.columns:
                high_conf_pairs = df_pred[df_pred['pred_prob'] >= 0.8]
                if len(high_conf_pairs) > 0:
                    top_pair = high_conf_pairs.groupby(['m1', 'm2']).size().idxmax()
                    st.markdown(f"""
                    **Most Significant Pairing:**
                    - {top_pair[0]} ↔ {top_pair[1]}
                    - High confidence events: {len(high_conf_pairs)}
                    """)
        
        # Parameter sensitivity analysis
        st.markdown("#### ⚙️ Parameter Sensitivity Analysis")
        
        sensitivity_param = st.selectbox(
            "Parameter to analyze:",
            ['threshold', 'time_window', 'angular_resolution']
        )
        
        if sensitivity_param == 'threshold':
            # Threshold sensitivity
            thresholds = np.arange(0.1, 1.0, 0.1)
            positive_counts = []
            
            for t in thresholds:
                count = len(df_pred[df_pred['pred_prob'] >= t])
                positive_counts.append(count)
            
            fig_sens = px.line(x=thresholds, y=positive_counts,
                             title='Sensitivity to Confidence Threshold',
                             labels={'x': 'Threshold', 'y': 'Positive Associations'})
            fig_sens.add_vline(x=threshold, line_dash="dash", 
                             annotation_text=f"Current: {threshold}")
            st.plotly_chart(fig_sens, width="stretch")
        
        # Export scientific data
        st.markdown("#### 📤 Scientific Data Export")
        
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            # Create publication-ready dataset
            pub_data = df_pred[['m1', 'm2', 'dt', 'dtheta', 'strength_ratio', 
                               'pred_prob', 'pred_label']].copy()
            pub_data.columns = ['Messenger_1', 'Messenger_2', 'Time_Delay_s', 
                               'Angular_Separation_deg', 'Strength_Ratio', 
                               'AI_Confidence', 'Association_Flag']
            
            pub_csv = pub_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📊 **Publication Dataset**",
                data=pub_csv,
                file_name=f'multimessenger_science_data_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv'
            )
        
        with col_export2:
            # Create analysis metadata
            metadata_export = {
                "analysis_timestamp": datetime.now().isoformat(),
                "model_info": metadata if metadata else {},
                "analysis_parameters": {
                    "threshold": threshold,
                    "total_events": total_events,
                    "positive_associations": positive_associations
                },
                "statistics": {
                    "mean_confidence": avg_confidence,
                    "high_confidence_events": high_confidence,
                    "data_quality_score": data_quality
                }
            }
            
            metadata_json = json.dumps(metadata_export, indent=2, default=str).encode('utf-8')
            st.download_button(
                label="📋 **Analysis Metadata**",
                data=metadata_json,
                file_name=f'analysis_metadata_{datetime.now().strftime("%Y%m%d")}.json',
                mime='application/json'
            )

else:
    # No data loaded - show getting started guide
    st.markdown("---")
    st.markdown("## 🚀 **Getting Started with Multimessenger Analysis**")
    
    st.markdown("""
    <div class="input-card">
    <h3>👋 Welcome to the Multimessenger AI Observatory!</h3>
    <p>Follow these steps to start analyzing multimessenger astronomical events:</p>
    
    <ol>
    <li><strong>Select an AI Model</strong> - Choose from the trained models in the sidebar</li>
    <li><strong>Load Data</strong> - Use one of the data input methods above:
        <ul>
        <li>🚀 Generate demo data for testing</li>
        <li>📂 Upload your own CSV files</li>
        <li>🌐 Fetch real astronomical data via APIs</li>
        <li>⚡ Simulate real-time event detection</li>
        </ul>
    </li>
    <li><strong>Run Analysis</strong> - Click the analysis button to process your data</li>
    <li><strong>Explore Results</strong> - View visualizations, download data, and generate alerts</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Educational content for students
    with st.expander("📚 **Learn About Multimessenger Astronomy**"):
        st.markdown("""
        ### What is Multimessenger Astronomy?
        
        Multimessenger astronomy is a revolutionary approach that combines observations from different cosmic messengers:
        
        - **🌊 Gravitational Waves**: Ripples in spacetime from accelerating massive objects
        - **🔺 Neutrinos**: Nearly massless particles that travel through matter unimpeded  
        - **⚡ Gamma Rays**: High-energy electromagnetic radiation
        - **🔍 Optical Light**: Traditional electromagnetic observations
        - **📡 Radio Waves**: Low-energy electromagnetic signals
        
        ### Why Use AI?
        
        Machine learning helps identify subtle correlations between these different signals that might indicate they originated from the same astrophysical event, even when individual signals are weak or noisy.
        
        ### Key Parameters:
        
        - **Time Delay (Δt)**: Time difference between detection of different messengers
        - **Angular Separation (Δθ)**: Difference in sky position between events  
        - **Strength Ratio**: Relative intensity of the two signals
        - **Confidence Score**: AI-predicted probability of true association
        """)

# Footer with enhanced information
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(90deg, rgba(31, 119, 180, 0.1) 0%, rgba(0, 212, 255, 0.1) 100%); border-radius: 10px; margin-top: 2rem;">
    <h3>🌌 Multimessenger AI Observatory</h3>
    <p>Advanced AI-powered analysis platform for multimessenger astronomical events</p>
    <p><strong>Built for researchers, students, and educators</strong></p>
    <p style="font-size: 0.9rem; opacity: 0.8;">
        Powered by Machine Learning | Real-time Analysis | Educational Tools | Scientific Export
    </p>
</div>
""", unsafe_allow_html=True)
