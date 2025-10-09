#!/usr/bin/env python3
"""
Enhanced Multimessenger AI Analysis Platform
Features: API integration, advanced visualizations, event clustering, modern UI
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
from model_loader import list_model_files, load_model_by_name, get_model_info
from inference import predict_df

# Page configuration
st.set_page_config(
    page_title="Multimessenger AI Platform",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern violet-themed UI
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, #9c88ff 0%, #c8b3ff 50%, #e1d5ff 100%);
    }
    
    .main-header {
        background: linear-gradient(90deg, #7c4dff 0%, #512da8 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(124, 77, 255, 0.3);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #7c4dff 0%, #9575cd 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 16px rgba(124, 77, 255, 0.3);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(124, 77, 255, 0.4);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.2), rgba(255, 255, 255, 0.1));
        border-radius: 10px 10px 0 0;
        padding: 0.5rem 1rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .feature-box {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #7c4dff;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }
    
    .success-box {
        background: linear-gradient(90deg, #7c4dff 0%, #9575cd 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(124, 77, 255, 0.3);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .warning-box {
        background: linear-gradient(90deg, #b39ddb 0%, #d1c4e9 100%);
        color: #4a148c;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(179, 157, 219, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
        font-weight: 600;
    }
    
    .info-box {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.25), rgba(255, 255, 255, 0.15));
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: #4a148c;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #7c4dff 0%, #512da8 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #7c4dff, #512da8);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(124, 77, 255, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(124, 77, 255, 0.4);
        background: linear-gradient(135deg, #512da8, #7c4dff);
    }
    
    /* Data frame styling */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Prediction results */
    .prediction-same {
        background: linear-gradient(135deg, #7c4dff, #9575cd);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
        box-shadow: 0 4px 16px rgba(124, 77, 255, 0.3);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .prediction-different {
        background: linear-gradient(135deg, #d1c4e9, #b39ddb);
        color: #4a148c;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
        box-shadow: 0 4px 16px rgba(179, 157, 219, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🌌 Advanced Multimessenger AI Analysis Platform</h1>
    <p>Next-generation AI-powered detection and analysis of multimessenger astronomical events</p>
    <p>✨ Real-time data • 🎯 API integration • 📊 Advanced visualizations • 🔬 Event clustering</p>
</div>
""", unsafe_allow_html=True)

# Progress Tracker
st.markdown("## 📋 Analysis Progress Tracker")

# Define analysis stages
stages = [
    {"name": "🤖 Model Selection", "key": "model_selected", "description": "Select and load AI model"},
    {"name": "📊 Data Loading", "key": "data_loaded", "description": "Upload or generate analysis data"},
    {"name": "🔬 Data Analysis", "key": "analysis_complete", "description": "Run AI analysis on event pairs"},
    {"name": "📈 Results Ready", "key": "results_available", "description": "View visualizations and reports"},
    {"name": "📥 Download Complete", "key": "download_ready", "description": "Export results and reports"}
]

# Check current progress
progress_status = {}
for stage in stages:
    if stage["key"] == "model_selected":
        progress_status[stage["key"]] = "model_selector" in st.session_state and st.session_state.get("model_selector") is not None
    elif stage["key"] == "data_loaded":
        progress_status[stage["key"]] = (st.session_state.get("current_data") is not None or 
                                       st.session_state.get("uploaded_data") is not None or
                                       st.session_state.get("demo_data") is not None)
    elif stage["key"] == "analysis_complete":
        progress_status[stage["key"]] = st.session_state.get("analysis_complete", False)
    elif stage["key"] == "results_available":
        progress_status[stage["key"]] = st.session_state.get("results") is not None
    elif stage["key"] == "download_ready":
        progress_status[stage["key"]] = st.session_state.get("results") is not None
    else:
        progress_status[stage["key"]] = False

# Calculate overall progress
completed_stages = sum(1 for status in progress_status.values() if status)
total_stages = len(stages)
progress_percentage = (completed_stages / total_stages) * 100

# Display progress bar
st.progress(progress_percentage / 100, text=f"Overall Progress: {completed_stages}/{total_stages} stages complete ({progress_percentage:.0f}%)")

# Display stage indicators
progress_cols = st.columns(len(stages))

for i, (stage, col) in enumerate(zip(stages, progress_cols)):
    with col:
        is_completed = progress_status[stage["key"]]
        is_current = (not is_completed and 
                     (i == 0 or progress_status[stages[i-1]["key"]]))
        
        # Determine status icon and color
        if is_completed:
            status_icon = "✅"
            status_color = "#28a745"  # Green
            status_text = "Complete"
        elif is_current:
            status_icon = "🔄"
            status_color = "#ffc107"  # Yellow
            status_text = "In Progress"
        else:
            status_icon = "⭕"
            status_color = "#6c757d"  # Gray
            status_text = "Pending"
        
        # Display stage card
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
            border: 2px solid {status_color};
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
            margin-bottom: 1rem;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        ">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">{status_icon}</div>
            <div style="font-weight: bold; margin-bottom: 0.5rem; color: {status_color};">{stage['name']}</div>
            <div style="font-size: 0.8rem; color: #666; margin-bottom: 0.5rem;">{stage['description']}</div>
            <div style="font-size: 0.75rem; color: {status_color}; font-weight: bold;">{status_text}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# Sidebar
st.sidebar.markdown("## 🎛️ Analysis Controls")

# Model selection
st.sidebar.markdown("### 🤖 AI Model")
model_files = list_model_files()

if model_files:
    # Extract just the filenames for display
    model_names = [f[0] for f in model_files] if model_files and isinstance(model_files[0], tuple) else model_files
    
    model_choice = st.sidebar.selectbox(
        "Choose trained model:",
        model_names,
        key="model_selector"
    )
else:
    model_choice = None
    st.sidebar.markdown("""
    <div class="warning-box">
        ⚠️ <strong>No Models Found</strong><br>
        Please ensure model files (.pkl) are available
    </div>
    """, unsafe_allow_html=True)

# Initialize session state
for key in ['current_data', 'results', 'api_data', 'clustering_results']:
    if key not in st.session_state:
        st.session_state[key] = None

# Load model
model = None
scaler = None
metadata = None

if model_choice:
    try:
        model, scaler, metadata = load_model_by_name(model_choice)
        st.sidebar.markdown(f"""
        <div class="success-box">
            ✅ <strong>Model Loaded Successfully</strong><br>
            {model_choice}
        </div>
        """, unsafe_allow_html=True)
        
        if metadata:
            st.sidebar.markdown(f"""
            <div class="info-box">
                <h4 style="margin-bottom: 1rem;">📊 Model Information</h4>
                <p><strong>Algorithm:</strong> {metadata.get('best_model', 'Unknown')}</p>
                <p><strong>AUC Score:</strong> {metadata.get('best_auc', 'N/A'):.3f}</p>
                <p><strong>Features:</strong> {len(metadata.get('feature_names', []))} columns</p>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.sidebar.markdown(f"""
        <div class="warning-box">
            ❌ <strong>Model Loading Failed</strong><br>
            {str(e)}
        </div>
        """, unsafe_allow_html=True)
        model = None

# Analysis parameters
st.sidebar.markdown("### ⚙️ Analysis Parameters")
threshold = st.sidebar.slider("🎯 Association Threshold", 0.0, 1.0, 0.5, 0.05)
clustering_eps = st.sidebar.slider("🔗 Clustering Sensitivity", 0.1, 2.0, 0.5, 0.1)
min_samples = st.sidebar.slider("👥 Min Cluster Size", 2, 10, 3)

# Advanced options
with st.sidebar.expander("🔬 Advanced Options"):
    confidence_interval = st.slider("Confidence Interval", 0.90, 0.99, 0.95, 0.01)
    show_debug = st.checkbox("Show Debug Info")
    auto_refresh = st.checkbox("Auto-refresh Real-time Data")

# Clear functionality in sidebar
st.sidebar.markdown("### 🗑️ Clear Data")

clear_sidebar_col1, clear_sidebar_col2 = st.sidebar.columns(2)

with clear_sidebar_col1:
    if st.button("🗑️ Clear Results", key="sidebar_clear_results", help="Clear analysis results only"):
        # Clear only analysis results
        if 'results' in st.session_state:
            del st.session_state.results
        if 'analysis_complete' in st.session_state:
            del st.session_state.analysis_complete
        st.sidebar.success("Results cleared!")
        st.rerun()

with clear_sidebar_col2:
    if st.button("🔄 Reset All", key="sidebar_reset_all", help="Clear all data and reset session"):
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.sidebar.success("Session reset!")
        st.rerun()

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Input", 
    "🔬 AI Analysis", 
    "📈 Visualizations", 
    "🎯 Event Clustering",
    "🌐 Real-time Monitor"
])

with tab1:
    st.markdown("## 📊 Data Input Options")
    
    # Data input method selection
    input_col1, input_col2 = st.columns(2)
    
    with input_col1:
        st.markdown("""
        <div class="feature-box">
        <h4>🗂️ Standard Data Sources</h4>
        <p>Upload files or generate demo data for analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        input_method = st.radio(
            "Choose input method:",
            ["🎲 Generate Demo Data", "📂 Upload Files", "📋 Manual Entry", "🌐 Database Import", "📡 Real-time API"],
            horizontal=False
        )
    
    with input_col2:
        st.markdown("""
        <div class="feature-box">
        <h4>🌐 Live Data Sources</h4>
        <p>Connect to real-time astronomical databases and observatories</p>
        </div>
        """, unsafe_allow_html=True)
        
        api_source = st.selectbox(
            "Real-time data source:",
            ["🔭 LIGO/Virgo GW Events", "🌟 Gamma-ray Bursts (GRB)", "⚡ Neutrino Alerts", "🎯 Multi-messenger Alerts"]
        )
        
        if st.button("🔄 Fetch Live Data", type="secondary"):
            with st.spinner("Fetching real-time data..."):
                # Mock API integration (replace with real APIs)
                time.sleep(2)
                
                # Generate realistic mock data
                np.random.seed(int(time.time()) % 1000)
                n_events = np.random.randint(5, 25)
                
                api_data = {
                    'dt': np.abs(np.random.normal(0, 500, n_events)),
                    'dtheta': np.random.exponential(0.8, n_events),
                    'strength_ratio': np.random.exponential(1.5, n_events),
                    'event_id': [f"API_{api_source.split()[0]}_{i:03d}" for i in range(n_events)],
                    'timestamp': [datetime.now() - timedelta(hours=np.random.randint(0, 48)) for _ in range(n_events)],
                    'confidence': np.random.uniform(0.7, 0.99, n_events)
                }
                
                st.session_state.api_data = pd.DataFrame(api_data)
                st.success(f"✅ Fetched {n_events} events from {api_source}")
    
    # Standard data input
    df = None
    
    if input_method == "🎲 Generate Demo Data":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_pairs = st.number_input("Number of pairs", 50, 500, 100)
        with col2:
            noise_level = st.slider("Noise level", 0.1, 2.0, 1.0)
        with col3:
            seed = st.number_input("Random seed", 1, 1000, 42)
        
        if st.button("🎲 Generate Data", type="primary"):
            np.random.seed(seed)
            
            # Generate more realistic astronomical data
            data = {
                'dt': np.abs(np.random.normal(0, 1000 * noise_level, n_pairs)),
                'dtheta': np.random.exponential(1.0 * noise_level, n_pairs),
                'strength_ratio': np.random.exponential(2.0 / noise_level, n_pairs),
                'event_type': np.random.choice(['GW-Gamma', 'GW-Neutrino', 'Gamma-Neutrino'], n_pairs),
                'detection_time': [datetime.now() - timedelta(days=np.random.randint(0, 30)) for _ in range(n_pairs)]
            }
            
            df = pd.DataFrame(data)
            st.session_state.current_data = df
            st.session_state.data_loaded = True  # Mark data loading stage as complete
            
            st.markdown(f"""
            <div class="success-box">
            <h4>✅ Generated {len(df)} astronomical event pairs</h4>
            <p>Noise level: {noise_level} | Seed: {seed} | Ready for AI analysis</p>
            </div>
            """, unsafe_allow_html=True)
    
    elif input_method == "📂 Upload Files":
        st.markdown("### 📁 File Upload Options")
        
        file_type = st.selectbox(
            "Select file format:",
            ["CSV Files (.csv)", "Excel Files (.xlsx, .xls)", "JSON Files (.json)", "Parquet Files (.parquet)"]
        )
        
        if file_type == "CSV Files (.csv)":
            uploaded_file = st.file_uploader(
                "Choose CSV file", 
                type=["csv"],
                help="Upload astronomical event data in CSV format"
            )
            
            if uploaded_file:
                try:
                    # CSV options
                    col1, col2 = st.columns(2)
                    with col1:
                        separator = st.selectbox("Separator:", [",", ";", "\t", "|"])
                    with col2:
                        encoding = st.selectbox("Encoding:", ["utf-8", "latin-1", "cp1252"])
                    
                    df = pd.read_csv(uploaded_file, sep=separator, encoding=encoding)
                    st.session_state.current_data = df
                    st.session_state.data_loaded = True  # Mark data loading stage as complete
                    
                    st.markdown(f"""
                    <div class="success-box">
                    <h4>✅ CSV File Loaded Successfully</h4>
                    <p>Filename: {uploaded_file.name} | Rows: {len(df)} | Columns: {len(df.columns)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error reading CSV file: {e}")
        
        elif file_type == "Excel Files (.xlsx, .xls)":
            uploaded_file = st.file_uploader(
                "Choose Excel file", 
                type=["xlsx", "xls"],
                help="Upload astronomical event data in Excel format"
            )
            
            if uploaded_file:
                try:
                    # Excel options
                    sheet_names = pd.ExcelFile(uploaded_file).sheet_names
                    selected_sheet = st.selectbox("Select sheet:", sheet_names)
                    
                    df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                    st.session_state.current_data = df
                    st.session_state.data_loaded = True  # Mark data loading stage as complete
                    
                    st.markdown(f"""
                    <div class="success-box">
                    <h4>✅ Excel File Loaded Successfully</h4>
                    <p>Sheet: {selected_sheet} | Rows: {len(df)} | Columns: {len(df.columns)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error reading Excel file: {e}")
        
        elif file_type == "JSON Files (.json)":
            uploaded_file = st.file_uploader(
                "Choose JSON file", 
                type=["json"],
                help="Upload astronomical event data in JSON format"
            )
            
            if uploaded_file:
                try:
                    json_data = json.load(uploaded_file)
                    
                    # Handle different JSON structures
                    if isinstance(json_data, list):
                        df = pd.DataFrame(json_data)
                    elif isinstance(json_data, dict):
                        if 'data' in json_data:
                            df = pd.DataFrame(json_data['data'])
                        else:
                            df = pd.DataFrame([json_data])
                    
                    st.session_state.current_data = df
                    st.session_state.data_loaded = True  # Mark data loading stage as complete
                    
                    st.markdown(f"""
                    <div class="success-box">
                    <h4>✅ JSON File Loaded Successfully</h4>
                    <p>Filename: {uploaded_file.name} | Rows: {len(df)} | Columns: {len(df.columns)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error reading JSON file: {e}")
        
        elif file_type == "Parquet Files (.parquet)":
            uploaded_file = st.file_uploader(
                "Choose Parquet file", 
                type=["parquet"],
                help="Upload astronomical event data in Parquet format"
            )
            
            if uploaded_file:
                try:
                    df = pd.read_parquet(uploaded_file)
                    st.session_state.current_data = df
                    st.session_state.data_loaded = True  # Mark data loading stage as complete
                    
                    st.markdown(f"""
                    <div class="success-box">
                    <h4>✅ Parquet File Loaded Successfully</h4>
                    <p>Filename: {uploaded_file.name} | Rows: {len(df)} | Columns: {len(df.columns)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error reading Parquet file: {e}")
    
    elif input_method == "🌐 Database Import":
        st.markdown("### 🗄️ Database Connection")
        
        db_type = st.selectbox(
            "Database type:",
            ["PostgreSQL", "MySQL", "SQLite", "MongoDB", "InfluxDB"]
        )
        
        if db_type in ["PostgreSQL", "MySQL"]:
            col1, col2 = st.columns(2)
            with col1:
                host = st.text_input("Host:", "localhost")
                database = st.text_input("Database:", "astronomy")
            with col2:
                port = st.number_input("Port:", value=5432 if db_type == "PostgreSQL" else 3306)
                table = st.text_input("Table/Collection:", "events")
            
            username = st.text_input("Username:")
            password = st.text_input("Password:", type="password")
            
            if st.button("🔗 Connect & Import", type="primary"):
                st.markdown("""
                <div class="warning-box">
                <h4>🚧 Database Connection Demo</h4>
                <p>This is a demo interface. In production, this would connect to your actual database.</p>
                </div>
                """, unsafe_allow_html=True)
        
        elif db_type == "SQLite":
            uploaded_db = st.file_uploader("Upload SQLite database file", type=["db", "sqlite", "sqlite3"])
            if uploaded_db:
                st.info("SQLite database upload functionality would be implemented here.")
    
    elif input_method == "📡 Real-time API":
        st.markdown("### 🌐 Real-time Data Sources")
        
        api_choice = st.selectbox(
            "Select API source:",
            ["GW Open Science Center", "GCN (Gamma-ray Coordinates Network)", "ANTARES", "Custom API"]
        )
        
        if api_choice == "Custom API":
            api_url = st.text_input("API Endpoint URL:")
            api_key = st.text_input("API Key (optional):", type="password")
            
            col1, col2 = st.columns(2)
            with col1:
                method = st.selectbox("HTTP Method:", ["GET", "POST"])
            with col2:
                timeout = st.number_input("Timeout (seconds):", value=30)
            
            if st.button("🔄 Fetch Data", type="primary") and api_url:
                try:
                    with st.spinner("Fetching data from API..."):
                        headers = {}
                        if api_key:
                            headers["Authorization"] = f"Bearer {api_key}"
                        
                        # Mock API call for demo
                        time.sleep(2)
                        st.markdown("""
                        <div class="info-box">
                        <h4>📡 API Demo Mode</h4>
                        <p>In production, this would fetch real data from your API endpoint.</p>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"API Error: {e}")
        
        else:
            if st.button(f"🔄 Fetch from {api_choice}", type="primary"):
                with st.spinner(f"Connecting to {api_choice}..."):
                    time.sleep(2)
                    
                    # Generate mock data for demo
                    np.random.seed(42)
                    n_events = np.random.randint(10, 50)
                    
                    api_data = {
                        'dt': np.abs(np.random.normal(0, 800, n_events)),
                        'dtheta': np.random.exponential(0.5, n_events),
                        'strength_ratio': np.random.exponential(2.2, n_events),
                        'event_id': [f"{api_choice.split()[0]}_{i:04d}" for i in range(n_events)],
                        'source': [api_choice] * n_events,
                        'timestamp': [datetime.now() - timedelta(hours=np.random.randint(0, 72)) for _ in range(n_events)],
                        'confidence': np.random.uniform(0.6, 0.98, n_events)
                    }
                    
                    df = pd.DataFrame(api_data)
                    st.session_state.current_data = df
                    
                    st.markdown(f"""
                    <div class="success-box">
                    <h4>✅ API Data Retrieved Successfully</h4>
                    <p>Source: {api_choice} | Events: {n_events} | Latest: {df['timestamp'].max().strftime('%Y-%m-%d %H:%M')}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    elif input_method == "📋 Manual Entry":
        st.markdown("### Enter event pair data manually:")
        
        with st.form("manual_entry"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                dt_val = st.number_input("Time difference (dt)", 0.0, 10000.0, 100.0)
            with col2:
                dtheta_val = st.number_input("Angular separation (dtheta)", 0.0, 10.0, 1.0)
            with col3:
                strength_val = st.number_input("Strength ratio", 0.0, 100.0, 2.0)
            
            if st.form_submit_button("➕ Add Entry"):
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
                
                st.success("✅ Entry added!")
    
    # Use data from session state or API
    if st.session_state.current_data is not None:
        df = st.session_state.current_data
    elif st.session_state.api_data is not None and st.checkbox("🌐 Use API data for analysis"):
        df = st.session_state.api_data
    
    # Data preview
    if df is not None:
        st.markdown("### 📋 Data Preview")
        
        # Enhanced metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📊 Total Pairs", len(df))
        with col2:
            st.metric("🔧 Features", len(df.columns))
        with col3:
            if 'strength_ratio' in df.columns:
                st.metric("💪 Avg Strength", f"{df['strength_ratio'].mean():.2f}")
            else:
                st.metric("📁 Data Ready", "✅")
        with col4:
            if 'dt' in df.columns:
                st.metric("⏱️ Avg Time Δ", f"{df['dt'].mean():.0f}s")
            else:
                st.metric("🔄 Processing", "✅")
        with col5:
            if 'dtheta' in df.columns:
                st.metric("📐 Avg Angle", f"{df['dtheta'].mean():.2f}°")
            else:
                st.metric("🎯 Ready", "✅")
        
        # Interactive data table
        st.dataframe(
            df.head(10), 
            use_container_width=True,
            column_config={
                "dt": st.column_config.NumberColumn("Time Δ (s)", format="%.1f"),
                "dtheta": st.column_config.NumberColumn("Angular Sep (°)", format="%.3f"),
                "strength_ratio": st.column_config.NumberColumn("Strength Ratio", format="%.2f"),
            }
        )

with tab2:
    st.markdown("## 🔬 AI Analysis & Prediction")
    
    if df is not None and model is not None:
        analysis_col1, analysis_col2 = st.columns([2, 1])
        
        with analysis_col1:
            if st.button("🚀 Run AI Analysis", type="primary", use_container_width=True):
                with st.spinner("🤖 AI processing multimessenger events..."):
                    try:
                        # Run prediction
                        results = predict_df(df, model, scaler, threshold)
                        st.session_state.results = results
                        st.session_state.analysis_complete = True  # Mark analysis stage as complete
                        
                        if results is not None:
                            st.markdown("""
                            <div class="success-box">
                            <h4>✅ AI Analysis Complete!</h4>
                            <p>Advanced multimessenger event associations detected and analyzed</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Enhanced results summary
                            st.markdown("### 📈 Analysis Results")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            associated = (results['pred_prob'] > threshold).sum()
                            total_pairs = len(results)
                            max_prob = results['pred_prob'].max()
                            avg_prob = results['pred_prob'].mean()
                            
                            with col1:
                                st.metric(
                                    "🎯 Associated Events", 
                                    associated,
                                    delta=f"{(associated/total_pairs)*100:.1f}% of total"
                                )
                            with col2:
                                st.metric(
                                    "🏆 Max Confidence", 
                                    f"{max_prob:.3f}",
                                    delta="Peak association"
                                )
                            with col3:
                                st.metric(
                                    "📊 Average Confidence", 
                                    f"{avg_prob:.3f}",
                                    delta="Overall score"
                                )
                            with col4:
                                reliability = "High" if avg_prob > 0.7 else "Medium" if avg_prob > 0.4 else "Low"
                                st.metric(
                                    "🔍 Reliability", 
                                    reliability,
                                    delta="Model confidence"
                                )
                        
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {e}")
                        if show_debug:
                            st.exception(e)
        
        with analysis_col2:
            st.markdown("### 🎛️ Analysis Settings")
            st.markdown(f"""
            **Current Configuration:**
            - Model: `{model_choice}`
            - Threshold: `{threshold}`
            - Data points: `{len(df) if df is not None else 0}`
            - Features: `{len(df.columns) if df is not None else 0}`
            """)
            
            if metadata:
                st.markdown(f"""
                **Model Performance:**
                - Algorithm: `{metadata.get('best_model', 'Unknown')}`
                - AUC Score: `{metadata.get('best_auc', 'N/A'):.3f}`
                """)
    
    elif df is None:
        st.info("👆 Please load data in the Data Input tab first")
    elif model is None:
        st.info("👆 Please select a model in the sidebar first")
    
    # Display detailed results if available
    if st.session_state.results is not None:
        st.markdown("---")
        st.markdown("### 🎯 Enhanced Event Classification Results")
        
        results = st.session_state.results
        
        # Enhanced summary with confidence breakdown
        st.markdown("#### 📊 Classification Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Confidence level breakdown
        very_high = (results['confidence_level'] == 'Very High').sum()
        high = (results['confidence_level'] == 'High').sum()
        moderate = (results['confidence_level'] == 'Moderate').sum()
        low = (results['confidence_level'] == 'Low').sum()
        
        with col1:
            st.metric("🔥 Very High Confidence", very_high, delta=f"{(very_high/len(results)*100):.1f}%")
        with col2:
            st.metric("🎯 High Confidence", high, delta=f"{(high/len(results)*100):.1f}%")
        with col3:
            st.metric("⚡ Moderate Confidence", moderate, delta=f"{(moderate/len(results)*100):.1f}%")
        with col4:
            st.metric("📊 Low Confidence", low, delta=f"{(low/len(results)*100):.1f}%")
        
        # Detailed event analysis
        st.markdown("#### 🔬 Individual Event Analysis")
        
        # Create expandable sections for each event
        for idx, row in results.iterrows():
            prob = row['pred_prob']
            confidence = row['confidence_level']
            classification = row['event_classification']
            reasoning = row['physical_reasoning']
            risk = row['risk_assessment']
            
            # Color code based on confidence
            if confidence == 'Very High':
                color = '#28a745'  # Green
            elif confidence == 'High':
                color = '#20c997'  # Teal
            elif confidence == 'Moderate':
                color = '#ffc107'  # Yellow
            else:
                color = '#dc3545'  # Red
            
            with st.expander(f"Event Pair {idx+1}: {classification} (Confidence: {prob:.3f})", expanded=False):
                
                # Two column layout for details
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {color}20, {color}10); 
                                padding: 1rem; border-radius: 10px; border-left: 4px solid {color};">
                    <h5 style="color: {color}; margin-bottom: 0.5rem;">📊 Classification Details</h5>
                    <p><strong>Probability:</strong> {prob:.4f}</p>
                    <p><strong>Confidence:</strong> {confidence}</p>
                    <p><strong>Status:</strong> {classification}</p>
                    <p><strong>Risk Assessment:</strong> {risk}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with detail_col2:
                    st.markdown(f"""
                    <div style="background: rgba(255, 255, 255, 0.1); 
                                padding: 1rem; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h5 style="color: #ff6b9d; margin-bottom: 0.5rem;">🔬 Physical Analysis</h5>
                    <p style="font-size: 0.9em; line-height: 1.4;">{reasoning}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show raw measurements
                st.markdown("**🔢 Raw Measurements:**")
                meas_col1, meas_col2, meas_col3 = st.columns(3)
                
                with meas_col1:
                    st.metric("Time Difference", f"{row.get('dt', 0):.1f} s")
                with meas_col2:
                    st.metric("Angular Separation", f"{row.get('dtheta', 0):.3f} °")
                with meas_col3:
                    st.metric("Strength Ratio", f"{row.get('strength_ratio', 0):.2f}")
                
                # Event type information if available
                if 'm1' in row and 'm2' in row:
                    st.markdown(f"**🌌 Event Types:** {row['m1']} ↔ {row['m2']}")
        
        # Summary statistics
        st.markdown("#### 📈 Statistical Summary")
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.markdown("""
            <div class="info-box">
            <h5>🎯 Association Statistics</h5>
            </div>
            """, unsafe_allow_html=True)
            
            total_associated = (results['pred_prob'] > threshold).sum()
            association_rate = (total_associated / len(results)) * 100
            avg_confidence = results['pred_prob'].mean()
            
            st.write(f"**Total Associations:** {total_associated} out of {len(results)} pairs")
            st.write(f"**Association Rate:** {association_rate:.1f}%")
            st.write(f"**Average Confidence:** {avg_confidence:.3f}")
            st.write(f"**Threshold Used:** {threshold}")
        
        with summary_col2:
            st.markdown("""
            <div class="info-box">
            <h5>🔍 Quality Assessment</h5>
            </div>
            """, unsafe_allow_html=True)
            
            high_quality = ((results['confidence_level'] == 'Very High') | 
                           (results['confidence_level'] == 'High')).sum()
            low_risk = (results['risk_assessment'] == '🟢 High Confidence').sum()
            
            st.write(f"**High Quality Classifications:** {high_quality} ({(high_quality/len(results)*100):.1f}%)")
            st.write(f"**Low Risk Classifications:** {low_risk} ({(low_risk/len(results)*100):.1f}%)")
            st.write(f"**Standard Deviation:** {results['pred_prob'].std():.3f}")
            st.write(f"**Confidence Range:** {results['pred_prob'].min():.3f} - {results['pred_prob'].max():.3f}")
        
        # Interactive results table with enhanced columns
        st.markdown("#### 📋 Complete Results Table")
        
        display_columns = ['event_classification', 'pred_prob', 'confidence_level', 
                          'confidence_description', 'risk_assessment', 'dt', 'dtheta', 'strength_ratio']
        available_display_cols = [col for col in display_columns if col in results.columns]
        
        st.dataframe(
            results[available_display_cols],
            use_container_width=True,
            column_config={
                "pred_prob": st.column_config.ProgressColumn(
                    "Association Probability",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.4f"
                ),
                "event_classification": st.column_config.TextColumn("Classification"),
                "confidence_level": st.column_config.TextColumn("Confidence Level"),
                "confidence_description": st.column_config.TextColumn("Description"),
                "risk_assessment": st.column_config.TextColumn("Risk Assessment"),
                "dt": st.column_config.NumberColumn("Time Δ (s)", format="%.1f"),
                "dtheta": st.column_config.NumberColumn("Angular Sep (°)", format="%.3f"),
                "strength_ratio": st.column_config.NumberColumn("Strength Ratio", format="%.2f"),
            }
        )
        
        # Download functionality section
        st.markdown("#### 📥 Download Analysis Results")
        
        download_col1, download_col2, download_col3 = st.columns(3)
        
        with download_col1:
            # CSV Download
            csv_data = results.to_csv(index=False)
            st.download_button(
                label="📊 Download CSV",
                data=csv_data,
                file_name=f"multimessenger_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with download_col2:
            # JSON Download
            json_data = results.to_json(orient='records', indent=2, date_format='iso')
            st.download_button(
                label="📋 Download JSON",
                data=json_data,
                file_name=f"multimessenger_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with download_col3:
            # Detailed Report Download
            def generate_detailed_report(results_df):
                """Generate a comprehensive analysis report"""
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Calculate summary statistics
                total_pairs = len(results_df)
                same_events = (results_df['event_classification'] == 'Same Event').sum()
                different_events = (results_df['event_classification'] == 'Different Events').sum()
                high_conf = ((results_df['confidence_level'] == 'Very High') | 
                           (results_df['confidence_level'] == 'High')).sum()
                avg_prob = results_df['pred_prob'].mean()
                
                report = f"""# Multimessenger AI Analysis Report
Generated: {timestamp}

## Executive Summary
- **Total Event Pairs Analyzed**: {total_pairs}
- **Same Events Detected**: {same_events} ({same_events/total_pairs*100:.1f}%)
- **Different Events**: {different_events} ({different_events/total_pairs*100:.1f}%)
- **High Confidence Classifications**: {high_conf} ({high_conf/total_pairs*100:.1f}%)
- **Average Association Probability**: {avg_prob:.3f}

## Confidence Distribution
- **Very High Confidence**: {(results_df['confidence_level'] == 'Very High').sum()} events
- **High Confidence**: {(results_df['confidence_level'] == 'High').sum()} events
- **Moderate Confidence**: {(results_df['confidence_level'] == 'Moderate').sum()} events
- **Low Confidence**: {(results_df['confidence_level'] == 'Low').sum()} events

## Statistical Analysis
- **Mean Probability**: {results_df['pred_prob'].mean():.4f}
- **Standard Deviation**: {results_df['pred_prob'].std():.4f}
- **Minimum Probability**: {results_df['pred_prob'].min():.4f}
- **Maximum Probability**: {results_df['pred_prob'].max():.4f}
- **Median Probability**: {results_df['pred_prob'].median():.4f}

## Detailed Classifications
"""
                
                for idx, row in results_df.iterrows():
                    report += f"""
### Event Pair {idx+1}
- **Classification**: {row['event_classification']}
- **Probability**: {row['pred_prob']:.4f}
- **Confidence Level**: {row['confidence_level']}
- **Risk Assessment**: {row['risk_assessment']}
"""
                    if 'physical_reasoning' in row:
                        report += f"- **Physical Reasoning**: {row['physical_reasoning']}\n"
                    if 'dt' in row:
                        report += f"- **Time Difference**: {row['dt']:.2f} seconds\n"
                    if 'dtheta' in row:
                        report += f"- **Angular Separation**: {row['dtheta']:.4f} degrees\n"
                
                report += f"""

## Summary & Recommendations

Based on the analysis of {total_pairs} event pairs:

1. **Data Quality**: {high_conf/total_pairs*100:.1f}% of classifications have high confidence
2. **Association Rate**: {same_events/total_pairs*100:.1f}% of events appear to be associated
3. **Follow-up Priority**: Focus on {(results_df['confidence_level'] == 'Very High').sum()} very high confidence events

Generated by Multimessenger AI Platform
NASA Space Apps Challenge Project
"""
                return report
            
            report_text = generate_detailed_report(results)
            st.download_button(
                label="📝 Download Report",
                data=report_text,
                file_name=f"multimessenger_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        # Clear functionality section
        st.markdown("---")
        st.markdown("#### 🗑️ Clear Analysis Data")
        
        clear_col1, clear_col2 = st.columns([1, 1])
        
        with clear_col1:
            if st.button("🗑️ Clear Results", type="secondary", use_container_width=True):
                # Clear session state
                if 'results' in st.session_state:
                    del st.session_state.results
                if 'uploaded_data' in st.session_state:
                    del st.session_state.uploaded_data
                if 'demo_data' in st.session_state:
                    del st.session_state.demo_data
                if 'analysis_complete' in st.session_state:
                    del st.session_state.analysis_complete
                st.success("✅ Analysis results cleared!")
                st.rerun()
        
        with clear_col2:
            if st.button("🔄 Reset All Data", type="primary", use_container_width=True):
                # Clear all session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.success("✅ All data reset! Page will refresh...")
                st.rerun()
    
    else:
        st.markdown("---")
        st.info("👆 **Run AI analysis first to see results and download options**")
        
        # Clear functionality for when no results exist
        st.markdown("#### 🗑️ Clear Session Data")
        if st.button("🔄 Reset Session", type="secondary", use_container_width=True):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("✅ Session reset! Page will refresh...")
            st.rerun()

with tab3:
    st.markdown("## 📈 Advanced Visualizations")
    
    if st.session_state.results is not None:
        results = st.session_state.results
        
        # Visualization options
        viz_col1, viz_col2 = st.columns([1, 3])
        
        with viz_col1:
            st.markdown("### 🎨 Visualization Options")
            viz_type = st.selectbox(
                "Choose visualization:",
                [
                    "📊 Probability Distribution",
                    "🎯 3D Feature Space", 
                    "📈 Correlation Matrix",
                    "⏱️ Time Series Analysis",
                    "🌌 Sky Map (3D)",
                    "📋 Statistical Summary",
                    "🔥 Density Heatmap",
                    "📈 Pair-wise Scatter",
                    "🎪 Box Plot Analysis",
                    "📊 Violin Plot Distributions",
                    "🌊 Ridge Plot",
                    "📉 ROC Curve Analysis"
                ]
            )
            
            # Customization options
            color_scheme = st.selectbox("Color scheme:", ["Viridis", "Plasma", "Cividis", "Magma", "Turbo", "Purples"])
            show_threshold = st.checkbox("Show threshold line", True)
            interactive_mode = st.checkbox("Interactive mode", True)
            
            # Map color scheme names to plotly color scales
            color_map = {
                "Viridis": "Viridis",
                "Plasma": "Plasma", 
                "Cividis": "Cividis",
                "Magma": "Magma",
                "Turbo": "Turbo",
                "Purples": "Purples"
            }
            plotly_color_scheme = color_map.get(color_scheme, "Viridis")
            
        with viz_col2:
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
                    fig.add_vline(
                        x=threshold, 
                        line_dash="dash", 
                        line_color="red",
                        annotation_text=f"Threshold ({threshold})"
                    )
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "🎯 3D Feature Space":
                fig = px.scatter_3d(
                    results,
                    x='dt',
                    y='dtheta', 
                    z='strength_ratio',
                    color='pred_prob',
                    size='pred_prob',
                    title="🌌 3D Feature Space Visualization",
                    labels={
                        'dt': 'Time Difference (s)',
                        'dtheta': 'Angular Separation (°)',
                        'strength_ratio': 'Strength Ratio',
                        'pred_prob': 'Association Probability'
                    },
                    color_continuous_scale=plotly_color_scheme
                )
                
                fig.update_layout(
                    scene=dict(
                        bgcolor='rgba(0,0,0,0)',
                        xaxis_title="Time Difference (s)",
                        yaxis_title="Angular Separation (°)",
                        zaxis_title="Strength Ratio"
                    ),
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "📈 Correlation Matrix":
                # Select numeric columns for correlation
                numeric_cols = ['dt', 'dtheta', 'strength_ratio', 'pred_prob']
                available_cols = [col for col in numeric_cols if col in results.columns]
                
                if len(available_cols) > 1:
                    corr_matrix = results[available_cols].corr()
                    
                    fig = px.imshow(
                        corr_matrix,
                        title="🔗 Feature Correlation Matrix",
                        color_continuous_scale=plotly_color_scheme,
                        aspect="auto"
                    )
                    
                    # Add correlation values as text
                    fig.update_traces(
                        text=np.around(corr_matrix.values, decimals=2),
                        texttemplate="%{text}",
                        textfont={"size": 12}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Not enough numeric columns for correlation analysis")
            
            elif viz_type == "⏱️ Time Series Analysis":
                # Check for existing time columns or create synthetic time data
                if 'detection_time' in results.columns or 'timestamp' in results.columns:
                    time_col = 'detection_time' if 'detection_time' in results.columns else 'timestamp'
                    
                    # Time-based analysis
                    time_results = results.copy()
                    time_results['hour'] = pd.to_datetime(time_results[time_col]).dt.hour
                    time_results['day'] = pd.to_datetime(time_results[time_col]).dt.date
                    
                    # Hourly distribution
                    hourly_assoc = time_results.groupby('hour')['pred_prob'].mean().reset_index()
                    
                    fig = px.line(
                        hourly_assoc,
                        x='hour',
                        y='pred_prob',
                        title="⏰ Association Probability by Hour",
                        labels={'hour': 'Hour of Day', 'pred_prob': 'Average Association Probability'},
                        line_shape="spline"
                    )
                    
                    fig.update_traces(line_color='#7c4dff', line_width=3)
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Create synthetic time series for demonstration
                    st.info("📊 Creating synthetic time series analysis for demonstration")
                    
                    # Generate synthetic time data based on data indices
                    time_results = results.copy()
                    base_time = datetime.now() - timedelta(hours=len(results))
                    time_results['synthetic_time'] = [base_time + timedelta(hours=i) for i in range(len(results))]
                    time_results['hour'] = time_results['synthetic_time'].dt.hour
                    
                    # Create time series plot
                    fig = px.line(
                        time_results.reset_index(),
                        x='index',
                        y='pred_prob',
                        title="⏰ Association Probability Over Event Sequence",
                        labels={'index': 'Event Sequence', 'pred_prob': 'Association Probability'},
                        line_shape="spline"
                    )
                    
                    # Add threshold line
                    if show_threshold:
                        fig.add_hline(
                            y=threshold, 
                            line_dash="dash", 
                            line_color="#d1c4e9",
                            annotation_text=f"Threshold ({threshold})"
                        )
                    
                    fig.update_traces(line_color='#7c4dff', line_width=3)
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Also show hourly pattern
                    hourly_pattern = time_results.groupby('hour')['pred_prob'].mean().reset_index()
                    
                    fig2 = px.bar(
                        hourly_pattern,
                        x='hour',
                        y='pred_prob',
                        title="📊 Average Association Probability by Hour",
                        labels={'hour': 'Hour of Day', 'pred_prob': 'Average Association Probability'}
                    )
                    
                    fig2.update_traces(marker_color='#7c4dff')
                    fig2.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
            
            elif viz_type == "🌌 Sky Map (3D)":
                # Create a mock sky map using angular separation
                if 'dtheta' in results.columns:
                    # Convert to spherical coordinates for sky visualization
                    phi = np.random.uniform(0, 2*np.pi, len(results))
                    theta = results['dtheta'].values
                    
                    # Convert to cartesian for 3D plotting
                    x = np.sin(theta) * np.cos(phi)
                    y = np.sin(theta) * np.sin(phi)
                    z = np.cos(theta)
                    
                    fig = go.Figure(data=[
                        go.Scatter3d(
                            x=x, y=y, z=z,
                            mode='markers',
                            marker=dict(
                                size=8,
                                color=results['pred_prob'],
                                colorscale=plotly_color_scheme,
                                colorbar=dict(title="Association Probability"),
                                showscale=True
                            ),
                            text=[f"Prob: {p:.3f}<br>θ: {t:.3f}°" for p, t in zip(results['pred_prob'], results['dtheta'])],
                            hovertemplate="<b>Event Pair</b><br>%{text}<extra></extra>"
                        )
                    ])
                    
                    fig.update_layout(
                        title="🌌 3D Sky Map of Event Associations",
                        scene=dict(
                            xaxis_title="X (RA direction)",
                            yaxis_title="Y (Dec direction)", 
                            zaxis_title="Z (Celestial pole)",
                            bgcolor='rgba(0,0,0,0.9)',
                            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                        ),
                        height=600,
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Angular separation data not available for sky map")
            
            elif viz_type == "🔥 Density Heatmap":
                if 'dt' in results.columns and 'dtheta' in results.columns:
                    fig = px.density_heatmap(
                        results,
                        x='dt',
                        y='dtheta',
                        z='pred_prob',
                        title="🔥 Event Density Heatmap",
                        labels={
                            'dt': 'Time Difference (s)',
                            'dtheta': 'Angular Separation (°)',
                            'pred_prob': 'Association Probability'
                        },
                        color_continuous_scale=plotly_color_scheme
                    )
                    
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Time and angular data required for heatmap")
            
            elif viz_type == "📈 Pair-wise Scatter":
                numeric_cols = ['dt', 'dtheta', 'strength_ratio', 'pred_prob']
                available_cols = [col for col in numeric_cols if col in results.columns]
                
                if len(available_cols) >= 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        x_axis = st.selectbox("X-axis:", available_cols, index=0)
                    with col2:
                        y_axis = st.selectbox("Y-axis:", available_cols, index=1)
                    
                    fig = px.scatter(
                        results,
                        x=x_axis,
                        y=y_axis,
                        color='pred_prob',
                        size='pred_prob',
                        title=f"🎯 {x_axis.title()} vs {y_axis.title()}",
                        color_continuous_scale=plotly_color_scheme,
                        hover_data=['dt', 'dtheta', 'strength_ratio']
                    )
                    
                    # Add trend line
                    if st.checkbox("Show trend line", False):
                        from sklearn.linear_model import LinearRegression
                        X_trend = results[[x_axis]].values
                        y_trend = results[y_axis].values
                        reg = LinearRegression().fit(X_trend, y_trend)
                        trend_y = reg.predict(X_trend)
                        
                        fig.add_trace(go.Scatter(
                            x=results[x_axis],
                            y=trend_y,
                            mode='lines',
                            name='Trend Line',
                            line=dict(color='red', dash='dash')
                        ))
                    
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Not enough numeric columns for scatter plot")
            
            elif viz_type == "🎪 Box Plot Analysis":
                numeric_cols = ['dt', 'dtheta', 'strength_ratio', 'pred_prob']
                available_cols = [col for col in numeric_cols if col in results.columns]
                
                if available_cols:
                    selected_features = st.multiselect(
                        "Select features to analyze:",
                        available_cols,
                        default=available_cols[:3]
                    )
                    
                    if selected_features:
                        # Create subplot for multiple box plots
                        fig = make_subplots(
                            rows=1, cols=len(selected_features),
                            subplot_titles=selected_features,
                            horizontal_spacing=0.1
                        )
                        
                        for i, feature in enumerate(selected_features):
                            fig.add_trace(
                                go.Box(
                                    y=results[feature],
                                    name=feature,
                                    boxpoints='outliers',
                                    marker_color=px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)]
                                ),
                                row=1, col=i+1
                            )
                        
                        fig.update_layout(
                            title="🎪 Feature Distribution Box Plots",
                            height=500,
                            showlegend=False,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No numeric columns available for box plot")
            
            elif viz_type == "📊 Violin Plot Distributions":
                numeric_cols = ['dt', 'dtheta', 'strength_ratio', 'pred_prob']
                available_cols = [col for col in numeric_cols if col in results.columns]
                
                if available_cols:
                    selected_feature = st.selectbox("Select feature:", available_cols)
                    
                    # Group by prediction label for comparison
                    results_copy = results.copy()
                    results_copy['prediction'] = results_copy['pred_prob'].apply(
                        lambda x: 'Same Event' if x > threshold else 'Different Events'
                    )
                    
                    fig = px.violin(
                        results_copy,
                        y=selected_feature,
                        x='prediction',
                        color='prediction',
                        title=f"📊 {selected_feature.title()} Distribution by Prediction",
                        color_discrete_map={
                            'Same Event': '#7c4dff',
                            'Different Events': '#d1c4e9'
                        }
                    )
                    
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No numeric columns available for violin plot")
            
            elif viz_type == "🌊 Ridge Plot":
                # Create ridge plot using multiple histograms
                numeric_cols = ['dt', 'dtheta', 'strength_ratio']
                available_cols = [col for col in numeric_cols if col in results.columns]
                
                if available_cols:
                    fig = make_subplots(
                        rows=len(available_cols), cols=1,
                        subplot_titles=[f"{col.title()} Distribution" for col in available_cols],
                        vertical_spacing=0.15
                    )
                    
                    for i, col in enumerate(available_cols):
                        fig.add_trace(
                            go.Histogram(
                                x=results[col],
                                name=col,
                                opacity=0.7,
                                nbinsx=30,
                                marker_color=px.colors.sequential.Viridis[i * 3]
                            ),
                            row=i+1, col=1
                        )
                    
                    fig.update_layout(
                        title="🌊 Ridge Plot - Feature Distributions",
                        height=200 * len(available_cols),
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No numeric columns available for ridge plot")
            
            elif viz_type == "📉 ROC Curve Analysis":
                if 'pred_prob' in results.columns:
                    # For demonstration, create synthetic true labels based on features
                    # In real scenario, you'd have ground truth labels
                    st.info("🔬 **Demo Mode**: Generating synthetic ground truth for ROC analysis")
                    
                    # Create synthetic true labels based on multiple criteria
                    synthetic_true = (
                        (results['dt'] < results['dt'].median()) & 
                        (results['dtheta'] < results['dtheta'].median()) &
                        (results['strength_ratio'] > results['strength_ratio'].median())
                    ).astype(int)
                    
                    # Calculate ROC curve points
                    from sklearn.metrics import roc_curve, auc
                    fpr, tpr, thresholds = roc_curve(synthetic_true, results['pred_prob'])
                    roc_auc = auc(fpr, tpr)
                    
                    # Create ROC curve plot
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f'ROC Curve (AUC = {roc_auc:.3f})',
                        line=dict(color='#ff6b9d', width=3)
                    ))
                    
                    # Add diagonal reference line
                    fig.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='lines',
                        name='Random Classifier',
                        line=dict(color='gray', dash='dash')
                    ))
                    
                    fig.update_layout(
                        title='📉 ROC Curve Analysis',
                        xaxis_title='False Positive Rate',
                        yaxis_title='True Positive Rate',
                        height=500,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # ROC statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("AUC Score", f"{roc_auc:.3f}")
                    with col2:
                        optimal_idx = np.argmax(tpr - fpr)
                        optimal_threshold = thresholds[optimal_idx]
                        st.metric("Optimal Threshold", f"{optimal_threshold:.3f}")
                    with col3:
                        sensitivity = tpr[optimal_idx]
                        specificity = 1 - fpr[optimal_idx]
                        st.metric("Sensitivity", f"{sensitivity:.3f}")
                else:
                    st.warning("Prediction probabilities required for ROC analysis")
            
            elif viz_type == "📋 Statistical Summary":
                st.markdown("### 📊 Comprehensive Statistical Analysis")
                
                # Main statistics tabs
                stat_tab1, stat_tab2, stat_tab3, stat_tab4 = st.tabs([
                    "📈 Descriptive Statistics", 
                    "🔬 Hypothesis Testing", 
                    "📊 Distribution Analysis", 
                    "🔗 Correlation Analysis"
                ])
                
                with stat_tab1:
                    # Descriptive Statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**🎯 Association Statistics:**")
                        total_events = len(results)
                        associated_events = (results['pred_prob'] > threshold).sum()
                        association_rate = (associated_events / total_events) * 100
                        
                        st.metric("Total Event Pairs", total_events)
                        st.metric("Associated Pairs", f"{associated_events} ({association_rate:.1f}%)")
                        st.metric("Mean Probability", f"{results['pred_prob'].mean():.3f}")
                        st.metric("Std Deviation", f"{results['pred_prob'].std():.3f}")
                    
                    with col2:
                        st.markdown("**📈 Distribution Properties:**")
                        
                        # Quartiles
                        q1 = results['pred_prob'].quantile(0.25)
                        q2 = results['pred_prob'].quantile(0.50)
                        q3 = results['pred_prob'].quantile(0.75)
                        
                        st.metric("Q1 (25th percentile)", f"{q1:.3f}")
                        st.metric("Q2 (Median)", f"{q2:.3f}")
                        st.metric("Q3 (75th percentile)", f"{q3:.3f}")
                        st.metric("IQR", f"{q3-q1:.3f}")
                    
                    with col3:
                        st.markdown("**🔍 Advanced Metrics:**")
                        
                        # Skewness and Kurtosis
                        skewness = stats.skew(results['pred_prob'])
                        kurtosis_val = stats.kurtosis(results['pred_prob'])
                        
                        st.metric("Skewness", f"{skewness:.3f}")
                        st.metric("Kurtosis", f"{kurtosis_val:.3f}")
                        
                        # Coefficient of variation
                        cv = results['pred_prob'].std() / results['pred_prob'].mean()
                        st.metric("Coeff. of Variation", f"{cv:.3f}")
                        
                        # Range
                        data_range = results['pred_prob'].max() - results['pred_prob'].min()
                        st.metric("Range", f"{data_range:.3f}")
                    
                    # Confidence intervals
                    st.markdown("**📊 Confidence Intervals:**")
                    mean_prob = results['pred_prob'].mean()
                    std_prob = results['pred_prob'].std()
                    n = len(results)
                    
                    # Multiple confidence levels
                    confidence_levels = [90, 95, 99]
                    z_scores = [1.645, 1.96, 2.576]
                    
                    ci_data = []
                    for conf_level, z in zip(confidence_levels, z_scores):
                        margin_error = z * (std_prob / np.sqrt(n))
                        ci_lower = mean_prob - margin_error
                        ci_upper = mean_prob + margin_error
                        ci_data.append({
                            'Confidence Level': f"{conf_level}%",
                            'Lower Bound': f"{ci_lower:.3f}",
                            'Upper Bound': f"{ci_upper:.3f}",
                            'Margin of Error': f"{margin_error:.3f}"
                        })
                    
                    ci_df = pd.DataFrame(ci_data)
                    st.dataframe(ci_df, use_container_width=True)
                
                with stat_tab2:
                    # Hypothesis Testing
                    st.markdown("**🧪 Hypothesis Testing Suite**")
                    
                    # Test selection
                    test_type = st.selectbox(
                        "Select hypothesis test:",
                        [
                            "One-sample t-test (μ = 0.5)",
                            "Normality test (Shapiro-Wilk)",
                            "Homogeneity test (Levene)",
                            "Two-sample comparison",
                            "Chi-square goodness of fit"
                        ]
                    )
                    
                    if test_type == "One-sample t-test (μ = 0.5)":
                        # Test if mean probability significantly different from 0.5
                        from scipy.stats import ttest_1samp
                        
                        t_stat, p_value = ttest_1samp(results['pred_prob'], 0.5)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("T-statistic", f"{t_stat:.4f}")
                        with col2:
                            st.metric("P-value", f"{p_value:.4f}")
                        with col3:
                            significance = "Significant" if p_value < 0.05 else "Not Significant"
                            st.metric("Result (α=0.05)", significance)
                        
                        st.info(f"**Interpretation:** The mean probability ({mean_prob:.3f}) is {'significantly' if p_value < 0.05 else 'not significantly'} different from 0.5")
                    
                    elif test_type == "Normality test (Shapiro-Wilk)":
                        from scipy.stats import shapiro
                        
                        # Sample for Shapiro-Wilk (max 5000 samples)
                        sample_data = results['pred_prob'].sample(min(len(results), 5000))
                        stat, p_value = shapiro(sample_data)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("W-statistic", f"{stat:.4f}")
                        with col2:
                            st.metric("P-value", f"{p_value:.4f}")
                        
                        normality = "Normal" if p_value > 0.05 else "Not Normal"
                        st.info(f"**Result:** Data appears to be **{normality}** (α=0.05)")
                    
                    elif test_type == "Two-sample comparison":
                        # Compare high vs low probability groups
                        high_prob = results[results['pred_prob'] > threshold]['pred_prob']
                        low_prob = results[results['pred_prob'] <= threshold]['pred_prob']
                        
                        if len(high_prob) > 0 and len(low_prob) > 0:
                            from scipy.stats import ttest_ind
                            t_stat, p_value = ttest_ind(high_prob, low_prob)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("High Group Mean", f"{high_prob.mean():.3f}")
                            with col2:
                                st.metric("Low Group Mean", f"{low_prob.mean():.3f}")
                            with col3:
                                st.metric("P-value", f"{p_value:.4f}")
                            
                            st.info(f"**Result:** Groups are {'significantly' if p_value < 0.05 else 'not significantly'} different")
                        else:
                            st.warning("Need both high and low probability groups for comparison")
                    
                    elif test_type == "Chi-square goodness of fit":
                        # Test if distribution follows uniform distribution
                        from scipy.stats import chisquare
                        
                        # Create bins
                        n_bins = 10
                        observed, bin_edges = np.histogram(results['pred_prob'], bins=n_bins)
                        expected = np.full(n_bins, len(results) / n_bins)
                        
                        chi2_stat, p_value = chisquare(observed, expected)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Chi-square statistic", f"{chi2_stat:.4f}")
                        with col2:
                            st.metric("P-value", f"{p_value:.4f}")
                        
                        uniform_fit = "Uniform" if p_value > 0.05 else "Not Uniform"
                        st.info(f"**Result:** Data {'fits' if p_value > 0.05 else 'does not fit'} a uniform distribution")
                
                with stat_tab3:
                    # Distribution Analysis
                    st.markdown("**📊 Distribution Fitting & Analysis**")
                    
                    # Distribution fitting
                    dist_col1, dist_col2 = st.columns(2)
                    
                    with dist_col1:
                        st.markdown("**Distribution Parameters:**")
                        
                        # Fit common distributions
                        from scipy import stats as scipy_stats
                        
                        distributions = ['norm', 'beta', 'gamma', 'uniform']
                        best_dist = None
                        best_aic = np.inf
                        
                        fit_results = []
                        for dist_name in distributions:
                            dist = getattr(scipy_stats, dist_name)
                            try:
                                params = dist.fit(results['pred_prob'])
                                log_likelihood = np.sum(dist.logpdf(results['pred_prob'], *params))
                                aic = 2 * len(params) - 2 * log_likelihood
                                
                                fit_results.append({
                                    'Distribution': dist_name.title(),
                                    'AIC': f"{aic:.2f}",
                                    'Log-Likelihood': f"{log_likelihood:.2f}",
                                    'Parameters': str([f"{p:.3f}" for p in params])
                                })
                                
                                if aic < best_aic:
                                    best_aic = aic
                                    best_dist = dist_name
                            except:
                                continue
                        
                        fit_df = pd.DataFrame(fit_results)
                        st.dataframe(fit_df, use_container_width=True)
                        
                        if best_dist:
                            st.success(f"**Best Fit:** {best_dist.title()} distribution (lowest AIC)")
                    
                    with dist_col2:
                        st.markdown("**Outlier Detection:**")
                        
                        # Z-score method
                        z_scores = np.abs(stats.zscore(results['pred_prob']))
                        outliers_z = (z_scores > 3).sum()
                        
                        # IQR method
                        q1 = results['pred_prob'].quantile(0.25)
                        q3 = results['pred_prob'].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        outliers_iqr = ((results['pred_prob'] < lower_bound) | 
                                       (results['pred_prob'] > upper_bound)).sum()
                        
                        st.metric("Z-score Outliers (|z| > 3)", outliers_z)
                        st.metric("IQR Outliers", outliers_iqr)
                        st.metric("Outlier Rate (Z-score)", f"{(outliers_z/len(results)*100):.1f}%")
                        st.metric("Outlier Rate (IQR)", f"{(outliers_iqr/len(results)*100):.1f}%")
                
                with stat_tab4:
                    # Correlation Analysis
                    st.markdown("**🔗 Advanced Correlation Analysis**")
                    
                    numeric_cols = ['dt', 'dtheta', 'strength_ratio', 'pred_prob']
                    available_cols = [col for col in numeric_cols if col in results.columns]
                    
                    if len(available_cols) >= 2:
                        # Correlation methods
                        corr_method = st.selectbox(
                            "Correlation method:",
                            ["Pearson", "Spearman", "Kendall"]
                        )
                        
                        # Calculate correlations
                        if corr_method == "Pearson":
                            corr_matrix = results[available_cols].corr(method='pearson')
                        elif corr_method == "Spearman":
                            corr_matrix = results[available_cols].corr(method='spearman')
                        else:
                            corr_matrix = results[available_cols].corr(method='kendall')
                        
                        # Display correlation matrix
                        st.markdown(f"**{corr_method} Correlation Matrix:**")
                        st.dataframe(corr_matrix.round(3), use_container_width=True)
                        
                        # Correlation significance testing
                        st.markdown("**Correlation Significance Tests:**")
                        
                        sig_results = []
                        for i, col1 in enumerate(available_cols):
                            for col2 in available_cols[i+1:]:
                                if corr_method == "Pearson":
                                    corr_coef, p_val = stats.pearsonr(results[col1], results[col2])
                                elif corr_method == "Spearman":
                                    corr_coef, p_val = stats.spearmanr(results[col1], results[col2])
                                else:
                                    corr_coef, p_val = stats.kendalltau(results[col1], results[col2])
                                
                                significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                                
                                sig_results.append({
                                    'Variable Pair': f"{col1} - {col2}",
                                    'Correlation': f"{corr_coef:.3f}",
                                    'P-value': f"{p_val:.4f}",
                                    'Significance': significance
                                })
                        
                        sig_df = pd.DataFrame(sig_results)
                        st.dataframe(sig_df, use_container_width=True)
                        
                        st.caption("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
                    else:
                        st.warning("Need at least 2 numeric columns for correlation analysis")
    
    else:
        st.info("👆 Run AI analysis first to generate visualizations")

with tab4:
    st.markdown("## 🎯 Event Clustering Analysis")
    st.markdown("*Detect single vs multiple event sources and analyze temporal/spatial clustering*")
    
    if st.session_state.results is not None:
        results = st.session_state.results
        
        cluster_col1, cluster_col2 = st.columns([1, 2])
        
        with cluster_col1:
            st.markdown("### ⚙️ Clustering Parameters")
            
            # Clustering options
            cluster_features = st.multiselect(
                "Features for clustering:",
                ['dt', 'dtheta', 'strength_ratio', 'pred_prob'],
                default=['dt', 'dtheta', 'strength_ratio']
            )
            
            if st.button("🔍 Perform Clustering Analysis", type="primary"):
                if len(cluster_features) >= 2:
                    with st.spinner("🔬 Analyzing event clusters..."):
                        # Prepare clustering data
                        cluster_data = results[cluster_features].copy()
                        
                        # Standardize features
                        scaler_cluster = StandardScaler()
                        cluster_scaled = scaler_cluster.fit_transform(cluster_data)
                        
                        # DBSCAN clustering
                        clustering = DBSCAN(eps=clustering_eps, min_samples=min_samples)
                        cluster_labels = clustering.fit_predict(cluster_scaled)
                        
                        # Add cluster results
                        results_clustered = results.copy()
                        results_clustered['cluster'] = cluster_labels
                        results_clustered['is_outlier'] = cluster_labels == -1
                        
                        st.session_state.clustering_results = results_clustered
                        
                        # Cluster statistics
                        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                        n_noise = list(cluster_labels).count(-1)
                        
                        st.success(f"✅ Found {n_clusters} event clusters and {n_noise} outliers")
                else:
                    st.warning("Please select at least 2 features for clustering")
        
        with cluster_col2:
            if st.session_state.clustering_results is not None:
                cluster_results = st.session_state.clustering_results
                
                # Cluster summary
                st.markdown("### 📊 Clustering Results")
                
                n_clusters = len(set(cluster_results['cluster'])) - (1 if -1 in cluster_results['cluster'].values else 0)
                n_outliers = (cluster_results['cluster'] == -1).sum()
                largest_cluster = cluster_results['cluster'].value_counts().iloc[0] if len(cluster_results) > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🎯 Event Clusters", n_clusters)
                with col2:
                    st.metric("👤 Isolated Events", n_outliers)
                with col3:
                    st.metric("👥 Largest Cluster", largest_cluster)
                
                # Cluster visualization
                if len(cluster_features) >= 2:
                    fig = px.scatter(
                        cluster_results,
                        x=cluster_features[0],
                        y=cluster_features[1],
                        color='cluster',
                        size='pred_prob',
                        title="🎯 Event Clustering Visualization",
                        labels={
                            cluster_features[0]: cluster_features[0].replace('_', ' ').title(),
                            cluster_features[1]: cluster_features[1].replace('_', ' ').title()
                        },
                        hover_data=['pred_prob']
                    )
                    
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Event source analysis
                st.markdown("### 🌟 Event Source Analysis")
                
                if n_clusters > 0:
                    # Analyze each cluster
                    for cluster_id in sorted(cluster_results['cluster'].unique()):
                        if cluster_id != -1:  # Skip outliers
                            cluster_data = cluster_results[cluster_results['cluster'] == cluster_id]
                            cluster_size = len(cluster_data)
                            avg_prob = cluster_data['pred_prob'].mean()
                            
                            with st.expander(f"🎯 Cluster {cluster_id} ({cluster_size} events)"):
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Events in Cluster", cluster_size)
                                with col2:
                                    st.metric("Avg Association Prob", f"{avg_prob:.3f}")
                                with col3:
                                    cluster_type = "Multi-event Source" if cluster_size > 3 else "Possible Coincidence"
                                    st.metric("Likely Source Type", cluster_type)
                                
                                # Temporal analysis for cluster
                                if 'dt' in cluster_data.columns:
                                    time_span = cluster_data['dt'].max() - cluster_data['dt'].min()
                                    st.info(f"⏱️ Temporal span: {time_span:.1f} seconds")
                                
                                # Spatial analysis for cluster  
                                if 'dtheta' in cluster_data.columns:
                                    spatial_span = cluster_data['dtheta'].max() - cluster_data['dtheta'].min()
                                    st.info(f"📐 Spatial span: {spatial_span:.3f} degrees")
                
                # Summary interpretation
                st.markdown("### 🔬 Scientific Interpretation")
                
                if n_clusters == 0:
                    st.info("🔍 **No significant clustering detected** - Events appear to be independent or noise-dominated")
                elif n_clusters == 1:
                    st.success("🎯 **Single source detected** - Events likely originate from the same astrophysical phenomenon")
                else:
                    st.warning(f"🌟 **Multiple sources detected** - {n_clusters} distinct astrophysical sources identified")
                
                if n_outliers > 0:
                    outlier_fraction = n_outliers / len(cluster_results) * 100
                    st.info(f"👤 **{outlier_fraction:.1f}% isolated events** - Possible instrumental noise or rare phenomena")
    
    else:
        st.info("👆 Run AI analysis first to enable clustering analysis")

with tab5:
    st.markdown("## 🌐 Real-time Event Monitor")
    st.markdown("*Live monitoring of multimessenger astronomical events*")
    
    # Real-time dashboard
    monitor_col1, monitor_col2 = st.columns([2, 1])
    
    with monitor_col1:
        st.markdown("### 📡 Live Event Stream")
        
        # Auto-refresh toggle
        if auto_refresh:
            # Create placeholder for live updates
            placeholder = st.empty()
            
            # Simulate real-time data stream
            for i in range(10):
                with placeholder.container():
                    current_time = datetime.now()
                    
                    # Mock real-time event
                    event_type = np.random.choice(['🌊 Gravitational Wave', '⚡ Gamma-ray Burst', '👻 Neutrino Detection'])
                    confidence = np.random.uniform(0.6, 0.95)
                    significance = np.random.uniform(3.0, 8.0)
                    
                    st.markdown(f"""
                    **⏰ {current_time.strftime('%H:%M:%S')} UTC**
                    
                    🚨 **NEW EVENT DETECTED**
                    - **Type**: {event_type}
                    - **Confidence**: {confidence:.2f}
                    - **Significance**: {significance:.1f}σ
                    - **Status**: 🔄 Processing for associations...
                    """)
                    
                    # Simulated processing bar
                    progress_bar = st.progress(0)
                    for j in range(100):
                        progress_bar.progress(j + 1)
                        time.sleep(0.01)
                    
                    st.success("✅ Event processed and added to database")
                
                time.sleep(2)  # Update every 2 seconds
        else:
            st.info("🔄 Enable auto-refresh in the sidebar to see live event stream")
            
            # Static event log
            st.markdown("### 📋 Recent Events Log")
            
            # Mock recent events
            recent_events = []
            for i in range(10):
                event = {
                    'Time': datetime.now() - timedelta(minutes=i*15),
                    'Type': np.random.choice(['GW', 'GRB', 'Neutrino', 'Optical']),
                    'Confidence': np.random.uniform(0.5, 0.95),
                    'Associated': np.random.choice(['Yes', 'No', 'Pending'])
                }
                recent_events.append(event)
            
            events_df = pd.DataFrame(recent_events)
            st.dataframe(events_df, use_container_width=True)
    
    with monitor_col2:
        st.markdown("### 📊 Live Statistics")
        
        # Real-time metrics
        st.metric("🔴 Active Alerts", "7")
        st.metric("📈 Events Today", "23")
        st.metric("🎯 Associations", "4")
        st.metric("⚡ Detection Rate", "1.2/hr")
        
        st.markdown("### 🎛️ Monitor Settings")
        
        # Alert settings
        alert_threshold = st.slider("Alert threshold", 0.5, 0.95, 0.8)
        notification_types = st.multiselect(
            "Event types to monitor:",
            ["Gravitational Waves", "Gamma-ray Bursts", "Neutrino Events", "Optical Transients"],
            default=["Gravitational Waves", "Gamma-ray Bursts"]
        )
        
        # Observatory status
        st.markdown("### 🔭 Observatory Status")
        observatories = {
            "LIGO Hanford": "🟢 Online",
            "LIGO Livingston": "🟢 Online", 
            "Virgo": "🟡 Maintenance",
            "IceCube": "🟢 Online",
            "Fermi-GBM": "🟢 Online"
        }
        
        for obs, status in observatories.items():
            st.markdown(f"**{obs}**: {status}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em; padding: 2rem;'>
🌌 <b>Advanced Multimessenger AI Analysis Platform</b><br>
🚀 Real-time Analysis • 🤖 Machine Learning • 📊 Advanced Visualizations • 🔬 Event Clustering<br>
Powered by cutting-edge AI and astronomical data science
</div>
""", unsafe_allow_html=True)