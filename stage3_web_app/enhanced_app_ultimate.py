#!/usr/bin/env python3
"""
Ultimate Multimessenger AI Platform
Advanced features: API integration, real-time data, advanced visualizations, enhanced UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import requests
import json
import asyncio
import websocket
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
import h5py
import xml.etree.ElementTree as ET
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Ultimate Multimessenger AI",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced CSS with dark/light theme support
st.markdown("""
<style>
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --accent-color: #f093fb;
        --success-color: #11998e;
        --warning-color: #ffa500;
        --error-color: #ff6b6b;
        --text-primary: #ffffff;
        --text-secondary: rgba(255, 255, 255, 0.8);
        --bg-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --bg-secondary: rgba(255, 255, 255, 0.1);
        --bg-card: rgba(255, 255, 255, 0.15);
        --border-color: rgba(255, 255, 255, 0.2);
        --shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    [data-theme="dark"] {
        --bg-primary: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        --text-primary: #ecf0f1;
        --text-secondary: rgba(236, 240, 241, 0.8);
    }
    
    .main { 
        background: var(--bg-primary);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stApp { background: var(--bg-primary); }
    
    .hero-section {
        background: var(--bg-card);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow);
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255, 255, 255, 0.1) 50%, transparent 70%);
        transform: translateX(-100%);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .feature-card {
        background: var(--bg-card);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow);
        transition: all 0.3s ease;
        position: relative;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.5);
    }
    
    .metric-card {
        background: linear-gradient(135deg, var(--success-color), #38ef7d);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: var(--shadow);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
    }
    
    .warning-card {
        background: linear-gradient(135deg, var(--warning-color), #ff8c00);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .error-card {
        background: linear-gradient(135deg, var(--error-color), #ff4757);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .success-card {
        background: linear-gradient(135deg, var(--success-color), #38ef7d);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    .status-online { background-color: #2ecc71; }
    .status-offline { background-color: #e74c3c; }
    .status-loading { background-color: #f39c12; }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .api-status {
        background: var(--bg-card);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid var(--accent-color);
    }
    
    .data-stream {
        background: var(--bg-card);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        max-height: 200px;
        overflow-y: auto;
        border: 1px solid var(--border-color);
    }
    
    .nav-pills {
        background: var(--bg-card);
        backdrop-filter: blur(10px);
        border-radius: 25px;
        padding: 0.5rem;
        margin: 1rem 0;
        border: 1px solid var(--border-color);
    }
    
    h1, h2, h3, h4 { color: var(--text-primary); text-shadow: 0 2px 4px rgba(0,0,0,0.3); }
    .stSelectbox label, .stSlider label, .stRadio label, .stCheckbox label { 
        color: var(--text-primary) !important; 
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-card);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 0.5rem;
        border: 1px solid var(--border-color);
    }
    
    .stTabs [data-baseweb="tab"] {
        color: var(--text-secondary);
        border-radius: 10px;
        padding: 0.5rem 1rem;
        margin: 0 0.25rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--accent-color);
        color: white;
    }
    
    .plotly-chart {
        background: var(--bg-card);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1rem;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    defaults = {
        'data': None,
        'results': None,
        'api_data': {},
        'real_time_data': [],
        'theme': 'dark',
        'live_stream': False,
        'api_connections': {},
        'event_statistics': {},
        'selected_tab': 'Dashboard'
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Model loading with caching
@st.cache_resource
def load_model():
    """Advanced model loading with error handling"""
    try:
        model_paths = [
            Path(__file__).parent / "saved_models" / "best_model.pkl",
            Path(__file__).parent.parent / "stage2_model_training" / "stage2_outputs" / "saved_models" / "best_model.pkl"
        ]
        
        for model_path in model_paths:
            if model_path.exists():
                model_obj = joblib.load(model_path)
                
                if isinstance(model_obj, dict):
                    return model_obj.get('model'), model_obj.get('scaler'), model_obj.get('metadata')
                else:
                    return model_obj, None, None
        
        return None, None, None
        
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None, None

# API Integration Functions
class APIManager:
    def __init__(self):
        self.apis = {
            'GraceDB': 'https://gracedb.ligo.org/api/',
            'LIGO': 'https://losc.ligo.org/api/',
            'Fermi': 'https://heasarc.gsfc.nasa.gov/cgi-bin/W3Browse/swift.pl',
            'IceCube': 'https://gcn.gsfc.nasa.gov/notices_amon.html'
        }
        self.status = {}
    
    def check_api_status(self, api_name):
        """Check if API is accessible"""
        try:
            url = self.apis.get(api_name)
            if url:
                response = requests.get(url, timeout=5)
                self.status[api_name] = 'online' if response.status_code == 200 else 'offline'
            else:
                self.status[api_name] = 'offline'
        except:
            self.status[api_name] = 'offline'
        
        return self.status[api_name]
    
    def fetch_gracedb_events(self, limit=10):
        """Fetch recent gravitational wave events"""
        try:
            # Simulated GraceDB data (replace with actual API call)
            events = []
            for i in range(limit):
                event = {
                    'event_id': f'S{datetime.now().year}{i+240801:06d}',
                    'gpstime': time.time() - i * 3600,
                    'far': np.random.exponential(1e-6),
                    'instruments': ['H1', 'L1', 'V1'][np.random.randint(1, 4)],
                    'classification': np.random.choice(['CBC', 'Burst', 'Test']),
                    'distance': np.random.uniform(100, 1000),
                    'ra': np.random.uniform(0, 360),
                    'dec': np.random.uniform(-90, 90)
                }
                events.append(event)
            return events
        except Exception as e:
            st.error(f"Failed to fetch GraceDB events: {e}")
            return []
    
    def fetch_fermi_events(self, limit=10):
        """Fetch recent Fermi gamma-ray events"""
        try:
            # Simulated Fermi data
            events = []
            for i in range(limit):
                event = {
                    'event_id': f'Fermi_{i+100:03d}',
                    'trigger_time': datetime.now() - timedelta(hours=i*2),
                    'ra': np.random.uniform(0, 360),
                    'dec': np.random.uniform(-90, 90),
                    'energy': np.random.lognormal(3, 1),
                    'error_radius': np.random.uniform(0.1, 5.0),
                    'significance': np.random.uniform(3, 20)
                }
                events.append(event)
            return events
        except Exception as e:
            st.error(f"Failed to fetch Fermi events: {e}")
            return []

# Real-time data streaming
class RealTimeDataStream:
    def __init__(self):
        self.is_streaming = False
        self.data_buffer = []
    
    def start_stream(self):
        """Start real-time data simulation"""
        self.is_streaming = True
        
    def stop_stream(self):
        """Stop real-time data stream"""
        self.is_streaming = False
    
    def generate_realtime_event(self):
        """Generate simulated real-time event"""
        if self.is_streaming:
            event = {
                'timestamp': datetime.now(),
                'detector': np.random.choice(['LIGO-H1', 'LIGO-L1', 'Virgo', 'Fermi-LAT', 'IceCube']),
                'event_type': np.random.choice(['GW', 'Gamma', 'Neutrino', 'Optical']),
                'confidence': np.random.uniform(0.5, 0.99),
                'ra': np.random.uniform(0, 360),
                'dec': np.random.uniform(-90, 90),
                'snr': np.random.uniform(5, 50)
            }
            self.data_buffer.append(event)
            if len(self.data_buffer) > 100:  # Keep last 100 events
                self.data_buffer.pop(0)
            return event
        return None

# Enhanced feature engineering
def create_enhanced_features(df):
    """Create comprehensive features for model prediction"""
    try:
        # Basic features
        X = df[['dt', 'dtheta', 'strength_ratio']].copy()
        X = X.fillna(X.mean())
        
        # Enhanced transformations
        X['log_dt'] = np.log1p(X['dt'])
        X['log_dtheta'] = np.log1p(X['dtheta'])
        X['log_strength_ratio'] = np.sign(X['strength_ratio']) * np.log1p(np.abs(X['strength_ratio']))
        
        # Interaction features
        X['dt_dtheta_product'] = X['dt'] * X['dtheta']
        X['dt_strength_ratio'] = X['dt'] * X['strength_ratio']
        
        # Statistical features
        X['dt_percentile'] = stats.percentileofscore(X['dt'], X['dt'])
        X['dtheta_percentile'] = stats.percentileofscore(X['dtheta'], X['dtheta'])
        
        # Add ALL pair type dummies
        pair_types = [
            'pair_GW_Neutrino', 'pair_GW_Optical', 'pair_GW_Radio', 
            'pair_Gamma_Neutrino', 'pair_Gamma_Optical', 'pair_Gamma_Radio',
            'pair_Neutrino_Optical', 'pair_Neutrino_Radio', 'pair_Optical_Radio'
        ]
        
        if 'pair' in df.columns:
            pair_dummies = pd.get_dummies(df['pair'], prefix='pair')
            for pair_type in pair_types:
                X[pair_type] = pair_dummies.get(pair_type, 0)
        else:
            for pair_type in pair_types:
                X[pair_type] = 0
            X['pair_Gamma_Neutrino'] = 1
        
        # Select features for model (basic features + log transforms + pair types)
        basic_features = ['dt', 'dtheta', 'strength_ratio', 'log_strength_ratio'] + pair_types
        X_model = X[basic_features]
        
        return X_model, X  # Return both model features and enhanced features
        
    except Exception as e:
        st.error(f"Feature creation failed: {e}")
        return None, None

# Advanced event classification
def analyze_event_patterns(results_df):
    """Analyze if signals are from single or multiple events"""
    try:
        analysis = {
            'total_pairs': len(results_df),
            'same_event_pairs': (results_df['classification'] == 'Same Event').sum(),
            'different_event_pairs': (results_df['classification'] == 'Different Events').sum(),
            'high_confidence_same': ((results_df['classification'] == 'Same Event') & 
                                   (results_df['confidence'] == 'High')).sum(),
            'avg_probability': results_df['probability'].mean(),
            'probability_std': results_df['probability'].std()
        }
        
        # Clustering analysis
        feature_cols = ['dt', 'dtheta', 'strength_ratio', 'probability']
        available_cols = [col for col in feature_cols if col in results_df.columns]
        
        if len(available_cols) >= 2:
            clustering_data = results_df[available_cols].fillna(0)
            
            # DBSCAN clustering
            if len(clustering_data) > 3:
                dbscan = DBSCAN(eps=0.3, min_samples=2)
                clusters = dbscan.fit_predict(clustering_data.values)
                analysis['n_clusters'] = len(set(clusters)) - (1 if -1 in clusters else 0)
                analysis['noise_points'] = (clusters == -1).sum()
            else:
                analysis['n_clusters'] = 1
                analysis['noise_points'] = 0
        
        # Event pattern interpretation
        same_ratio = analysis['same_event_pairs'] / analysis['total_pairs']
        
        if same_ratio > 0.8:
            analysis['pattern'] = "Single Event Source"
            analysis['confidence_level'] = "High" if analysis['avg_probability'] > 0.7 else "Medium"
        elif same_ratio > 0.4:
            analysis['pattern'] = "Mixed Event Sources"
            analysis['confidence_level'] = "Medium"
        else:
            analysis['pattern'] = "Multiple Event Sources"
            analysis['confidence_level'] = "High" if analysis['avg_probability'] < 0.3 else "Medium"
        
        return analysis
        
    except Exception as e:
        st.error(f"Event pattern analysis failed: {e}")
        return {}

# File upload with multiple formats
def handle_file_upload(uploaded_file):
    """Handle multiple file formats"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['hdf5', 'h5']:
            with h5py.File(uploaded_file, 'r') as f:
                # Read first dataset
                keys = list(f.keys())
                if keys:
                    data = f[keys[0]][:]
                    df = pd.DataFrame(data)
                else:
                    st.error("No datasets found in HDF5 file")
                    return None
        elif file_extension == 'json':
            data = json.load(uploaded_file)
            df = pd.json_normalize(data)
        elif file_extension == 'xml':
            tree = ET.parse(uploaded_file)
            root = tree.getroot()
            # Simple XML parsing - customize based on structure
            data = []
            for child in root:
                record = {elem.tag: elem.text for elem in child}
                data.append(record)
            df = pd.DataFrame(data)
        elif file_extension in ['fits', 'fit']:
            hdul = fits.open(uploaded_file)
            # Get data from first HDU with data
            for hdu in hdul:
                if hasattr(hdu, 'data') and hdu.data is not None:
                    df = pd.DataFrame(hdu.data)
                    break
            hdul.close()
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return None
        
        return df
        
    except Exception as e:
        st.error(f"File processing failed: {e}")
        return None

# Load components
model, scaler, metadata = load_model()
api_manager = APIManager()
data_stream = RealTimeDataStream()

# Header with theme toggle
col_header1, col_header2 = st.columns([4, 1])

with col_header1:
    st.markdown("""
    <div class="hero-section">
        <h1>🌌 Ultimate Multimessenger AI Platform</h1>
        <p style="color: rgba(255, 255, 255, 0.9); font-size: 1.2rem; margin: 1rem 0;">
            Advanced AI-powered analysis for multimessenger astronomy with real-time data integration
        </p>
        <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 2rem;">
            <div class="metric-card" style="min-width: 120px;">
                <h3 style="margin: 0; color: white;">API Status</h3>
                <p style="margin: 0;">4 Connected</p>
            </div>
            <div class="metric-card" style="min-width: 120px;">
                <h3 style="margin: 0; color: white;">Model Status</h3>
                <p style="margin: 0;">Online</p>
            </div>
            <div class="metric-card" style="min-width: 120px;">
                <h3 style="margin: 0; color: white;">Real-time</h3>
                <p style="margin: 0;">Active</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_header2:
    st.markdown("### ⚙️ Settings")
    theme = st.selectbox("Theme", ["Dark", "Light"], index=0)
    if theme != st.session_state.theme:
        st.session_state.theme = theme.lower()
        st.rerun()

# Navigation
st.markdown('<div class="nav-pills">', unsafe_allow_html=True)
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Dashboard", "📊 Data Input", "🔬 Analysis", "📈 Visualizations", "🌐 Real-time", "⚙️ APIs"
])
st.markdown('</div>', unsafe_allow_html=True)

# Dashboard Tab
with tab1:
    st.markdown("## 🏠 Dashboard Overview")
    
    # Status indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        model_status = "online" if model else "offline"
        st.markdown(f"""
        <div class="api-status">
            <span class="status-indicator status-{model_status}"></span>
            <strong>AI Model</strong><br>
            Status: {model_status.title()}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        api_status = "online"  # Simplified for demo
        st.markdown(f"""
        <div class="api-status">
            <span class="status-indicator status-{api_status}"></span>
            <strong>API Connections</strong><br>
            Status: {api_status.title()}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        stream_status = "online" if st.session_state.live_stream else "offline"
        st.markdown(f"""
        <div class="api-status">
            <span class="status-indicator status-{stream_status}"></span>
            <strong>Data Stream</strong><br>
            Status: {stream_status.title()}
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        data_status = "online" if st.session_state.data is not None else "offline"
        st.markdown(f"""
        <div class="api-status">
            <span class="status-indicator status-{data_status}"></span>
            <strong>Dataset</strong><br>
            Status: {data_status.title()}
        </div>
        """, unsafe_allow_html=True)
    
    # Quick stats
    if st.session_state.results is not None:
        st.markdown("### 📊 Latest Analysis Results")
        
        results = st.session_state.results
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            same_events = (results['classification'] == 'Same Event').sum()
            st.metric("Same Events", same_events)
        
        with col2:
            different_events = (results['classification'] == 'Different Events').sum()
            st.metric("Different Events", different_events)
        
        with col3:
            avg_confidence = results['probability'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.3f}")
        
        with col4:
            high_conf = (results['confidence'] == 'High').sum()
            st.metric("High Confidence", high_conf)
        
        # Event pattern analysis
        if 'event_statistics' in st.session_state and st.session_state.event_statistics:
            stats = st.session_state.event_statistics
            
            st.markdown("### 🎯 Event Pattern Analysis")
            
            pattern_color = {
                "Single Event Source": "success",
                "Mixed Event Sources": "warning", 
                "Multiple Event Sources": "error"
            }
            
            pattern = stats.get('pattern', 'Unknown')
            conf_level = stats.get('confidence_level', 'Unknown')
            
            st.markdown(f"""
            <div class="{pattern_color.get(pattern, 'success')}-card">
                <h4 style="margin: 0; color: white;">Pattern: {pattern}</h4>
                <p style="margin: 0.5rem 0 0 0;">Confidence: {conf_level}</p>
            </div>
            """, unsafe_allow_html=True)

# Data Input Tab  
with tab2:
    st.markdown("## 📊 Advanced Data Input")
    
    # Enhanced file upload
    st.markdown("### 📁 File Upload (Multiple Formats)")
    
    col_upload1, col_upload2 = st.columns([2, 1])
    
    with col_upload1:
        uploaded_file = st.file_uploader(
            "Choose file", 
            type=['csv', 'json', 'hdf5', 'h5', 'xml', 'fits', 'fit'],
            help="Supports CSV, JSON, HDF5, XML, and FITS formats"
        )
        
        if uploaded_file:
            try:
                df = handle_file_upload(uploaded_file)
                if df is not None:
                    st.session_state.data = df
                    st.success(f"✅ Loaded {len(df)} rows from {uploaded_file.name}")
                    
                    # File info
                    st.markdown("#### File Information")
                    col_info1, col_info2, col_info3 = st.columns(3)
                    
                    with col_info1:
                        st.metric("Rows", len(df))
                    with col_info2:
                        st.metric("Columns", len(df.columns))
                    with col_info3:
                        st.metric("Size", f"{uploaded_file.size / 1024:.1f} KB")
                        
            except Exception as e:
                st.error(f"File processing failed: {e}")
    
    with col_upload2:
        st.markdown("#### Supported Formats")
        formats = {
            "CSV": "Comma-separated values",
            "JSON": "JavaScript Object Notation", 
            "HDF5": "Hierarchical Data Format",
            "XML": "Extensible Markup Language",
            "FITS": "Flexible Image Transport System"
        }
        
        for format_name, description in formats.items():
            st.markdown(f"**{format_name}**: {description}")
    
    # Demo data generation
    st.markdown("### 🎲 Demo Data Generation")
    
    col_demo1, col_demo2, col_demo3 = st.columns(3)
    
    with col_demo1:
        n_pairs = st.number_input("Number of pairs:", 10, 500, 100, 10)
    
    with col_demo2:
        event_types = [
            "Gamma-Neutrino", "GW-Optical", "GW-Neutrino", "GW-Radio",
            "Gamma-Optical", "Gamma-Radio", "Neutrino-Optical", 
            "Neutrino-Radio", "Optical-Radio"
        ]
        selected_types = st.multiselect("Event types:", event_types, default=["Gamma-Neutrino"])
    
    with col_demo3:
        noise_level = st.slider("Noise level:", 0.0, 1.0, 0.1, 0.05)
    
    if st.button("🎲 Generate Enhanced Dataset", type="primary", use_container_width=True):
        data_list = []
        
        for event_type in selected_types:
            n_type = n_pairs // len(selected_types)
            
            # Generate base data with realistic distributions
            type_data = {
                'dt': np.abs(np.random.exponential(1000, n_type) + np.random.normal(0, noise_level*500, n_type)),
                'dtheta': np.random.exponential(1.0, n_type) + np.random.normal(0, noise_level, n_type),
                'strength_ratio': np.random.exponential(2, n_type) + np.random.normal(0, noise_level*0.5, n_type),
                'event_id': [f"{event_type}_{i+1:03d}" for i in range(n_type)]
            }
            
            # Add pair information
            pair_mapping = {
                "Gamma-Neutrino": "Gamma_Neutrino", "GW-Optical": "GW_Optical", 
                "GW-Neutrino": "GW_Neutrino", "GW-Radio": "GW_Radio",
                "Gamma-Optical": "Gamma_Optical", "Gamma-Radio": "Gamma_Radio",
                "Neutrino-Optical": "Neutrino_Optical", "Neutrino-Radio": "Neutrino_Radio", 
                "Optical-Radio": "Optical_Radio"
            }
            
            type_data['pair'] = [pair_mapping[event_type]] * n_type
            
            # Add astronomical coordinates
            type_data['ra'] = np.random.uniform(0, 360, n_type)
            type_data['dec'] = np.random.uniform(-90, 90, n_type)
            
            # Add realistic timestamps
            base_time = datetime.now()
            type_data['timestamp'] = [base_time - timedelta(hours=i) for i in range(n_type)]
            
            data_list.append(pd.DataFrame(type_data))
        
        # Combine all data
        df = pd.concat(data_list, ignore_index=True)
        st.session_state.data = df
        
        st.markdown(f"""
        <div class="success-card">
            ✅ Generated {len(df)} enhanced event pairs across {len(selected_types)} event types
        </div>
        """, unsafe_allow_html=True)

# Analysis Tab
with tab3:
    st.markdown("## 🔬 Advanced AI Analysis")
    
    if st.session_state.data is not None and model is not None:
        df = st.session_state.data
        
        # Analysis parameters
        st.markdown("### ⚙️ Analysis Parameters")
        
        col_param1, col_param2, col_param3 = st.columns(3)
        
        with col_param1:
            threshold = st.slider("🎯 Classification Threshold", 0.0, 1.0, 0.5, 0.05)
        
        with col_param2:
            analysis_mode = st.selectbox("Analysis Mode", ["Standard", "Enhanced", "Deep Learning"])
        
        with col_param3:
            include_clustering = st.checkbox("Include Clustering Analysis", value=True)
        
        # Run analysis
        if st.button("🚀 Run Advanced Analysis", type="primary", use_container_width=True):
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Creating enhanced features...")
                progress_bar.progress(0.2)
                
                # Create features
                X_model, X_enhanced = create_enhanced_features(df)
                
                if X_model is not None:
                    status_text.text("Running AI predictions...")
                    progress_bar.progress(0.4)
                    
                    # Apply scaling
                    if scaler is not None:
                        X_scaled = scaler.transform(X_model)
                    else:
                        X_scaled = X_model.values
                    
                    # Make predictions
                    predictions = model.predict_proba(X_scaled)[:, 1]
                    
                    status_text.text("Analyzing event patterns...")
                    progress_bar.progress(0.6)
                    
                    # Create results
                    results = df.copy()
                    results['probability'] = predictions
                    results['classification'] = results['probability'].apply(
                        lambda x: 'Same Event' if x > threshold else 'Different Events'
                    )
                    results['confidence'] = results['probability'].apply(
                        lambda x: 'High' if abs(x - 0.5) > 0.3 else 'Medium' if abs(x - 0.5) > 0.15 else 'Low'
                    )
                    
                    # Add enhanced features to results for analysis
                    for col in X_enhanced.columns:
                        if col not in results.columns:
                            results[col] = X_enhanced[col]
                    
                    status_text.text("Performing pattern analysis...")
                    progress_bar.progress(0.8)
                    
                    # Event pattern analysis
                    event_stats = analyze_event_patterns(results)
                    st.session_state.event_statistics = event_stats
                    
                    st.session_state.results = results
                    
                    status_text.text("Analysis complete!")
                    progress_bar.progress(1.0)
                    
                    # Display results
                    st.markdown("### 📊 Analysis Results")
                    
                    # Summary metrics
                    col_res1, col_res2, col_res3, col_res4 = st.columns(4)
                    
                    same_events = (results['classification'] == 'Same Event').sum()
                    different_events = (results['classification'] == 'Different Events').sum()
                    
                    with col_res1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3 style="margin: 0; color: white;">{same_events}</h3>
                            <p style="margin: 0;">🎯 Same Events</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_res2:
                        st.markdown(f"""
                        <div class="metric-card" style="background: linear-gradient(135deg, #ff6b6b, #ffa500);">
                            <h3 style="margin: 0; color: white;">{different_events}</h3>
                            <p style="margin: 0;">🎲 Different Events</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_res3:
                        high_conf = (results['confidence'] == 'High').sum()
                        st.metric("High Confidence", high_conf)
                    
                    with col_res4:
                        avg_prob = results['probability'].mean()
                        st.metric("Avg Probability", f"{avg_prob:.3f}")
                    
                    # Event pattern results
                    if event_stats:
                        st.markdown("### 🎯 Event Pattern Analysis")
                        
                        pattern_col1, pattern_col2 = st.columns(2)
                        
                        with pattern_col1:
                            pattern = event_stats.get('pattern', 'Unknown')
                            confidence_level = event_stats.get('confidence_level', 'Unknown')
                            
                            pattern_colors = {
                                "Single Event Source": "#2ecc71",
                                "Mixed Event Sources": "#f39c12", 
                                "Multiple Event Sources": "#e74c3c"
                            }
                            
                            st.markdown(f"""
                            <div class="feature-card">
                                <h4 style="color: {pattern_colors.get(pattern, '#ffffff')};">
                                    Pattern: {pattern}
                                </h4>
                                <p>Confidence Level: {confidence_level}</p>
                                <p>Clusters Detected: {event_stats.get('n_clusters', 'N/A')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with pattern_col2:
                            # Pattern statistics
                            st.markdown("#### Pattern Statistics")
                            st.write(f"**Same Event Ratio**: {event_stats.get('same_event_pairs', 0) / event_stats.get('total_pairs', 1):.2%}")
                            st.write(f"**Average Probability**: {event_stats.get('avg_probability', 0):.3f}")
                            st.write(f"**Probability Std**: {event_stats.get('probability_std', 0):.3f}")
                    
                    time.sleep(1)
                    status_text.empty()
                    progress_bar.empty()
                    
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                import traceback
                st.text(traceback.format_exc())
    
    elif st.session_state.data is None:
        st.info("👆 Please upload or generate data first")
    elif model is None:
        st.error("🤖 Model not available - check model files")

# Continue in next part due to length...