#!/usr/bin/env python3
"""
Enhanced Multimessenger AI Platform v2.0
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
import time
from datetime import datetime, timedelta
from pathlib import Path
import xml.etree.ElementTree as ET
from scipy import stats
from sklearn.cluster import DBSCAN
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Enhanced Multimessenger AI",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with modern design
st.markdown("""
<style>
    .main { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    
    .hero-section {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
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
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.5);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
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
    
    .api-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #f093fb;
    }
    
    .success-card {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #ffa500, #ff8c00);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .error-card {
        background: linear-gradient(135deg, #ff6b6b, #ff4757);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    h1, h2, h3, h4 { 
        color: white; 
        text-shadow: 0 2px 4px rgba(0,0,0,0.3); 
    }
    
    .stSelectbox label, .stSlider label, .stRadio label, .stCheckbox label { 
        color: white !important; 
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        padding: 0.5rem 1rem;
        margin: 0 0.25rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: #f093fb;
        color: white;
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
        'live_stream': False,
        'event_statistics': {}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Model loading
@st.cache_resource
def load_model():
    """Enhanced model loading"""
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

# API Manager
class APIManager:
    def __init__(self):
        self.apis = {
            'GraceDB': 'https://gracedb.ligo.org',
            'LIGO': 'https://losc.ligo.org',
            'Fermi': 'https://fermi.gsfc.nasa.gov',
            'IceCube': 'https://icecube.wisc.edu'
        }
        self.status = {}
    
    def check_api_status(self, api_name):
        """Check API accessibility"""
        try:
            url = self.apis.get(api_name)
            if url:
                response = requests.get(url, timeout=3)
                self.status[api_name] = 'online' if response.status_code == 200 else 'offline'
            else:
                self.status[api_name] = 'offline'
        except:
            self.status[api_name] = 'offline'
        
        return self.status[api_name]
    
    def generate_mock_gw_events(self, limit=10):
        """Generate mock gravitational wave events"""
        events = []
        for i in range(limit):
            event = {
                'event_id': f'S{datetime.now().year}{i+240801:06d}',
                'gpstime': time.time() - i * 3600,
                'far': np.random.exponential(1e-6),
                'instruments': np.random.choice([['H1', 'L1'], ['H1', 'L1', 'V1'], ['L1', 'V1']]),
                'classification': np.random.choice(['CBC', 'Burst', 'Test']),
                'distance': np.random.uniform(100, 1000),
                'ra': np.random.uniform(0, 360),
                'dec': np.random.uniform(-90, 90),
                'snr': np.random.uniform(8, 25)
            }
            events.append(event)
        return events
    
    def generate_mock_gamma_events(self, limit=10):
        """Generate mock gamma-ray events"""
        events = []
        for i in range(limit):
            event = {
                'event_id': f'Fermi_{i+100:03d}',
                'trigger_time': datetime.now() - timedelta(hours=i*2),
                'ra': np.random.uniform(0, 360),
                'dec': np.random.uniform(-90, 90),
                'energy': np.random.lognormal(3, 1),
                'error_radius': np.random.uniform(0.1, 5.0),
                'significance': np.random.uniform(3, 20),
                'fluence': np.random.exponential(1e-6)
            }
            events.append(event)
        return events

# Real-time data stream
class RealTimeStream:
    def __init__(self):
        self.is_active = False
        self.buffer = []
    
    def start(self):
        self.is_active = True
    
    def stop(self):
        self.is_active = False
    
    def generate_event(self):
        """Generate simulated real-time event"""
        if self.is_active:
            event = {
                'timestamp': datetime.now(),
                'detector': np.random.choice(['LIGO-H1', 'LIGO-L1', 'Virgo', 'Fermi-LAT', 'IceCube', 'HAWC']),
                'event_type': np.random.choice(['GW', 'Gamma', 'Neutrino', 'Optical', 'Radio']),
                'confidence': np.random.uniform(0.5, 0.99),
                'ra': np.random.uniform(0, 360),
                'dec': np.random.uniform(-90, 90),
                'snr': np.random.uniform(5, 50),
                'energy': np.random.lognormal(2, 1) if np.random.choice(['Gamma', 'Neutrino']) else None
            }
            self.buffer.append(event)
            if len(self.buffer) > 50:  # Keep last 50 events
                self.buffer.pop(0)
            return event
        return None

# Enhanced feature engineering
def create_enhanced_features(df):
    """Create comprehensive features"""
    try:
        # Basic features
        X = df[['dt', 'dtheta', 'strength_ratio']].copy()
        X = X.fillna(X.mean())
        
        # Enhanced transformations
        X['log_dt'] = np.log1p(X['dt'])
        X['log_dtheta'] = np.log1p(X['dtheta'])
        X['log_strength_ratio'] = np.sign(X['strength_ratio']) * np.log1p(np.abs(X['strength_ratio']))
        
        # Statistical features
        X['dt_percentile'] = [stats.percentileofscore(X['dt'].values, val) for val in X['dt']]
        X['dtheta_percentile'] = [stats.percentileofscore(X['dtheta'].values, val) for val in X['dtheta']]
        
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
        
        # Select model features
        model_features = ['dt', 'dtheta', 'strength_ratio', 'log_strength_ratio'] + pair_types
        X_model = X[model_features]
        
        return X_model, X
        
    except Exception as e:
        st.error(f"Feature creation failed: {e}")
        return None, None

# Advanced event analysis
def analyze_event_patterns(results_df):
    """Comprehensive event pattern analysis"""
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
        if len(results_df) > 3:
            feature_cols = ['dt', 'dtheta', 'strength_ratio', 'probability']
            available_cols = [col for col in feature_cols if col in results_df.columns]
            
            if len(available_cols) >= 2:
                clustering_data = results_df[available_cols].fillna(0)
                dbscan = DBSCAN(eps=0.3, min_samples=2)
                clusters = dbscan.fit_predict(clustering_data.values)
                analysis['n_clusters'] = len(set(clusters)) - (1 if -1 in clusters else 0)
                analysis['noise_points'] = (clusters == -1).sum()
            else:
                analysis['n_clusters'] = 1
                analysis['noise_points'] = 0
        
        # Pattern interpretation
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
        st.error(f"Pattern analysis failed: {e}")
        return {}

# File upload handler
def handle_file_upload(uploaded_file):
    """Handle multiple file formats"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension == 'json':
            data = json.load(uploaded_file)
            df = pd.json_normalize(data)
        elif file_extension == 'xml':
            # Simplified XML handling
            st.warning("XML files: Basic support. Use CSV for full compatibility.")
            return None
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
data_stream = RealTimeStream()

# Header
st.markdown("""
<div class="hero-section">
    <h1>🌌 Enhanced Multimessenger AI Platform</h1>
    <p style="color: rgba(255, 255, 255, 0.9); font-size: 1.2rem; margin: 1rem 0;">
        Next-Generation AI Analysis • Real-time Data • API Integration • Advanced Visualizations
    </p>
    <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 2rem;">
        <div class="metric-card" style="min-width: 120px;">
            <h3 style="margin: 0; color: white;">AI Model</h3>
            <p style="margin: 0;">{'🟢 Online' if model else '🔴 Offline'}</p>
        </div>
        <div class="metric-card" style="min-width: 120px;">
            <h3 style="margin: 0; color: white;">APIs</h3>
            <p style="margin: 0;">🟢 Ready</p>
        </div>
        <div class="metric-card" style="min-width: 120px;">
            <h3 style="margin: 0; color: white;">Real-time</h3>
            <p style="margin: 0;">{'🟢 Live' if st.session_state.live_stream else '🔴 Idle'}</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Dashboard", "📊 Data Input", "🔬 AI Analysis", "📈 Visualizations", "🌐 Real-time", "⚙️ APIs"
])

# Dashboard Tab
with tab1:
    st.markdown("## 🏠 System Dashboard")
    
    # Quick status overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        model_status = "🟢 Online" if model else "🔴 Offline"
        st.markdown(f"""
        <div class="api-card">
            <span class="status-indicator status-{'online' if model else 'offline'}"></span>
            <strong>AI Model</strong><br>
            Status: {model_status}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        data_status = "🟢 Loaded" if st.session_state.data is not None else "🔴 No Data"
        st.markdown(f"""
        <div class="api-card">
            <span class="status-indicator status-{'online' if st.session_state.data is not None else 'offline'}"></span>
            <strong>Dataset</strong><br>
            Status: {data_status}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        results_status = "🟢 Available" if st.session_state.results is not None else "🔴 Not Run"
        st.markdown(f"""
        <div class="api-card">
            <span class="status-indicator status-{'online' if st.session_state.results is not None else 'offline'}"></span>
            <strong>Analysis</strong><br>
            Status: {results_status}
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        stream_status = "🟢 Active" if st.session_state.live_stream else "🔴 Inactive"
        st.markdown(f"""
        <div class="api-card">
            <span class="status-indicator status-{'online' if st.session_state.live_stream else 'offline'}"></span>
            <strong>Live Stream</strong><br>
            Status: {stream_status}
        </div>
        """, unsafe_allow_html=True)
    
    # Recent analysis results
    if st.session_state.results is not None:
        st.markdown("### 📊 Latest Analysis Summary")
        
        results = st.session_state.results
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            same_events = (results['classification'] == 'Same Event').sum()
            st.metric("Same Events", same_events, delta=f"{same_events/len(results)*100:.1f}%")
        
        with col2:
            different_events = (results['classification'] == 'Different Events').sum()
            st.metric("Different Events", different_events, delta=f"{different_events/len(results)*100:.1f}%")
        
        with col3:
            avg_prob = results['probability'].mean()
            st.metric("Avg Probability", f"{avg_prob:.3f}")
        
        with col4:
            high_conf = (results['confidence'] == 'High').sum()
            st.metric("High Confidence", high_conf, delta=f"{high_conf/len(results)*100:.1f}%")
        
        # Pattern analysis summary
        if st.session_state.event_statistics:
            stats = st.session_state.event_statistics
            pattern = stats.get('pattern', 'Unknown')
            
            pattern_colors = {
                "Single Event Source": "success",
                "Mixed Event Sources": "warning", 
                "Multiple Event Sources": "error"
            }
            
            st.markdown(f"""
            <div class="{pattern_colors.get(pattern, 'success')}-card">
                <h4 style="margin: 0; color: white;">🎯 Event Pattern: {pattern}</h4>
                <p style="margin: 0.5rem 0 0 0;">
                    Confidence: {stats.get('confidence_level', 'Unknown')} | 
                    Clusters: {stats.get('n_clusters', 'N/A')}
                </p>
            </div>
            """, unsafe_allow_html=True)

# Data Input Tab
with tab2:
    st.markdown("## 📊 Enhanced Data Input")
    
    # File upload section
    st.markdown("### 📁 File Upload (Multiple Formats)")
    
    col_upload1, col_upload2 = st.columns([3, 1])
    
    with col_upload1:
        uploaded_file = st.file_uploader(
            "Upload your data file", 
            type=['csv', 'json', 'txt'],
            help="Supports CSV, JSON, and TXT formats"
        )
        
        if uploaded_file:
            df = handle_file_upload(uploaded_file)
            if df is not None:
                st.session_state.data = df
                st.markdown(f"""
                <div class="success-card">
                    ✅ Successfully loaded {len(df)} rows from {uploaded_file.name}
                </div>
                """, unsafe_allow_html=True)
                
                # File preview
                st.markdown("#### Data Preview")
                st.dataframe(df.head(), use_container_width=True)
    
    with col_upload2:
        st.markdown("#### Supported Formats")
        st.markdown("**CSV**: Comma-separated values")
        st.markdown("**JSON**: JavaScript Object Notation")
        st.markdown("**TXT**: Plain text (comma-separated)")
    
    # Enhanced demo data generation
    st.markdown("### 🎲 Enhanced Demo Data Generation")
    
    col_demo1, col_demo2, col_demo3 = st.columns(3)
    
    with col_demo1:
        n_pairs = st.number_input("Number of pairs:", 10, 500, 100, 10)
        complexity = st.selectbox("Data Complexity:", ["Simple", "Realistic", "Complex"])
    
    with col_demo2:
        event_types = [
            "Gamma-Neutrino", "GW-Optical", "GW-Neutrino", "GW-Radio",
            "Gamma-Optical", "Gamma-Radio", "Neutrino-Optical", 
            "Neutrino-Radio", "Optical-Radio"
        ]
        selected_types = st.multiselect("Event types:", event_types, default=["Gamma-Neutrino", "GW-Optical"])
    
    with col_demo3:
        noise_level = st.slider("Noise level:", 0.0, 1.0, 0.1, 0.05)
        include_coords = st.checkbox("Include sky coordinates", value=True)
    
    if st.button("🎲 Generate Enhanced Dataset", type="primary", use_container_width=True):
        with st.spinner("Generating enhanced dataset..."):
            data_list = []
            
            for event_type in selected_types:
                n_type = max(1, n_pairs // len(selected_types))
                
                # Generate realistic data based on complexity
                if complexity == "Simple":
                    dt_data = np.abs(np.random.normal(1000, 500, n_type))
                    dtheta_data = np.random.exponential(1.0, n_type)
                    strength_data = np.random.exponential(2, n_type)
                elif complexity == "Realistic":
                    dt_data = np.abs(np.random.lognormal(6, 1, n_type))
                    dtheta_data = np.random.gamma(2, 0.5, n_type)
                    strength_data = np.random.beta(2, 5, n_type) * 10
                else:  # Complex
                    dt_data = np.abs(np.random.weibull(1.5, n_type) * 2000)
                    dtheta_data = np.random.pareto(1.5, n_type)
                    strength_data = np.random.lognormal(1, 1, n_type)
                
                # Add noise
                dt_data += np.random.normal(0, noise_level * dt_data.std(), n_type)
                dtheta_data += np.random.normal(0, noise_level * dtheta_data.std(), n_type)
                strength_data += np.random.normal(0, noise_level * strength_data.std(), n_type)
                
                type_data = {
                    'dt': np.abs(dt_data),
                    'dtheta': np.abs(dtheta_data),
                    'strength_ratio': np.abs(strength_data),
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
                
                # Add coordinates if requested
                if include_coords:
                    type_data['ra'] = np.random.uniform(0, 360, n_type)
                    type_data['dec'] = np.random.uniform(-90, 90, n_type)
                
                # Add timestamps
                base_time = datetime.now()
                type_data['timestamp'] = [base_time - timedelta(hours=np.random.uniform(0, 24)) for _ in range(n_type)]
                
                data_list.append(pd.DataFrame(type_data))
            
            # Combine all data
            df = pd.concat(data_list, ignore_index=True)
            st.session_state.data = df
            
            st.markdown(f"""
            <div class="success-card">
                ✅ Generated {len(df)} enhanced event pairs with {complexity.lower()} complexity
            </div>
            """, unsafe_allow_html=True)
            
            # Show sample
            st.markdown("#### Generated Data Sample")
            st.dataframe(df.head(10), use_container_width=True)

# AI Analysis Tab
with tab3:
    st.markdown("## 🔬 Advanced AI Analysis")
    
    if st.session_state.data is not None and model is not None:
        df = st.session_state.data
        
        # Analysis parameters
        st.markdown("### ⚙️ Analysis Configuration")
        
        col_param1, col_param2, col_param3 = st.columns(3)
        
        with col_param1:
            threshold = st.slider("🎯 Classification Threshold", 0.0, 1.0, 0.5, 0.05)
            batch_size = st.number_input("Batch Size", 10, 1000, min(100, len(df)), 10)
        
        with col_param2:
            analysis_mode = st.selectbox("Analysis Mode", ["Standard", "Enhanced", "Comprehensive"])
            include_clustering = st.checkbox("Include Clustering Analysis", value=True)
        
        with col_param3:
            confidence_bands = st.checkbox("Calculate Confidence Intervals", value=True)
            feature_importance = st.checkbox("Analyze Feature Importance", value=True)
        
        # Run analysis
        if st.button("🚀 Run Advanced AI Analysis", type="primary", use_container_width=True):
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("🔧 Creating enhanced features...")
                progress_bar.progress(0.1)
                
                # Create features
                X_model, X_enhanced = create_enhanced_features(df)
                
                if X_model is not None:
                    status_text.text("🤖 Running AI predictions...")
                    progress_bar.progress(0.3)
                    
                    # Apply scaling
                    if scaler is not None:
                        X_scaled = scaler.transform(X_model)
                    else:
                        X_scaled = X_model.values
                    
                    # Make predictions
                    predictions = model.predict_proba(X_scaled)[:, 1]
                    
                    status_text.text("📊 Analyzing event patterns...")
                    progress_bar.progress(0.5)
                    
                    # Create comprehensive results
                    results = df.copy()
                    results['probability'] = predictions
                    results['classification'] = results['probability'].apply(
                        lambda x: 'Same Event' if x > threshold else 'Different Events'
                    )
                    results['confidence'] = results['probability'].apply(
                        lambda x: 'High' if abs(x - 0.5) > 0.3 else 'Medium' if abs(x - 0.5) > 0.15 else 'Low'
                    )
                    
                    # Add enhanced features to results
                    for col in X_enhanced.columns:
                        if col not in results.columns:
                            results[col] = X_enhanced[col]
                    
                    status_text.text("🎯 Performing pattern analysis...")
                    progress_bar.progress(0.7)
                    
                    # Event pattern analysis
                    event_stats = analyze_event_patterns(results)
                    st.session_state.event_statistics = event_stats
                    
                    st.session_state.results = results
                    
                    status_text.text("✅ Analysis complete!")
                    progress_bar.progress(1.0)
                    
                    # Display comprehensive results
                    st.markdown("### 📊 Comprehensive Analysis Results")
                    
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
                        st.metric("High Confidence", high_conf, delta=f"{high_conf/len(results)*100:.1f}%")
                    
                    with col_res4:
                        avg_prob = results['probability'].mean()
                        st.metric("Avg Probability", f"{avg_prob:.3f}", delta=f"σ={results['probability'].std():.3f}")
                    
                    # Enhanced pattern analysis
                    if event_stats:
                        st.markdown("### 🎯 Advanced Event Pattern Analysis")
                        
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
                                    🎯 Pattern: {pattern}
                                </h4>
                                <p><strong>Confidence Level:</strong> {confidence_level}</p>
                                <p><strong>Clusters Detected:</strong> {event_stats.get('n_clusters', 'N/A')}</p>
                                <p><strong>Noise Points:</strong> {event_stats.get('noise_points', 'N/A')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with pattern_col2:
                            st.markdown("#### 📈 Pattern Statistics")
                            same_ratio = event_stats.get('same_event_pairs', 0) / event_stats.get('total_pairs', 1)
                            st.metric("Same Event Ratio", f"{same_ratio:.2%}")
                            st.metric("Average Probability", f"{event_stats.get('avg_probability', 0):.3f}")
                            st.metric("Probability Std Dev", f"{event_stats.get('probability_std', 0):.3f}")
                    
                    # Results table
                    st.markdown("### 📋 Detailed Results")
                    display_cols = ['event_id', 'classification', 'probability', 'confidence']
                    if 'pair' in results.columns:
                        display_cols.insert(1, 'pair')
                    
                    st.dataframe(
                        results[display_cols].round(4),
                        use_container_width=True,
                        height=400
                    )
                    
                    time.sleep(1)
                    status_text.empty()
                    progress_bar.empty()
                    
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
                import traceback
                st.text(traceback.format_exc())
    
    elif st.session_state.data is None:
        st.info("👆 Please upload or generate data first")
    elif model is None:
        st.error("🤖 Model not available - check model files")

# Visualizations Tab
with tab4:
    st.markdown("## 📈 Advanced Visualizations")
    
    if st.session_state.results is not None:
        results = st.session_state.results
        
        # Visualization controls
        st.markdown("### 🎨 Visualization Controls")
        
        # Enhanced visualization controls with better styling
        st.markdown("### 🎨 Visualization Controls")
        col_viz1, col_viz2, col_viz3 = st.columns(3)
        
        with col_viz1:
            viz_type = st.selectbox("📊 Visualization Type", [
                "Probability Distribution", "Sky Map", "Correlation Matrix", 
                "3D Feature Space", "Time Series", "Classification Summary"
            ], help="Choose the type of visualization to display")
        
        with col_viz2:
            color_scheme = st.selectbox("🎨 Color Scheme", [
                "Viridis", "Plasma", "Blues", "Reds", "Greens", "Rainbow"
            ], help="Select color palette for visualizations")
        
        with col_viz3:
            interactive_mode = st.checkbox("🔄 Interactive Mode", value=True, 
                                         help="Enable interactive features in plots")
        
        # Generate visualizations
        if viz_type == "Probability Distribution":
            st.markdown("### 📊 Probability Distribution Analysis")
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Probability Histogram', 'Classification Distribution', 
                               'Confidence Levels', 'Probability vs Features'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"type": "domain"}, {"secondary_y": False}]]
            )
            
            # Probability histogram
            fig.add_trace(
                go.Histogram(
                    x=results['probability'],
                    name='Probability',
                    opacity=0.7,
                    nbinsx=30,
                    marker_color='lightblue'
                ), row=1, col=1
            )
            
            # Classification distribution
            class_counts = results['classification'].value_counts()
            fig.add_trace(
                go.Bar(
                    x=class_counts.index,
                    y=class_counts.values,
                    name='Classifications',
                    marker_color=['green', 'red']
                ), row=1, col=2
            )
            
            # Confidence levels
            conf_counts = results['confidence'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=conf_counts.index,
                    values=conf_counts.values,
                    name="Confidence"
                ), row=2, col=1
            )
            
            # Probability vs features
            if 'dt' in results.columns:
                fig.add_trace(
                    go.Scatter(
                        x=results['dt'],
                        y=results['probability'],
                        mode='markers',
                        marker=dict(
                            color=results['probability'],
                            colorscale=color_scheme.lower(),
                            size=8
                        ),
                        name='dt vs Probability'
                    ), row=2, col=2
                )
            
            fig.update_layout(
                height=800,
                showlegend=True,
                title_text="Comprehensive Probability Analysis",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Sky Map":
            st.markdown("### 🌌 Sky Map Visualization")
            
            try:
                if 'ra' in results.columns and 'dec' in results.columns:
                    fig = go.Figure()
                    
                    # Color mapping for classifications
                    color_map = {'Same Event': '#2ecc71', 'Different Events': '#e74c3c'}
                    
                    # Add events colored by classification
                    for classification in results['classification'].unique():
                        mask = results['classification'] == classification
                        subset = results[mask]
                        
                        if len(subset) > 0:
                            fig.add_trace(
                                go.Scatterpolar(
                                    r=90 - subset['dec'],  # Convert dec to polar radius
                                    theta=subset['ra'],
                                    mode='markers',
                                    marker=dict(
                                        size=subset['probability'] * 15 + 8,
                                        color=color_map.get(classification, '#3498db'),
                                        opacity=0.8,
                                        line=dict(width=2, color='white')
                                    ),
                                    name=classification,
                                    text=[f"Event: {row.get('event_id', f'Event_{i}')}<br>"
                                         f"Probability: {row['probability']:.3f}<br>"
                                         f"RA: {row['ra']:.2f}°<br>"
                                         f"Dec: {row['dec']:.2f}°<br>"
                                         f"Classification: {row['classification']}" 
                                         for i, (_, row) in enumerate(subset.iterrows())],
                                    hovertemplate="%{text}<extra></extra>"
                                )
                            )
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True, 
                                range=[0, 90],
                                title="90° - Declination"
                            ),
                            angularaxis=dict(
                                direction="clockwise", 
                                period=360,
                                title="Right Ascension (degrees)"
                            )
                        ),
                        title="🌌 Multimessenger Event Sky Map",
                        height=650,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white',
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Sky map statistics
                    st.markdown("#### 📊 Sky Distribution Statistics")
                    col_sky1, col_sky2, col_sky3 = st.columns(3)
                    
                    with col_sky1:
                        ra_spread = results['ra'].max() - results['ra'].min()
                        st.metric("RA Spread", f"{ra_spread:.1f}°")
                    
                    with col_sky2:
                        dec_spread = results['dec'].max() - results['dec'].min()
                        st.metric("Dec Spread", f"{dec_spread:.1f}°")
                    
                    with col_sky3:
                        # Angular clustering measure
                        mean_ra = results['ra'].mean()
                        mean_dec = results['dec'].mean()
                        distances = np.sqrt((results['ra'] - mean_ra)**2 + (results['dec'] - mean_dec)**2)
                        clustering = distances.std()
                        st.metric("Angular Clustering", f"{clustering:.1f}°")
                        
                else:
                    st.warning("⚠️ Sky coordinates (RA/Dec) not available in current dataset")
                    st.info("💡 Generate data with 'Include sky coordinates' option enabled")
                    
            except Exception as e:
                st.error(f"❌ Sky map visualization failed: {e}")
                st.info("💡 Ensure your dataset includes RA and Dec coordinates")
        
        elif viz_type == "Correlation Matrix":
            st.markdown("### 🔗 Feature Correlation Analysis")
            
            try:
                # Select numeric columns
                numeric_cols = results.select_dtypes(include=[np.number]).columns
                correlation_cols = [col for col in numeric_cols if col in ['dt', 'dtheta', 'strength_ratio', 'probability']]
                
                if len(correlation_cols) > 1:
                    corr_matrix = results[correlation_cols].corr()
                    
                    fig = go.Figure(
                        data=go.Heatmap(
                            z=corr_matrix.values,
                            x=corr_matrix.columns,
                            y=corr_matrix.columns,
                            colorscale=color_scheme.lower(),
                            zmin=-1,
                            zmax=1,
                            text=np.round(corr_matrix.values, 3),
                            texttemplate="%{text}",
                            textfont={"size": 12},
                            hoverongaps=False
                        )
                    )
                    
                    fig.update_layout(
                        title="Feature Correlation Matrix",
                        height=600,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show correlation insights
                    st.markdown("#### 📊 Correlation Insights")
                    col_cor1, col_cor2 = st.columns(2)
                    
                    with col_cor1:
                        # Strongest positive correlation
                        corr_flat = corr_matrix.where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1))
                        max_corr = corr_flat.stack().max()
                        max_pair = corr_flat.stack().idxmax()
                        st.metric("Strongest Positive Correlation", f"{max_corr:.3f}", delta=f"{max_pair[0]} ↔ {max_pair[1]}")
                    
                    with col_cor2:
                        # Strongest negative correlation
                        min_corr = corr_flat.stack().min()
                        min_pair = corr_flat.stack().idxmin()
                        st.metric("Strongest Negative Correlation", f"{min_corr:.3f}", delta=f"{min_pair[0]} ↔ {min_pair[1]}")
                        
                else:
                    st.warning("⚠️ Insufficient numeric features for correlation analysis")
                    st.info("💡 Run analysis first to generate more features")
                    
            except Exception as e:
                st.error(f"❌ Correlation analysis failed: {e}")
                st.info("💡 Try running the analysis first to generate proper features")
        
        elif viz_type == "3D Feature Space":
            st.markdown("### 🎯 3D Feature Space")
            
            if all(col in results.columns for col in ['dt', 'dtheta', 'strength_ratio']):
                fig = go.Figure(
                    data=go.Scatter3d(
                        x=results['dt'],
                        y=results['dtheta'],
                        z=results['strength_ratio'],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=results['probability'],
                            colorscale=color_scheme.lower(),
                            colorbar=dict(title="Probability"),
                            line=dict(width=0.5, color='white')
                        ),
                        text=[f"Event: {results.loc[i, 'event_id'] if 'event_id' in results.columns else i}<br>"
                             f"Classification: {results.loc[i, 'classification']}<br>"
                             f"Probability: {results.loc[i, 'probability']:.3f}"
                             for i in results.index],
                        hovertemplate="%{text}<extra></extra>"
                    )
                )
                
                fig.update_layout(
                    scene=dict(
                        xaxis_title='Time Difference (dt)',
                        yaxis_title='Angular Separation (dθ)',
                        zaxis_title='Strength Ratio'
                    ),
                    title="3D Feature Space",
                    height=700,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Required features not available")
        
        elif viz_type == "Time Series":
            st.markdown("### ⏱️ Time Series Analysis")
            
            try:
                if 'dt' in results.columns:
                    # Sort by time difference
                    sorted_results = results.sort_values('dt')
                    
                    fig = go.Figure()
                    
                    # Add probability over time
                    fig.add_trace(
                        go.Scatter(
                            x=sorted_results['dt'],
                            y=sorted_results['probability'],
                            mode='lines+markers',
                            name='Probability',
                            line=dict(color='lightblue', width=2),
                            marker=dict(size=8, color=sorted_results['probability'], 
                                      colorscale=color_scheme.lower(), 
                                      colorbar=dict(title="Probability"))
                        )
                    )
                    
                    # Add threshold line
                    fig.add_hline(y=threshold, line_dash="dash", 
                                 line_color="red", annotation_text="Classification Threshold")
                    
                    fig.update_layout(
                        title="Probability Evolution Over Time",
                        xaxis_title="Time Difference (dt)",
                        yaxis_title="Event Probability",
                        height=500,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ Time data not available")
                    
            except Exception as e:
                st.error(f"❌ Time series visualization failed: {e}")
        
        elif viz_type == "Classification Summary":
            st.markdown("### 📋 Classification Summary")
            
            col_summary1, col_summary2 = st.columns(2)
            
            with col_summary1:
                # Classification pie chart
                class_counts = results['classification'].value_counts()
                fig_pie = go.Figure(
                    data=go.Pie(
                        labels=class_counts.index,
                        values=class_counts.values,
                        hole=0.4,
                        marker_colors=['#2ecc71', '#e74c3c']
                    )
                )
                fig_pie.update_layout(
                    title="Event Classification Distribution",
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_summary2:
                # Confidence distribution
                conf_counts = results['confidence'].value_counts()
                fig_bar = go.Figure(
                    data=go.Bar(
                        x=conf_counts.index,
                        y=conf_counts.values,
                        marker_color=['#f39c12', '#3498db', '#e74c3c']
                    )
                )
                fig_bar.update_layout(
                    title="Confidence Level Distribution",
                    xaxis_title="Confidence Level",
                    yaxis_title="Number of Events",
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig_bar, use_container_width=True)
    
    else:
        st.info("📊 Run analysis first to generate visualizations")

# Real-time Data Tab
with tab5:
    st.markdown("## 🌐 Real-time Data Streaming")
    
    # Real-time controls
    col_rt1, col_rt2, col_rt3 = st.columns(3)
    
    with col_rt1:
        if st.button("🟢 Start Live Stream", type="primary"):
            st.session_state.live_stream = True
            data_stream.start()
            st.success("🚀 Live stream started!")
    
    with col_rt2:
        if st.button("🔴 Stop Stream", type="secondary"):
            st.session_state.live_stream = False
            data_stream.stop()
            st.info("⏹️ Live stream stopped")
    
    with col_rt3:
        auto_refresh = st.checkbox("Auto Refresh", value=True)
        refresh_rate = st.slider("Refresh Rate (sec)", 1, 10, 3)
    
    # Stream status
    stream_status = "🟢 LIVE" if st.session_state.live_stream else "🔴 OFFLINE"
    st.markdown(f"### Stream Status: {stream_status}")
    
    # Real-time data display
    if st.session_state.live_stream:
        # Generate new real-time events
        for _ in range(np.random.randint(1, 4)):  # Generate 1-3 events per refresh
            new_event = data_stream.generate_event()
            if new_event and new_event not in st.session_state.real_time_data:
                st.session_state.real_time_data.append(new_event)
        
        # Display recent events
        if st.session_state.real_time_data:
            st.markdown("### 📡 Live Event Feed")
            
            # Real-time metrics
            rt_data = st.session_state.real_time_data
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            with col_m1:
                st.metric("Total Events", len(rt_data))
            
            with col_m2:
                gw_events = len([e for e in rt_data if e['event_type'] == 'GW'])
                st.metric("GW Events", gw_events)
            
            with col_m3:
                gamma_events = len([e for e in rt_data if e['event_type'] == 'Gamma'])
                st.metric("Gamma Events", gamma_events)
            
            with col_m4:
                recent_events = len([e for e in rt_data if 
                                   (datetime.now() - e['timestamp']).seconds < 300])
                st.metric("Recent (5min)", recent_events)
            
            # Live event table (last 20 events)
            st.markdown("#### 📋 Recent Events")
            
            recent_data = rt_data[-20:]
            if recent_data:
                display_df = pd.DataFrame(recent_data)
                display_df['timestamp'] = display_df['timestamp'].dt.strftime('%H:%M:%S')
                display_df = display_df[['timestamp', 'detector', 'event_type', 'confidence', 'snr']]
                display_df = display_df.round(3)
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=350
                )
            
            # Real-time visualization
            if len(rt_data) > 5:
                st.markdown("#### 📈 Live Event Distribution")
                
                rt_df = pd.DataFrame(rt_data)
                
                # Event type distribution
                event_dist = rt_df['event_type'].value_counts()
                
                fig = go.Figure(data=[
                    go.Pie(
                        labels=event_dist.index,
                        values=event_dist.values,
                        hole=0.4,
                        marker_colors=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7']
                    )
                ])
                
                fig.update_layout(
                    title="Live Event Type Distribution",
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Auto-refresh
        if auto_refresh:
            time.sleep(refresh_rate)
            st.rerun()
    
    else:
        st.info("🔴 Start live stream to see real-time data")
        
        # Show historical data if available
        if st.session_state.real_time_data:
            st.markdown("### 📊 Historical Stream Data")
            
            hist_df = pd.DataFrame(st.session_state.real_time_data)
            
            # Summary stats
            col_h1, col_h2, col_h3 = st.columns(3)
            
            with col_h1:
                st.metric("Total Events Captured", len(hist_df))
            
            with col_h2:
                if len(hist_df) > 0:
                    avg_confidence = hist_df['confidence'].mean()
                    st.metric("Average Confidence", f"{avg_confidence:.3f}")
            
            with col_h3:
                if len(hist_df) > 0:
                    high_snr = (hist_df['snr'] > 15).sum()
                    st.metric("High SNR Events", high_snr)

# APIs Tab
with tab6:
    st.markdown("## ⚙️ External API Integrations")
    
    # API status overview
    st.markdown("### 🔌 Observatory API Status")
    
    api_services = {
        'GraceDB': {'url': 'https://gracedb.ligo.org', 'desc': 'Gravitational Wave Event Database'},
        'LIGO OSC': {'url': 'https://losc.ligo.org', 'desc': 'LIGO Open Science Center'},
        'Fermi LAT': {'url': 'https://fermi.gsfc.nasa.gov', 'desc': 'Fermi Gamma-ray Space Telescope'},
        'IceCube': {'url': 'https://icecube.wisc.edu', 'desc': 'IceCube Neutrino Observatory'}
    }
    
    for service_name, info in api_services.items():
        col_api1, col_api2, col_api3 = st.columns([1, 2, 1])
        
        with col_api1:
            status = api_manager.check_api_status(service_name)
            status_emoji = "🟢" if status == 'online' else "🔴"
            st.markdown(f"**{status_emoji} {service_name}**")
        
        with col_api2:
            st.markdown(f"*{info['desc']}*")
            st.markdown(f"`{info['url']}`")
        
        with col_api3:
            if st.button(f"Test {service_name}", key=f"test_{service_name}"):
                with st.spinner(f"Testing {service_name}..."):
                    status = api_manager.check_api_status(service_name)
                    if status == 'online':
                        st.success("✅ Connection OK")
                    else:
                        st.error("❌ Connection Failed")
    
    # Mock data fetching
    st.markdown("### 📡 Fetch Mock Observatory Data")
    
    col_fetch1, col_fetch2 = st.columns(2)
    
    with col_fetch1:
        st.markdown("#### 🌊 Gravitational Wave Events")
        
        gw_limit = st.number_input("Number of GW events:", 1, 50, 10, key="gw_limit")
        
        if st.button("🌊 Fetch GW Events", type="primary"):
            with st.spinner("Fetching mock GW events..."):
                gw_events = api_manager.generate_mock_gw_events(gw_limit)
                
                if gw_events:
                    st.session_state.api_data['gracedb'] = gw_events
                    
                    gw_df = pd.DataFrame(gw_events)
                    
                    st.markdown(f"""
                    <div class="success-card">
                        ✅ Fetched {len(gw_events)} mock GW events
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display sample
                    display_cols = ['event_id', 'classification', 'distance', 'snr']
                    st.dataframe(gw_df[display_cols].head(), use_container_width=True)
    
    with col_fetch2:
        st.markdown("#### 🌟 Gamma-ray Events")
        
        gamma_limit = st.number_input("Number of gamma events:", 1, 50, 10, key="gamma_limit")
        
        if st.button("🌟 Fetch Gamma Events", type="primary"):
            with st.spinner("Fetching mock gamma-ray events..."):
                gamma_events = api_manager.generate_mock_gamma_events(gamma_limit)
                
                if gamma_events:
                    st.session_state.api_data['fermi'] = gamma_events
                    
                    gamma_df = pd.DataFrame(gamma_events)
                    
                    st.markdown(f"""
                    <div class="success-card">
                        ✅ Fetched {len(gamma_events)} mock gamma-ray events
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display sample
                    display_cols = ['event_id', 'energy', 'significance', 'error_radius']
                    st.dataframe(gamma_df[display_cols].head(), use_container_width=True)
    
    # Data integration
    if st.session_state.api_data:
        st.markdown("### 🔄 Integrate External Data")
        
        available_sources = list(st.session_state.api_data.keys())
        selected_sources = st.multiselect("Select data sources to integrate:", available_sources)
        
        if selected_sources and st.button("🔄 Create Integrated Dataset", type="primary"):
            with st.spinner("Integrating external data sources..."):
                integrated_data = []
                
                for source in selected_sources:
                    source_data = st.session_state.api_data[source]
                    
                    # Convert to standard format for analysis
                    for event in source_data:
                        if source == 'gracedb':
                            std_event = {
                                'event_id': event['event_id'],
                                'ra': event['ra'],
                                'dec': event['dec'],
                                'timestamp': datetime.fromtimestamp(event['gpstime']),
                                'source': 'GraceDB',
                                'event_type': 'GW',
                                'confidence': min(1.0 - event['far'], 1.0),
                                'dt': np.random.exponential(1000),  # Mock time difference
                                'dtheta': np.random.exponential(1.0),  # Mock angular separation
                                'strength_ratio': event['snr'] / 10.0  # Mock strength ratio
                            }
                        elif source == 'fermi':
                            std_event = {
                                'event_id': event['event_id'],
                                'ra': event['ra'],
                                'dec': event['dec'],
                                'timestamp': event['trigger_time'],
                                'source': 'Fermi',
                                'event_type': 'Gamma',
                                'confidence': min(event['significance'] / 20.0, 1.0),
                                'dt': np.random.exponential(500),
                                'dtheta': event['error_radius'],
                                'strength_ratio': event['fluence'] * 1e6
                            }
                        
                        # Add pair information for analysis
                        std_event['pair'] = 'GW_Gamma' if source == 'gracedb' else 'Gamma_Neutrino'
                        
                        integrated_data.append(std_event)
                
                if integrated_data:
                    integrated_df = pd.DataFrame(integrated_data)
                    st.session_state.data = integrated_df
                    
                    st.markdown(f"""
                    <div class="success-card">
                        ✅ Successfully integrated {len(integrated_data)} events from {len(selected_sources)} external sources
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show integrated data preview
                    st.markdown("#### 📋 Integrated Dataset Preview")
                    preview_cols = ['event_id', 'source', 'event_type', 'confidence', 'timestamp']
                    st.dataframe(integrated_df[preview_cols].head(10), use_container_width=True)
                    
                    st.info("💡 Use the AI Analysis tab to analyze this integrated dataset!")

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem;">
    <div class="feature-card">
        <h3 style="color: white; margin: 0;">🌌 Enhanced Multimessenger AI Platform v2.0</h3>
        <p style="color: rgba(255, 255, 255, 0.8); margin: 0.5rem 0 0 0;">
            Next-Generation AI Analysis • Real-time Streaming • API Integration • Advanced Visualizations • Enhanced UI/UX
        </p>
        <div style="margin-top: 1rem; color: rgba(255, 255, 255, 0.6);">
            <small>Powered by Streamlit • Machine Learning • Plotly • Modern Web Technologies</small>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)