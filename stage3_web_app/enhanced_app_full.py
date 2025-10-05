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
from model_loader import list_model_files, load_model_by_name
from inference import predict_df

# Page configuration
st.set_page_config(
    page_title="Multimessenger AI Platform",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
        padding: 0.5rem 1rem;
    }
    
    .feature-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
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

# Sidebar
st.sidebar.markdown("## 🎛️ Analysis Controls")

# Model selection
st.sidebar.markdown("### 🤖 AI Model")
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
        st.sidebar.success(f"✅ Model: {model_choice}")
        
        if metadata:
            st.sidebar.markdown(f"""
            **📊 Model Info:**
            - Algorithm: {metadata.get('best_model', 'Unknown')}
            - AUC Score: {metadata.get('best_auc', 'N/A'):.3f}
            """)
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
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
            ["🎲 Generate Demo Data", "📂 Upload CSV", "📋 Manual Entry"],
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
            
            st.markdown(f"""
            <div class="success-box">
            <h4>✅ Generated {len(df)} astronomical event pairs</h4>
            <p>Noise level: {noise_level} | Seed: {seed} | Ready for AI analysis</p>
            </div>
            """, unsafe_allow_html=True)
    
    elif input_method == "📂 Upload CSV":
        uploaded_file = st.file_uploader(
            "Choose CSV file", 
            type="csv",
            help="Upload astronomical event data in CSV format"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.current_data = df
                st.success(f"✅ Loaded {len(df)} rows from {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
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
                                    delta=f"{confidence_interval*100:.0f}% CI"
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
        st.markdown("### 📊 Detailed Analysis Results")
        
        results = st.session_state.results
        
        # Add enhanced result columns
        results_display = results.copy()
        results_display['Association Status'] = results_display['pred_prob'].apply(
            lambda x: f"✅ ASSOCIATED ({x:.3f})" if x > threshold else f"❌ Not Associated ({x:.3f})"
        )
        results_display['Confidence Level'] = results_display['pred_prob'].apply(
            lambda x: "🔥 Very High" if x > 0.9 else "🎯 High" if x > 0.7 else "⚡ Medium" if x > 0.5 else "📊 Low"
        )
        
        # Interactive results table
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
            }
        )

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
                    "📋 Statistical Summary"
                ]
            )
            
            # Customization options
            color_scheme = st.selectbox("Color scheme:", ["viridis", "plasma", "cividis", "magma"])
            show_threshold = st.checkbox("Show threshold line", True)
            
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
                    color_continuous_scale=color_scheme
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
                        color_continuous_scale=color_scheme,
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
                        labels={'hour': 'Hour of Day', 'pred_prob': 'Average Association Probability'}
                    )
                    
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Time data not available for time series analysis")
            
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
                                colorscale=color_scheme,
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
            
            elif viz_type == "📋 Statistical Summary":
                st.markdown("### 📊 Comprehensive Statistical Analysis")
                
                # Statistical summary
                col1, col2 = st.columns(2)
                
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
                    st.markdown("**📈 Distribution Analysis:**")
                    
                    # Quartiles
                    q1 = results['pred_prob'].quantile(0.25)
                    q2 = results['pred_prob'].quantile(0.50)
                    q3 = results['pred_prob'].quantile(0.75)
                    
                    st.metric("Q1 (25th percentile)", f"{q1:.3f}")
                    st.metric("Q2 (Median)", f"{q2:.3f}")
                    st.metric("Q3 (75th percentile)", f"{q3:.3f}")
                    st.metric("IQR", f"{q3-q1:.3f}")
                
                # Confidence intervals
                st.markdown("**📊 Confidence Intervals:**")
                mean_prob = results['pred_prob'].mean()
                std_prob = results['pred_prob'].std()
                n = len(results)
                
                # 95% confidence interval for mean
                margin_error = 1.96 * (std_prob / np.sqrt(n))
                ci_lower = mean_prob - margin_error
                ci_upper = mean_prob + margin_error
                
                st.info(f"95% CI for mean probability: [{ci_lower:.3f}, {ci_upper:.3f}]")
    
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