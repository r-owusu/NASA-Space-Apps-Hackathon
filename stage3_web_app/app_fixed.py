#!/usr/bin/env python3
"""
Fixed version of the multimessenger analysis app with robust analysis button
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
from datetime import datetime
from model_loader import list_model_files, load_model_by_name
from inference import predict_df

# Page configuration
st.set_page_config(
    page_title="Multimessenger AI Analysis",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌌 Multimessenger AI Analysis Platform")
st.markdown("*AI-powered detection and analysis of multimessenger astronomical events*")

# Sidebar for controls
st.sidebar.header("🎛️ Analysis Controls")

# Model selection
model_files = list_model_files()
st.sidebar.subheader("🤖 Select Model")
model_choice = st.sidebar.selectbox(
    "Choose trained model:",
    ["(none)"] + model_files,
    key="model_selector"
)

# Initialize session state
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Load model
model = None
scaler = None
metadata = None

if model_choice and model_choice != "(none)":
    try:
        model, scaler, metadata = load_model_by_name(model_choice)
        st.sidebar.success(f"✅ Model loaded: {model_choice}")
        st.session_state.model_loaded = True
        
        if metadata:
            st.sidebar.info(f"🎯 Model: {metadata.get('best_model', 'Unknown')}")
            st.sidebar.info(f"📊 AUC: {metadata.get('best_auc', 'N/A'):.3f}")
            
    except Exception as e:
        st.sidebar.error(f"❌ Model load error: {e}")
        st.session_state.model_loaded = False
else:
    st.session_state.model_loaded = False

# Analysis threshold
threshold = st.sidebar.slider("🎯 Association Threshold", 0.0, 1.0, 0.5, 0.05)

# Data input section
st.header("📊 Data Input")

# Data input method selection
input_method = st.radio(
    "Choose data input method:",
    ["📁 Load sample/demo data", "📂 Upload CSV file"],
    horizontal=True
)

df = None

if input_method == "📁 Load sample/demo data":
    if st.button("🚀 Generate Demo Data", type="secondary"):
        # Create comprehensive demo data
        np.random.seed(42)
        n_events = 50
        
        messengers = ['GW', 'Gamma', 'Neutrino', 'Optical', 'Radio']
        data = {
            'dt': np.random.exponential(1.0, n_events),
            'dtheta': np.random.exponential(0.1, n_events),
            'strength_ratio': np.random.lognormal(0, 1, n_events),
            'ra': np.random.uniform(0, 360, n_events),
            'dec': np.random.uniform(-90, 90, n_events),
            'm1': np.random.choice(messengers, n_events),
            'm2': np.random.choice(messengers, n_events)
        }
        
        df = pd.DataFrame(data)
        # Ensure m1 != m2
        mask = df['m1'] == df['m2']
        df.loc[mask, 'm2'] = df.loc[mask, 'm1'].apply(
            lambda x: np.random.choice([m for m in messengers if m != x])
        )
        
        st.session_state.current_data = df
        st.success(f"✅ Demo data generated: {len(df)} events")

elif input_method == "📂 Upload CSV file":
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="File should contain columns: dt, dtheta, strength_ratio"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.current_data = df
            st.success(f"✅ File uploaded: {len(df)} events")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

# Use data from session state
if st.session_state.current_data is not None:
    df = st.session_state.current_data
    
    # Data overview
    st.header("📋 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Events", len(df))
    with col2:
        unique_messengers = set()
        if 'm1' in df.columns and 'm2' in df.columns:
            unique_messengers.update(df['m1'].unique())
            unique_messengers.update(df['m2'].unique())
        st.metric("🌌 Messenger Types", len(unique_messengers))
    with col3:
        time_span = df['dt'].max() - df['dt'].min() if 'dt' in df.columns else 0
        st.metric("⏱️ Time Span", f"{time_span:.1f}s")
    with col4:
        max_separation = df['dtheta'].max() if 'dtheta' in df.columns else 0
        st.metric("📐 Max Angular Sep", f"{max_separation:.2f}°")
    
    # Data preview
    with st.expander("📋 View Data Preview", expanded=False):
        st.dataframe(df.head(20), width="stretch")
    
    # Analysis section
    st.header("🚀 Multimessenger Analysis")
    
    # Clear results button
    if st.button("🧹 Clear Previous Results", type="secondary"):
        st.session_state.results = None
        st.success("✅ Results cleared")
    
    # Analysis button - using a unique approach to force update
    analysis_key = f"analyze_{len(df)}_{hash(str(df.iloc[0].to_dict()) if len(df) > 0 else 'empty')}"
    
    if st.button("🔍 **Run Analysis**", key=analysis_key, type="primary"):
        if not st.session_state.model_loaded or model is None:
            st.error("❌ Please select a trained model first!")
        else:
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("🔬 Preparing data...")
                progress_bar.progress(25)
                
                status_text.text("🤖 Running AI analysis...")
                progress_bar.progress(50)
                
                # Run prediction
                df_pred = predict_df(df, model, scaler=scaler, threshold=threshold)
                progress_bar.progress(75)
                
                status_text.text("📊 Processing results...")
                progress_bar.progress(100)
                
                # Store results
                st.session_state.results = df_pred
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                st.success("✅ Analysis completed successfully!")
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Analysis failed: {e}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())

# Display results if available
if st.session_state.results is not None:
    df_pred = st.session_state.results
    
    st.header("🎯 Analysis Results")
    
    # Results metrics
    total_events = len(df_pred)
    high_confidence = len(df_pred[df_pred['pred_prob'] >= 0.8])
    medium_confidence = len(df_pred[(df_pred['pred_prob'] >= 0.5) & (df_pred['pred_prob'] < 0.8)])
    positive_associations = len(df_pred[df_pred['pred_label'] == 1])
    avg_confidence = df_pred['pred_prob'].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔴 High Confidence", high_confidence, 
                 delta=f"{high_confidence/total_events*100:.1f}%")
    with col2:
        st.metric("🟡 Medium Confidence", medium_confidence,
                 delta=f"{medium_confidence/total_events*100:.1f}%")
    with col3:
        st.metric("✅ Positive Associations", positive_associations,
                 delta=f"Threshold: {threshold}")
    with col4:
        st.metric("📊 Average Confidence", f"{avg_confidence:.3f}")
    
    # Results tabs
    tab1, tab2, tab3 = st.tabs(["🎯 High Priority", "📈 All Results", "📊 Visualizations"])
    
    with tab1:
        high_priority = df_pred[df_pred['pred_prob'] >= 0.7].sort_values('pred_prob', ascending=False)
        
        if len(high_priority) > 0:
            st.markdown(f"**{len(high_priority)} high-priority associations found:**")
            st.dataframe(
                high_priority[['m1', 'm2', 'dt', 'dtheta', 'strength_ratio', 'pred_prob', 'pred_label']],
                width="stretch"
            )
        else:
            st.info("No high-priority associations found above 70% confidence threshold")
    
    with tab2:
        df_display = df_pred.sort_values('pred_prob', ascending=False)
        st.dataframe(df_display, width="stretch")
        
        # Download button
        csv = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name=f'analysis_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv'
        )
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Confidence distribution
            fig_conf = px.histogram(
                df_pred, x='pred_prob', nbins=20,
                title='🎯 Confidence Score Distribution',
                labels={'pred_prob': 'Confidence Score', 'count': 'Number of Events'}
            )
            st.plotly_chart(fig_conf, width="stretch")
        
        with col2:
            # Time vs angular separation
            fig_scatter = px.scatter(
                df_pred, x='dt', y='dtheta', color='pred_prob',
                title='⏱️ Time vs Angular Separation',
                labels={'dt': 'Time Difference (s)', 'dtheta': 'Angular Separation (°)'},
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_scatter, width="stretch")

else:
    if st.session_state.current_data is not None:
        st.info("ℹ️ Data loaded. Click 'Run Analysis' to start the analysis.")
    else:
        st.info("ℹ️ Please load data first, then run analysis.")

# Footer
st.markdown("---")
st.markdown("🌌 **Multimessenger AI Platform** | Built for astronomical event analysis")