import streamlit as st, os, pandas as pd, numpy as np
from model_loader import list_model_files, load_model_by_name
from inference import predict_df
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

Path('uploads').mkdir(exist_ok=True)
Path('results').mkdir(exist_ok=True)
Path('saved_models').mkdir(exist_ok=True)
Path('sample_data').mkdir(exist_ok=True)
Path('alerts').mkdir(exist_ok=True)

st.set_page_config(page_title='Multimessenger Astronomy AI', layout='wide', page_icon='🌟')

# Header
st.markdown("""
# 🌟 Multimessenger Astronomy AI Analysis Platform
**Real-time Detection and Analysis of Cosmic Multimessenger Events**

This platform uses advanced machine learning to identify genuine multimessenger associations between gravitational waves, 
gamma rays, neutrinos, optical transients, and radio signals - helping astronomers discover major astrophysical events 
like neutron star mergers, black hole collisions, and supernovae.
""")

# Sidebar configuration
st.sidebar.markdown("## 🔧 Analysis Configuration")
model_files = list_model_files()
model_choice = st.sidebar.selectbox('🤖 ML Model', options=model_files if model_files else ['(none)'], 
                                   help="Select the trained model for multimessenger association analysis")
threshold = st.sidebar.slider('🎯 Confidence Threshold', 0.0, 1.0, 0.5, 0.01,
                             help="Minimum probability to classify as a true multimessenger association")

st.sidebar.markdown("## 📊 Data Input")
file = st.sidebar.file_uploader('📁 Upload Detection Data (CSV)', type=['csv'],
                               help="Upload CSV file with multimessenger detection pairs")
use_sample = st.sidebar.button('🧪 Load Demo Data', help="Use synthetic multimessenger data for testing")

# Model loading and status
model=None; scaler=None; metadata=None
if model_choice and model_choice!='(none)':
    try:
        model, scaler, metadata = load_model_by_name(model_choice)
        st.sidebar.success(f'✅ Model loaded: {model_choice}')
        if metadata:
            st.sidebar.info(f"🎯 Model Type: {metadata.get('best_model', 'Unknown')}\n📊 Validation AUC: {metadata.get('best_auc', 'N/A'):.3f}")
    except Exception as e:
        st.sidebar.error(f'❌ Load failed: {e}')

# Data loading section

# Data loading section
if use_sample:
    sample_path = os.path.join('sample_data','sample_pairs.csv')
    if not os.path.exists(sample_path):
        import numpy as np
        rng = np.random.RandomState(0)
        # Create more realistic sample data with messenger pairs
        n_samples = 500
        messengers = ['GW', 'Gamma', 'Neutrino', 'Optical', 'Radio']
        pairs = [(m1, m2) for i, m1 in enumerate(messengers) for m2 in messengers[i+1:]]
        
        # Generate sample data
        pair_choices = [pairs[i % len(pairs)] for i in range(n_samples)]
        m1_list = [p[0] for p in pair_choices]
        m2_list = [p[1] for p in pair_choices]
        
        df = pd.DataFrame({
            'm1': m1_list,
            'm2': m2_list,
            'dt': rng.normal(0, 50, n_samples),  # Time difference in seconds
            'dtheta': np.abs(rng.normal(5, 3, n_samples)),  # Angular separation in degrees
            'strength_ratio': np.abs(rng.normal(1, 0.8, n_samples)),  # Strength ratio
            'ra': rng.uniform(0, 360, n_samples),  # Right ascension
            'dec': rng.uniform(-90, 90, n_samples)  # Declination
        })
        df.to_csv(sample_path, index=False)
    df = pd.read_csv(sample_path)
    st.success('✅ Demo multimessenger data loaded successfully')
elif file:
    df = pd.read_csv(file)
    outp = os.path.join('uploads', file.name)
    df.to_csv(outp, index=False)
    st.success(f'✅ Data uploaded and saved to {outp}')
else:
    df = None

# Main analysis interface
if df is not None:
    # Data overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Detections", len(df))
    with col2:
        unique_messengers = set()
        if 'm1' in df.columns and 'm2' in df.columns:
            unique_messengers.update(df['m1'].unique())
            unique_messengers.update(df['m2'].unique())
        st.metric("🌌 Messenger Types", len(unique_messengers) if unique_messengers else "N/A")
    with col3:
        time_span = df['dt'].max() - df['dt'].min() if 'dt' in df.columns else 0
        st.metric("⏱️ Time Span", f"{time_span:.1f}s" if time_span > 0 else "N/A")
    with col4:
        max_separation = df['dtheta'].max() if 'dtheta' in df.columns else 0
        st.metric("📐 Max Angular Sep", f"{max_separation:.2f}°" if max_separation > 0 else "N/A")
    
    # Data preview section
    st.markdown("### 📋 Detection Data Preview")
    with st.expander("View raw detection data", expanded=False):
        st.dataframe(df.head(100), width='stretch')
    # Analysis section
    st.markdown("### 🚀 Run Multimessenger Association Analysis")
    
    if st.button('🔍 **Analyze Multimessenger Associations**', type="primary", width='stretch'):
        if model is None:
            st.error('❌ Please select a trained model first')
        else:
            try:
                with st.spinner('🔬 Analyzing multimessenger associations...'):
                    df_pred = predict_df(df, model, scaler=scaler, threshold=threshold)
                
                # Analysis results metrics
                total_events = len(df_pred)
                high_confidence = len(df_pred[df_pred['pred_prob'] >= 0.8])
                medium_confidence = len(df_pred[(df_pred['pred_prob'] >= 0.5) & (df_pred['pred_prob'] < 0.8)])
                positive_associations = len(df_pred[df_pred['pred_label'] == 1])
                
                st.markdown("### 🎯 Analysis Results")
                
                # Results metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🔴 High Confidence", high_confidence, 
                             delta=f"{high_confidence/total_events*100:.1f}% of total")
                with col2:
                    st.metric("🟡 Medium Confidence", medium_confidence,
                             delta=f"{medium_confidence/total_events*100:.1f}% of total") 
                with col3:
                    st.metric("✅ Positive Associations", positive_associations,
                             delta=f"Threshold: {threshold}")
                with col4:
                    avg_confidence = df_pred['pred_prob'].mean()
                    st.metric("📊 Avg Confidence", f"{avg_confidence:.3f}")
                
                # Detailed results
                st.markdown("### 📊 Detailed Association Results")
                
                # Sort by confidence and show top results
                df_display = df_pred.sort_values('pred_prob', ascending=False)
                
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["🎯 High Priority Events", "📈 All Predictions", "🌌 Sky Distribution"])
                
                with tab1:
                    high_priority = df_display[df_display['pred_prob'] >= 0.7]
                    if len(high_priority) > 0:
                        st.markdown(f"**{len(high_priority)} high-priority multimessenger associations found:**")
                        st.dataframe(high_priority[['m1', 'm2', 'dt', 'dtheta', 'strength_ratio', 'pred_prob', 'pred_label']], 
                                   width='stretch')
                        
                        # Alert generation for high confidence events
                        if st.button("🚨 Generate Alerts for High-Priority Events"):
                            alert_file = f"alerts/multimessenger_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                            high_priority.to_csv(alert_file, index=False)
                            st.success(f"✅ Alert file generated: {alert_file}")
                    else:
                        st.info("No high-priority associations found above 70% confidence threshold")
                
                with tab2:
                    st.dataframe(df_display.head(200), width='stretch')
                    
                    # Download predictions
                    csv = df_display.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Full Analysis Results",
                        data=csv,
                        file_name=f'multimessenger_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                        mime='text/csv'
                    )
                
                with tab3:
                    if 'ra' in df_pred.columns and 'dec' in df_pred.columns:
                        # Sky map
                        fig_sky = px.scatter(df_pred, x='ra', y='dec', color='pred_prob',
                                           title='🌌 Multimessenger Associations Sky Distribution',
                                           labels={'ra': 'Right Ascension (degrees)', 'dec': 'Declination (degrees)'},
                                           color_continuous_scale='Viridis',
                                           hover_data=['dt', 'dtheta', 'strength_ratio'])
                        fig_sky.update_layout(xaxis_title="Right Ascension (°)", yaxis_title="Declination (°)")
                        st.plotly_chart(fig_sky, width='stretch')
                    else:
                        st.info("Sky coordinates (RA/Dec) not available in dataset")
                
                # Visualization section
                st.markdown("### 📈 Analysis Visualizations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Confidence distribution
                    fig1 = px.histogram(df_pred, x='pred_prob', nbins=40, 
                                      title='🎯 Association Confidence Distribution',
                                      labels={'pred_prob': 'Confidence Score', 'count': 'Number of Events'})
                    fig1.add_vline(x=threshold, line_dash="dash", line_color="red", 
                                  annotation_text=f"Threshold: {threshold}")
                    st.plotly_chart(fig1, width='stretch')
                
                with col2:
                    # Time vs Angular separation plot
                    if 'dt' in df_pred.columns and 'dtheta' in df_pred.columns:
                        fig2 = px.scatter(df_pred, x='dt', y='dtheta', color='pred_prob',
                                        title='⏱️ Time vs Angular Separation',
                                        labels={'dt': 'Time Difference (s)', 'dtheta': 'Angular Separation (°)'},
                                        color_continuous_scale='Viridis')
                        st.plotly_chart(fig2, width='stretch')
                
                # Messenger pair analysis
                if 'm1' in df_pred.columns and 'm2' in df_pred.columns:
                    st.markdown("### 🔬 Messenger Pair Analysis")
                    pair_stats = df_pred.groupby(['m1', 'm2']).agg({
                        'pred_prob': ['count', 'mean', 'max'],
                        'pred_label': 'sum'
                    }).round(3)
                    pair_stats.columns = ['Total Events', 'Avg Confidence', 'Max Confidence', 'Positive Associations']
                    st.dataframe(pair_stats, width='stretch')
                
            except Exception as e:
                st.error(f'❌ Analysis error: {e}')
                st.error("Please check your data format and try again.")

else:
    st.markdown("""
    ### 🚀 Getting Started
    
    1. **Load Data**: Upload your multimessenger detection CSV file or use demo data
    2. **Select Model**: Choose a trained ML model from the sidebar  
    3. **Set Threshold**: Adjust the confidence threshold for classifications
    4. **Run Analysis**: Click the analysis button to identify associations
    
    ### 📊 Required Data Format
    Your CSV should contain columns for:
    - `dt`: Time difference between detections (seconds)
    - `dtheta`: Angular separation on sky (degrees) 
    - `strength_ratio`: Ratio of signal strengths
    - `m1`, `m2`: Messenger types (optional but recommended)
    - `ra`, `dec`: Sky coordinates (optional, for visualization)
    
    ### 🌟 What This Analysis Does
    This platform identifies genuine multimessenger associations by analyzing:
    - **Temporal correlations**: Events close in time
    - **Spatial correlations**: Events close on the sky
    - **Signal characteristics**: Strength and duration patterns
    - **Messenger combinations**: Different physics for different pairs
    
    High-confidence associations may indicate major astrophysical events worthy of follow-up observations!
    """)
