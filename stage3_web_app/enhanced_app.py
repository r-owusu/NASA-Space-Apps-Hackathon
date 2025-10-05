import streamlit as st
import pandas as pd
import numpy as np
from model_loader import list_model_files, load_model_by_name
from inference import predict_df
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import json
import io
import os

# Helper function to safely create directories
def safe_mkdir(path):
    try:
        Path(path).mkdir(exist_ok=True)
    except PermissionError:
        # If we can't create directories, that's ok - we'll work without them
        pass

# Try to initialize directories (optional)
safe_mkdir('uploads')
safe_mkdir('results')
safe_mkdir('saved_models')
safe_mkdir('sample_data')
safe_mkdir('alerts')

st.set_page_config(page_title='Enhanced Multimessenger AI', layout='wide', page_icon='🌟')

# Analysis functions (defined early)
def analyze_data(df, model, scaler, threshold, max_events):
    """Perform the multimessenger analysis"""
    try:
        # Limit data if necessary
        if len(df) > max_events:
            df = df.head(max_events)
            st.warning(f"⚠️ Analysis limited to {max_events} events")
        
        with st.spinner('🔬 Analyzing multimessenger associations...'):
            df_pred = predict_df(df, model, scaler=scaler, threshold=threshold)
        
        # Store results
        st.session_state['analysis_results'] = df_pred
        
        # Display results
        display_analysis_results(df_pred, threshold)
        
    except Exception as e:
        st.error(f'❌ Analysis error: {e}')
        if st.checkbox("Show detailed error"):
            import traceback
            st.code(traceback.format_exc())

def display_analysis_results(df_pred, threshold):
    """Display comprehensive analysis results"""
    # Calculate metrics
    total_events = len(df_pred)
    high_confidence = len(df_pred[df_pred['pred_prob'] >= 0.8])
    medium_confidence = len(df_pred[(df_pred['pred_prob'] >= 0.5) & (df_pred['pred_prob'] < 0.8)])
    positive_associations = len(df_pred[df_pred['pred_label'] == 1])
    avg_confidence = df_pred['pred_prob'].mean()
    
    st.markdown("## 🎯 Analysis Results")
    
    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📊 Total Events", total_events)
    with col2:
        st.metric("🔴 High Confidence", high_confidence, delta=f"{high_confidence/total_events*100:.1f}%")
    with col3:
        st.metric("🟡 Medium Confidence", medium_confidence, delta=f"{medium_confidence/total_events*100:.1f}%")
    with col4:
        st.metric("✅ Positive Associations", positive_associations, delta=f"Threshold: {threshold}")
    with col5:
        st.metric("📈 Avg Confidence", f"{avg_confidence:.3f}")
    
    # Results tabs
    tab1, tab2, tab3 = st.tabs(["🎯 High Priority", "📊 All Results", "📈 Statistics"])
    
    with tab1:
        high_priority = df_pred[df_pred['pred_prob'] >= 0.7].sort_values('pred_prob', ascending=False)
        if len(high_priority) > 0:
            st.markdown(f"**{len(high_priority)} high-priority associations found:**")
            st.dataframe(high_priority[['m1', 'm2', 'dt', 'dtheta', 'strength_ratio', 'pred_prob', 'pred_label']], 
                       use_container_width=True)
        else:
            st.info("No high-priority associations found")
    
    with tab2:
        df_display = df_pred.sort_values('pred_prob', ascending=False)
        st.dataframe(df_display.head(200), use_container_width=True)
        
        # Download button
        csv = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name=f'analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv'
        )
    
    with tab3:
        # Statistics and correlations
        if 'm1' in df_pred.columns and 'm2' in df_pred.columns:
            pair_stats = df_pred.groupby(['m1', 'm2']).agg({
                'pred_prob': ['count', 'mean', 'max', 'std'],
                'pred_label': 'sum'
            }).round(3)
            pair_stats.columns = ['Count', 'Mean Prob', 'Max Prob', 'Std Prob', 'Positive']
            st.dataframe(pair_stats, use_container_width=True)
    
    # Event Source Classification Analysis
    st.markdown("## 🌠 Event Source Classification")
    
    # Calculate clustering metrics for event classification
    high_conf_events = df_pred[df_pred['pred_prob'] >= 0.7]
    
    if len(high_conf_events) > 0:
        # Time clustering analysis
        time_clustering = 0.0
        if 'dt' in high_conf_events.columns:
            time_std = high_conf_events['dt'].std()
            time_clustering = max(0, min(1, 1 - (time_std / 100)))  # Normalize to 0-1
        
        # Spatial clustering analysis  
        spatial_clustering = 0.0
        if 'dtheta' in high_conf_events.columns:
            spatial_std = high_conf_events['dtheta'].std()
            spatial_clustering = max(0, min(1, 1 - (spatial_std / 10)))  # Normalize to 0-1
        
        # Strength correlation analysis
        strength_correlation = 0.0
        if 'strength_ratio' in high_conf_events.columns and len(high_conf_events) > 1:
            strength_var = high_conf_events['strength_ratio'].var()
            strength_correlation = max(0, min(1, 1 - (strength_var / 2)))  # Normalize to 0-1
        
        # Combined single-event probability
        single_event_factors = []
        factor_names = []
        
        if time_clustering > 0:
            single_event_factors.append(time_clustering)
            factor_names.append(f"Time Clustering: {time_clustering:.2f}")
        
        if spatial_clustering > 0:
            single_event_factors.append(spatial_clustering)
            factor_names.append(f"Spatial Clustering: {spatial_clustering:.2f}")
        
        if strength_correlation > 0:
            single_event_factors.append(strength_correlation)
            factor_names.append(f"Strength Correlation: {strength_correlation:.2f}")
        
        # Weight by confidence and number of associations
        confidence_weight = high_conf_events['pred_prob'].mean()
        association_weight = min(1.0, len(high_conf_events) / 5.0)  # More associations = higher confidence
        
        if single_event_factors:
            base_probability = np.mean(single_event_factors)
            single_event_probability = base_probability * confidence_weight * association_weight
        else:
            single_event_probability = confidence_weight * 0.5  # Default moderate probability
        
        # Ensure probability is between 0 and 1
        single_event_probability = max(0.0, min(1.0, single_event_probability))
        multi_event_probability = 1.0 - single_event_probability
        
        # Display results with visual indicators
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Single Cosmic Event")
            if single_event_probability > 0.7:
                color = "#28a745"  # Green
                icon = "🟢"
            elif single_event_probability > 0.4:
                color = "#ffc107"  # Yellow  
                icon = "🟡"
            else:
                color = "#dc3545"  # Red
                icon = "🔴"
            
            st.markdown(f"""
            <div style="background: {color}20; padding: 1rem; border-radius: 8px; border-left: 4px solid {color};">
            <h4 style="color: {color}; margin: 0;">{icon} {single_event_probability*100:.1f}% Probability</h4>
            <p style="margin: 0.5rem 0 0 0;">Associations likely originate from a single astronomical source or event</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🌌 Multiple Events/Sources")
            if multi_event_probability > 0.7:
                color = "#28a745"  # Green
                icon = "🟢"
            elif multi_event_probability > 0.4:
                color = "#ffc107"  # Yellow
                icon = "🟡"  
            else:
                color = "#dc3545"  # Red
                icon = "🔴"
                
            st.markdown(f"""
            <div style="background: {color}20; padding: 1rem; border-radius: 8px; border-left: 4px solid {color};">
            <h4 style="color: {color}; margin: 0;">{icon} {multi_event_probability*100:.1f}% Probability</h4>
            <p style="margin: 0.5rem 0 0 0;">Associations likely from multiple independent sources or background</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Analysis breakdown
        st.markdown("### 📊 Analysis Breakdown")
        
        breakdown_col1, breakdown_col2 = st.columns(2)
        
        with breakdown_col1:
            st.markdown("**Contributing Factors:**")
            for factor in factor_names:
                st.write(f"• {factor}")
            st.write(f"• Confidence Weight: {confidence_weight:.2f}")
            st.write(f"• Association Weight: {association_weight:.2f}")
        
        with breakdown_col2:
            st.markdown("**Interpretation:**")
            if single_event_probability > 0.7:
                st.success("🎯 **Strong evidence** for single event origin - associations show tight temporal/spatial clustering")
            elif single_event_probability > 0.4:
                st.warning("⚠️ **Moderate evidence** for single event - some clustering observed but not definitive")
            else:
                st.info("📊 **Weak evidence** for single event - associations appear more random or from multiple sources")
                
        # Technical details in expander
        with st.expander("🔬 Technical Details"):
            st.write(f"**High Confidence Events:** {len(high_conf_events)}")
            st.write(f"**Average Confidence:** {confidence_weight:.3f}")
            if time_clustering > 0:
                st.write(f"**Time Standard Deviation:** {high_conf_events['dt'].std():.2f} seconds")
            if spatial_clustering > 0:
                st.write(f"**Angular Standard Deviation:** {high_conf_events['dtheta'].std():.2f} degrees")
            if strength_correlation > 0:
                st.write(f"**Strength Ratio Variance:** {high_conf_events['strength_ratio'].var():.3f}")
            
    else:
        st.info("No high-confidence associations found for event classification analysis.")

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1f4e79, #2d5aa0);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    margin-bottom: 2rem;
}
.metric-container {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #007acc;
}
.data-input-section {
    background: #ffffff;
    padding: 1.5rem;
    border-radius: 10px;
    border: 1px solid #e0e0e0;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
<h1>🌟 Enhanced Multimessenger Astronomy AI Platform</h1>
<p>Advanced real-time detection and analysis of cosmic multimessenger events with comprehensive data input options and rich visualizations</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.markdown("## 🔧 Analysis Configuration")
# Use absolute path for model files
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, 'saved_models')
model_files = list_model_files(models_dir)
st.sidebar.write(f"🔍 Found {len(model_files)} model files: {model_files}")  # Debug info
model_choice = st.sidebar.selectbox('🤖 ML Model', options=model_files if model_files else ['(none)'], 
                                   help="Select the trained model for multimessenger association analysis")
threshold = st.sidebar.slider('🎯 Confidence Threshold', 0.0, 1.0, 0.5, 0.01,
                             help="Minimum probability to classify as a true multimessenger association")

# Advanced settings
with st.sidebar.expander("⚙️ Advanced Settings"):
    show_warnings = st.checkbox("Show ML warnings", value=False)
    max_events = st.number_input("Max events to process", min_value=10, max_value=10000, value=1000)
    enable_realtime = st.checkbox("Enable real-time mode", value=False)

# Model loading
model = None
scaler = None
metadata = None
if model_choice and model_choice != '(none)':
    try:
        model, scaler, metadata = load_model_by_name(model_choice, models_dir)
        st.sidebar.success(f'✅ Model loaded: {model_choice}')
        if metadata:
            st.sidebar.info(f"🎯 Model: {metadata.get('best_model', 'Unknown')}\n📊 AUC: {metadata.get('best_auc', 'N/A'):.3f}")
    except Exception as e:
        st.sidebar.error(f'❌ Load failed: {e}')

# Main content area
tab1, tab2, tab3 = st.tabs(["📊 Data Input & Analysis", "📈 Advanced Visualizations", "⚙️ Data Management"])

with tab1:
    st.markdown("## 📊 Multiple Data Input Methods")
    
    # Data input selection
    input_method = st.radio(
        "Choose your data input method:",
        ["🔄 Real-time Simulation", "📁 File Upload", "✋ Manual Entry", "🧪 Generate Samples", "🌐 API Integration"],
        horizontal=True
    )
    
    df = None
    
    if input_method == "🔄 Real-time Simulation":
        st.markdown("### 🔄 Real-time Multimessenger Event Simulation")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            n_events = st.number_input("Number of events", min_value=10, max_value=1000, value=100)
        with col2:
            time_window = st.number_input("Time window (hours)", min_value=1, max_value=168, value=24)
        with col3:
            noise_level = st.selectbox("Background noise", ["Low", "Medium", "High"])
        
        if st.button("🚀 Start Real-time Simulation", type="primary"):
            with st.spinner("Generating real-time multimessenger events..."):
                rng = np.random.RandomState(int(datetime.now().timestamp()) % 1000)
                
                # Simulate realistic detection timing
                base_time = datetime.now()
                times = []
                for i in range(n_events):
                    time_offset = rng.exponential(time_window * 3600 / n_events)
                    times.append(base_time + timedelta(seconds=time_offset))
                
                # Generate messenger combinations
                messengers = ['GW', 'Gamma', 'Neutrino', 'Optical', 'Radio']
                pairs = [(m1, m2) for i, m1 in enumerate(messengers) for m2 in messengers[i+1:]]
                
                data = []
                for i in range(n_events):
                    pair = pairs[rng.randint(0, len(pairs))]
                    
                    # Realistic parameters based on noise level
                    noise_factor = {"Low": 0.5, "Medium": 1.0, "High": 2.0}[noise_level]
                    
                    data.append({
                        'm1': pair[0],
                        'm2': pair[1],
                        'dt': rng.normal(0, 30 * noise_factor),
                        'dtheta': abs(rng.normal(2, 3 * noise_factor)),
                        'strength_ratio': abs(rng.lognormal(0, 0.5 * noise_factor)),
                        'ra': rng.uniform(0, 360),
                        'dec': rng.uniform(-90, 90),
                        'timestamp': times[i],
                        'detector_snr': abs(rng.normal(10, 3)),
                        'significance': abs(rng.normal(5, 2))
                    })
                
                df = pd.DataFrame(data)
                st.session_state['current_data'] = df
                st.success(f"✅ Generated {len(df)} real-time events")
    
    elif input_method == "📁 File Upload":
        st.markdown("### 📁 File Upload Options")
        
        upload_type = st.selectbox("File format:", ["CSV", "JSON", "Excel", "HDF5", "Parquet"])
        
        if upload_type == "CSV":
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state['current_data'] = df
                    st.success(f"✅ Loaded {len(df)} records from CSV")
                except Exception as e:
                    st.error(f"❌ Error reading CSV: {e}")
        
        elif upload_type == "JSON":
            uploaded_file = st.file_uploader("Upload JSON file", type=['json'])
            if uploaded_file:
                try:
                    data = json.load(uploaded_file)
                    df = pd.json_normalize(data)
                    st.session_state['current_data'] = df
                    st.success(f"✅ Loaded {len(df)} records from JSON")
                except Exception as e:
                    st.error(f"❌ Error reading JSON: {e}")
        
        elif upload_type == "Excel":
            uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx', 'xls'])
            if uploaded_file:
                try:
                    sheet_name = st.selectbox("Select sheet:", pd.ExcelFile(uploaded_file).sheet_names)
                    df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                    st.session_state['current_data'] = df
                    st.success(f"✅ Loaded {len(df)} records from Excel")
                except Exception as e:
                    st.error(f"❌ Error reading Excel: {e}")
    
    elif input_method == "✋ Manual Entry":
        st.markdown("### ✋ Manual Data Entry")
        
        with st.form("manual_entry"):
            st.markdown("**Enter detection pair information:**")
            
            col1, col2 = st.columns(2)
            with col1:
                m1 = st.selectbox("First messenger:", ['GW', 'Gamma', 'Neutrino', 'Optical', 'Radio'])
                dt = st.number_input("Time difference (seconds):", value=0.0, format="%.3f")
                strength_ratio = st.number_input("Strength ratio:", value=1.0, min_value=0.01, format="%.3f")
                ra = st.number_input("Right Ascension (deg):", value=0.0, min_value=0.0, max_value=360.0)
            
            with col2:
                m2 = st.selectbox("Second messenger:", ['GW', 'Gamma', 'Neutrino', 'Optical', 'Radio'])
                dtheta = st.number_input("Angular separation (deg):", value=0.0, min_value=0.0, format="%.3f")
                snr = st.number_input("Signal-to-noise ratio:", value=10.0, min_value=1.0)
                dec = st.number_input("Declination (deg):", value=0.0, min_value=-90.0, max_value=90.0)
            
            if st.form_submit_button("➕ Add Entry"):
                new_entry = pd.DataFrame([{
                    'm1': m1, 'm2': m2, 'dt': dt, 'dtheta': dtheta,
                    'strength_ratio': strength_ratio, 'ra': ra, 'dec': dec,
                    'snr': snr, 'timestamp': datetime.now()
                }])
                
                if 'manual_data' not in st.session_state:
                    st.session_state['manual_data'] = new_entry
                else:
                    st.session_state['manual_data'] = pd.concat([st.session_state['manual_data'], new_entry], ignore_index=True)
                
                st.success("✅ Entry added successfully!")
        
        if 'manual_data' in st.session_state:
            st.markdown("**Current manual entries:**")
            st.dataframe(st.session_state['manual_data'], width=800)
            
            if st.button("📊 Use Manual Data for Analysis"):
                st.session_state['current_data'] = st.session_state['manual_data']
                df = st.session_state['current_data']
    
    elif input_method == "🧪 Generate Samples":
        st.markdown("### 🧪 Sample Data Generation")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            sample_type = st.selectbox("Sample type:", ["Balanced", "High Background", "Burst Events", "Continuous Monitoring"])
        with col2:
            n_samples = st.number_input("Number of samples:", min_value=10, max_value=5000, value=500)
        with col3:
            random_seed = st.number_input("Random seed:", value=42)
        
        if st.button("🧪 Generate Sample Data", type="primary"):
            rng = np.random.RandomState(random_seed)
            
            if sample_type == "Balanced":
                # Equal mix of all messenger pairs
                messengers = ['GW', 'Gamma', 'Neutrino', 'Optical', 'Radio']
                pairs = [(m1, m2) for i, m1 in enumerate(messengers) for m2 in messengers[i+1:]]
                pair_choices = [pairs[i % len(pairs)] for i in range(n_samples)]
                
            elif sample_type == "High Background":
                # More random, uncorrelated events
                messengers = ['GW', 'Gamma', 'Neutrino', 'Optical', 'Radio']
                pair_choices = [(rng.choice(messengers), rng.choice(messengers)) for _ in range(n_samples)]
                
            elif sample_type == "Burst Events":
                # Clustered in time, focused on GW-related pairs
                gw_pairs = [('GW', 'Gamma'), ('GW', 'Optical'), ('GW', 'Neutrino')]
                pair_choices = [gw_pairs[rng.randint(0, len(gw_pairs))] for _ in range(n_samples)]
                
            else:  # Continuous Monitoring
                # Realistic distribution based on detection rates
                weighted_pairs = [('Gamma', 'Optical')] * 30 + [('GW', 'Gamma')] * 5 + [('Neutrino', 'Optical')] * 10
                pair_choices = [weighted_pairs[rng.randint(0, len(weighted_pairs))] for _ in range(n_samples)]
            
            m1_list = [p[0] for p in pair_choices]
            m2_list = [p[1] for p in pair_choices]
            
            df = pd.DataFrame({
                'm1': m1_list,
                'm2': m2_list,
                'dt': rng.normal(0, 50, n_samples),
                'dtheta': np.abs(rng.gamma(2, 2, n_samples)),
                'strength_ratio': np.abs(rng.lognormal(0, 0.8, n_samples)),
                'ra': rng.uniform(0, 360, n_samples),
                'dec': rng.uniform(-90, 90, n_samples),
                'snr': np.abs(rng.normal(10, 3, n_samples)),
                'significance': np.abs(rng.exponential(3, n_samples))
            })
            
            st.session_state['current_data'] = df
            st.success(f"✅ Generated {len(df)} {sample_type.lower()} events")
    
    elif input_method == "🌐 API Integration":
        st.markdown("### 🌐 API Data Integration")
        
        api_source = st.selectbox("Data source:", ["LIGO/Virgo GCN", "Fermi GBM", "IceCube Alerts", "ASASSN", "Custom API"])
        
        if api_source == "Custom API":
            api_url = st.text_input("API URL:", placeholder="https://api.example.com/data")
            api_key = st.text_input("API Key:", type="password")
            
            if st.button("🔄 Fetch Data") and api_url:
                st.info("🚧 Custom API integration coming soon!")
        else:
            st.info(f"🚧 {api_source} integration coming soon!")
    
    # Display current data if available
    if 'current_data' in st.session_state:
        df = st.session_state['current_data']
        
        # Data overview
        st.markdown("## 📋 Data Overview")
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
        with st.expander("👀 Preview Data", expanded=False):
            st.dataframe(df.head(20), width=1000)
        
        # Column validation
        required_cols = ['dt', 'dtheta', 'strength_ratio']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.warning(f"⚠️ Missing required columns: {missing_cols}")
            
            st.markdown("**Column Mapping:**")
            mapping = {}
            cols = st.columns(len(missing_cols))
            for i, missing_col in enumerate(missing_cols):
                with cols[i]:
                    mapping[missing_col] = st.selectbox(f"Map to {missing_col}:", df.columns)
            
            if st.button("🔄 Apply Column Mapping"):
                for missing_col, source_col in mapping.items():
                    df[missing_col] = df[source_col]
                st.session_state['current_data'] = df
                st.success("✅ Column mapping applied!")
                st.experimental_rerun()
        
        # Analysis button
        if not missing_cols:
            st.markdown("## 🚀 Run Analysis")
            
            if st.button('🔍 **Analyze Multimessenger Associations**', type="primary", use_container_width=True):
                if model is None:
                    st.error('❌ Please select a trained model first')
                else:
                    analyze_data(df, model, scaler, threshold, max_events)

# Visualization tab
with tab2:
    st.markdown("## 📈 Advanced Visualizations")
    
    if 'analysis_results' in st.session_state:
        df_results = st.session_state['analysis_results']
        
        # Visualization options
        viz_type = st.selectbox(
            "Choose visualization:",
            ["📊 Confidence Distribution", "🌌 Sky Map", "⏱️ Time Analysis", "🔥 Heatmaps", "📈 Correlations", "🎯 3D Analysis"]
        )
        
        if viz_type == "📊 Confidence Distribution":
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig1 = px.histogram(df_results, x='pred_prob', nbins=30, 
                                  title='Confidence Score Distribution')
                fig1.add_vline(x=0.5, line_dash="dash", line_color="red")
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Box plot by messenger pairs
                if 'm1' in df_results.columns:
                    df_results['pair'] = df_results['m1'] + '-' + df_results['m2']
                    fig2 = px.box(df_results, x='pair', y='pred_prob', 
                                title='Confidence by Messenger Pair')
                    fig2.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig2, use_container_width=True)
        
        elif viz_type == "🌌 Sky Map":
            if 'ra' in df_results.columns and 'dec' in df_results.columns:
                # 2D sky map
                fig = px.scatter(df_results, x='ra', y='dec', color='pred_prob',
                               size='strength_ratio', hover_data=['dt', 'dtheta'],
                               title='Multimessenger Events Sky Distribution',
                               color_continuous_scale='Viridis')
                fig.update_layout(xaxis_title="Right Ascension (°)", yaxis_title="Declination (°)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Mollweide projection (simplified)
                fig2 = go.Figure(data=go.Scattergeo(
                    lon=df_results['ra'],
                    lat=df_results['dec'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=df_results['pred_prob'],
                        colorscale='Viridis',
                        showscale=True
                    ),
                    text=df_results['pred_prob'].round(3)
                ))
                fig2.update_layout(title="Sky Map (Geographic Projection)")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Sky coordinates (RA/Dec) not available")
        
        elif viz_type == "⏱️ Time Analysis":
            col1, col2 = st.columns(2)
            
            with col1:
                # Time difference vs confidence
                fig1 = px.scatter(df_results, x='dt', y='pred_prob', color='dtheta',
                                title='Time Difference vs Confidence')
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Temporal clustering
                if 'timestamp' in df_results.columns:
                    df_results['hour'] = pd.to_datetime(df_results['timestamp']).dt.hour
                    hourly_counts = df_results.groupby('hour').size()
                    fig2 = px.bar(x=hourly_counts.index, y=hourly_counts.values,
                                title='Event Distribution by Hour')
                    st.plotly_chart(fig2, use_container_width=True)
        
        elif viz_type == "🔥 Heatmaps":
            col1, col2 = st.columns(2)
            
            with col1:
                # Correlation heatmap
                numeric_cols = df_results.select_dtypes(include=[np.number]).columns
                corr_matrix = df_results[numeric_cols].corr()
                
                fig1 = px.imshow(corr_matrix, 
                               title='Feature Correlation Matrix',
                               color_continuous_scale='RdBu')
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # 2D histogram
                fig2 = px.density_heatmap(df_results, x='dt', y='dtheta',
                                        title='Time vs Angular Separation Density')
                st.plotly_chart(fig2, use_container_width=True)
        
        elif viz_type == "📈 Correlations":
            # Scatter matrix
            numeric_cols = ['dt', 'dtheta', 'strength_ratio', 'pred_prob']
            available_cols = [col for col in numeric_cols if col in df_results.columns]
            
            if len(available_cols) >= 2:
                fig = px.scatter_matrix(df_results[available_cols], 
                                      title='Feature Scatter Matrix')
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "🎯 3D Analysis":
            if all(col in df_results.columns for col in ['dt', 'dtheta', 'strength_ratio']):
                fig = px.scatter_3d(df_results, x='dt', y='dtheta', z='strength_ratio',
                                  color='pred_prob', size='pred_prob',
                                  title='3D Feature Space Analysis')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Required 3D features not available")
    
    else:
        st.info("Run analysis first to see visualizations")

# Data Management tab
with tab3:
    st.markdown("## ⚙️ Data Management & Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Current Session Data")
        if 'current_data' in st.session_state:
            df = st.session_state['current_data']
            st.info(f"📋 {len(df)} events loaded")
            
            # Data quality checks
            st.markdown("**Data Quality:**")
            completeness = (1 - df.isnull().sum() / len(df)) * 100
            for col in df.columns:
                st.progress(completeness[col] / 100, text=f"{col}: {completeness[col]:.1f}% complete")
        
        # Clear data button
        if st.button("🗑️ Clear All Data"):
            for key in ['current_data', 'analysis_results', 'manual_data']:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("✅ All data cleared")
            st.experimental_rerun()
    
    with col2:
        st.markdown("### 📤 Export Options")
        
        if 'analysis_results' in st.session_state:
            df_results = st.session_state['analysis_results']
            
            export_format = st.selectbox("Export format:", ["CSV", "JSON", "Excel", "HDF5"])
            include_metadata = st.checkbox("Include analysis metadata", value=True)
            
            if st.button("📥 Export Results"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                if export_format == "CSV":
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f'multimessenger_results_{timestamp}.csv',
                        mime='text/csv'
                    )
                
                elif export_format == "JSON":
                    json_data = df_results.to_json(orient='records', indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name=f'multimessenger_results_{timestamp}.json',
                        mime='application/json'
                    )
        else:
            st.info("No analysis results to export")

# Footer
st.markdown("---")
st.markdown("🌟 **Enhanced Multimessenger Astronomy AI Platform** - Discovering the universe through AI")
