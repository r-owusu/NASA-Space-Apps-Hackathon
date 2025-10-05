import streamlit as st
import pandas as pd
import numpy as np
from model_loader import load_model_by_name
from inference import predict_df

st.title("🌟 Multimessenger Analysis - Debug Version")

# Sidebar
model_files = ['best_model.pkl']
model_choice = st.sidebar.selectbox('Select Model', options=model_files)
threshold = st.sidebar.slider('Threshold', 0.0, 1.0, 0.5, 0.01)

# Load model
model = None
scaler = None
if model_choice:
    try:
        model, scaler, metadata = load_model_by_name(model_choice)
        st.sidebar.success(f'✅ Model loaded: {model_choice}')
    except Exception as e:
        st.sidebar.error(f'❌ Error: {e}')

# Create sample data button
if st.button('🧪 Generate Sample Data'):
    rng = np.random.RandomState(0)
    n_samples = 20
    messengers = ['GW', 'Gamma', 'Neutrino', 'Optical', 'Radio']
    pairs = [(m1, m2) for i, m1 in enumerate(messengers) for m2 in messengers[i+1:]]
    pair_choices = [pairs[i % len(pairs)] for i in range(n_samples)]
    m1_list = [p[0] for p in pair_choices]
    m2_list = [p[1] for p in pair_choices]
    
    df = pd.DataFrame({
        'm1': m1_list,
        'm2': m2_list,
        'dt': rng.normal(0, 50, n_samples),
        'dtheta': np.abs(rng.normal(5, 3, n_samples)),
        'strength_ratio': np.abs(rng.normal(1, 0.8, n_samples)),
        'ra': rng.uniform(0, 360, n_samples),
        'dec': rng.uniform(-90, 90, n_samples)
    })
    
    st.session_state['data'] = df
    st.success(f'✅ Generated {len(df)} sample detections')

# Display data if available
if 'data' in st.session_state:
    df = st.session_state['data']
    st.subheader('📊 Data Overview')
    st.write(f"**Shape:** {df.shape}")
    st.write(f"**Columns:** {list(df.columns)}")
    st.dataframe(df.head())
    
    # Analysis button
    if st.button('🔍 **RUN ANALYSIS**', type='primary'):
        if model is None:
            st.error('❌ No model loaded!')
        else:
            try:
                with st.spinner('🔬 Analyzing...'):
                    # Run the analysis
                    df_pred = predict_df(df, model, scaler=scaler, threshold=threshold)
                    
                    # Calculate metrics
                    total_events = len(df_pred)
                    high_confidence = len(df_pred[df_pred['pred_prob'] >= 0.8])
                    medium_confidence = len(df_pred[(df_pred['pred_prob'] >= 0.5) & (df_pred['pred_prob'] < 0.8)])
                    positive_associations = len(df_pred[df_pred['pred_label'] == 1])
                    avg_confidence = df_pred['pred_prob'].mean()
                    
                    # Display results
                    st.success('✅ Analysis Complete!')
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📊 Total Events", total_events)
                    with col2:
                        st.metric("🔴 High Confidence", high_confidence)
                    with col3:
                        st.metric("✅ Positive Associations", positive_associations)
                    with col4:
                        st.metric("📈 Avg Confidence", f"{avg_confidence:.3f}")
                    
                    # Show results table
                    st.subheader('📋 Results')
                    result_display = df_pred[['m1', 'm2', 'dt', 'dtheta', 'strength_ratio', 'pred_prob', 'pred_label']].copy()
                    result_display = result_display.sort_values('pred_prob', ascending=False)
                    st.dataframe(result_display)
                    
                    # Show confidence distribution
                    st.subheader('📊 Confidence Distribution')
                    st.bar_chart(df_pred['pred_prob'])
                    
            except Exception as e:
                st.error(f'❌ Analysis failed: {str(e)}')
                st.code(str(e))
                import traceback
                st.code(traceback.format_exc())
else:
    st.info('👆 Click "Generate Sample Data" to start')