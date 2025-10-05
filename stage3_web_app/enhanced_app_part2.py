# Continuation of enhanced_app_v2.py - Analysis and Visualization sections

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
            
            # Traditional sky map (Mollweide projection simulation)
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
            
            # Density map
            fig_density = ff.create_2d_density(
                df_pred['ra'], df_pred['dec'],
                title='Event Density Map',
                hist_color='rgba(31, 119, 180, 0.8)',
                point_color='rgba(31, 119, 180, 0.3)'
            )
            st.plotly_chart(fig_density, width="stretch")
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