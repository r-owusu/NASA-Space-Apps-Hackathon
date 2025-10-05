# Continuation of enhanced_app_ultimate.py - Part 2
# Visualizations Tab
with tab4:
    st.markdown("## 📈 Advanced Visualizations")
    
    if st.session_state.results is not None:
        results = st.session_state.results
        
        # Visualization controls
        st.markdown("### 🎨 Visualization Controls")
        
        col_viz1, col_viz2, col_viz3 = st.columns(3)
        
        with col_viz1:
            viz_type = st.selectbox("Visualization Type", [
                "Probability Distribution", "Sky Map", "Time Series", 
                "Correlation Matrix", "3D Scatter", "Event Timeline"
            ])
        
        with col_viz2:
            color_scheme = st.selectbox("Color Scheme", [
                "Viridis", "Plasma", "Cividis", "Turbo", "Rainbow"
            ])
        
        with col_viz3:
            show_confidence = st.checkbox("Show Confidence Bands", value=True)
        
        # Generate visualizations
        if viz_type == "Probability Distribution":
            st.markdown("### 📊 Probability Distribution Analysis")
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Probability Histogram', 'Classification by Event Type', 
                               'Confidence Levels', 'Probability vs Features'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Probability histogram
            fig.add_trace(
                go.Histogram(
                    x=results['probability'],
                    name='Probability',
                    opacity=0.7,
                    nbinsx=30
                ), row=1, col=1
            )
            
            # Classification by event type
            if 'pair' in results.columns:
                classification_counts = results.groupby(['pair', 'classification']).size().reset_index(name='count')
                
                fig.add_trace(
                    go.Bar(
                        x=classification_counts['pair'],
                        y=classification_counts['count'],
                        name='Classifications',
                        text=classification_counts['classification'],
                        textposition='auto'
                    ), row=1, col=2
                )
            
            # Confidence levels
            confidence_counts = results['confidence'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=confidence_counts.index,
                    values=confidence_counts.values,
                    name="Confidence"
                ), row=2, col=1
            )
            
            # Probability vs features scatter
            if 'dt' in results.columns and 'dtheta' in results.columns:
                fig.add_trace(
                    go.Scatter(
                        x=results['dt'],
                        y=results['probability'],
                        mode='markers',
                        marker=dict(
                            color=results['dtheta'],
                            colorscale=color_scheme,
                            size=8,
                            colorbar=dict(title="dθ")
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
            
            if 'ra' in results.columns and 'dec' in results.columns:
                # Convert to galactic coordinates for better visualization
                coords = SkyCoord(ra=results['ra']*u.degree, dec=results['dec']*u.degree, frame='icrs')
                galactic = coords.galactic
                
                fig = go.Figure()
                
                # Add events colored by classification
                for classification in results['classification'].unique():
                    mask = results['classification'] == classification
                    
                    fig.add_trace(
                        go.Scatterpolar(
                            r=90 - results.loc[mask, 'dec'],  # Convert dec to polar radius
                            theta=results.loc[mask, 'ra'],
                            mode='markers',
                            marker=dict(
                                size=results.loc[mask, 'probability'] * 20 + 5,
                                color=results.loc[mask, 'probability'],
                                colorscale=color_scheme,
                                colorbar=dict(title="Probability"),
                                line=dict(width=1, color='white')
                            ),
                            name=classification,
                            text=[f"Event: {results.loc[i, 'event_id'] if 'event_id' in results.columns else i}<br>"
                                 f"Prob: {results.loc[i, 'probability']:.3f}<br>"
                                 f"RA: {results.loc[i, 'ra']:.2f}°<br>"
                                 f"Dec: {results.loc[i, 'dec']:.2f}°" 
                                 for i in results[mask].index],
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
                    title="Sky Map of Detected Events",
                    height=600,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Sky coordinates (RA/Dec) not available in dataset")
        
        elif viz_type == "Time Series":
            st.markdown("### ⏰ Time Series Analysis")
            
            if 'timestamp' in results.columns:
                # Sort by timestamp
                time_sorted = results.sort_values('timestamp')
                
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=('Event Detection Rate', 'Probability Over Time', 'Cumulative Classifications'),
                    vertical_spacing=0.08
                )
                
                # Event detection rate (events per hour)
                time_sorted['hour'] = time_sorted['timestamp'].dt.floor('H')
                event_rate = time_sorted.groupby('hour').size().reset_index(name='events_per_hour')
                
                fig.add_trace(
                    go.Scatter(
                        x=event_rate['hour'],
                        y=event_rate['events_per_hour'],
                        mode='lines+markers',
                        name='Detection Rate',
                        line=dict(width=3)
                    ), row=1, col=1
                )
                
                # Probability over time
                fig.add_trace(
                    go.Scatter(
                        x=time_sorted['timestamp'],
                        y=time_sorted['probability'],
                        mode='markers',
                        marker=dict(
                            color=time_sorted['probability'],
                            colorscale=color_scheme,
                            size=8
                        ),
                        name='Probability'
                    ), row=2, col=1
                )
                
                # Cumulative classifications
                time_sorted['same_cumsum'] = (time_sorted['classification'] == 'Same Event').cumsum()
                time_sorted['diff_cumsum'] = (time_sorted['classification'] == 'Different Events').cumsum()
                
                fig.add_trace(
                    go.Scatter(
                        x=time_sorted['timestamp'],
                        y=time_sorted['same_cumsum'],
                        mode='lines',
                        name='Cumulative Same Events',
                        line=dict(color='green', width=3)
                    ), row=3, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=time_sorted['timestamp'],
                        y=time_sorted['diff_cumsum'],
                        mode='lines',
                        name='Cumulative Different Events',
                        line=dict(color='red', width=3)
                    ), row=3, col=1
                )
                
                fig.update_layout(
                    height=900,
                    title_text="Time Series Analysis",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Timestamp information not available in dataset")
        
        elif viz_type == "Correlation Matrix":
            st.markdown("### 🔗 Feature Correlation Analysis")
            
            # Select numeric columns
            numeric_cols = results.select_dtypes(include=[np.number]).columns
            correlation_cols = [col for col in numeric_cols if col in ['dt', 'dtheta', 'strength_ratio', 'probability', 'log_dt', 'log_dtheta', 'log_strength_ratio']]
            
            if len(correlation_cols) > 1:
                corr_matrix = results[correlation_cols].corr()
                
                fig = go.Figure(
                    data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale=color_scheme,
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
            else:
                st.warning("Insufficient numeric features for correlation analysis")
        
        elif viz_type == "3D Scatter":
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
                            colorscale=color_scheme,
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
                        zaxis_title='Strength Ratio',
                        bgcolor='rgba(0,0,0,0)'
                    ),
                    title="3D Feature Space Visualization",
                    height=700,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Required features (dt, dtheta, strength_ratio) not available")
    
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
            data_stream.start_stream()
            st.success("Live stream started!")
    
    with col_rt2:
        if st.button("🔴 Stop Stream", type="secondary"):
            st.session_state.live_stream = False
            data_stream.stop_stream()
            st.info("Live stream stopped")
    
    with col_rt3:
        auto_refresh = st.checkbox("Auto Refresh", value=True)
    
    # Stream status
    stream_status = "🟢 LIVE" if st.session_state.live_stream else "🔴 OFFLINE"
    st.markdown(f"### Stream Status: {stream_status}")
    
    # Real-time data display
    if st.session_state.live_stream:
        # Generate new real-time event
        new_event = data_stream.generate_realtime_event()
        if new_event:
            st.session_state.real_time_data.append(new_event)
        
        # Display recent events
        if st.session_state.real_time_data:
            st.markdown("### 📡 Live Event Feed")
            
            # Create DataFrame from recent events
            rt_df = pd.DataFrame(st.session_state.real_time_data[-20:])  # Last 20 events
            
            # Real-time metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            with col_m1:
                st.metric("Events Today", len(st.session_state.real_time_data))
            
            with col_m2:
                gw_events = len([e for e in st.session_state.real_time_data if e['event_type'] == 'GW'])
                st.metric("GW Events", gw_events)
            
            with col_m3:
                high_conf = len([e for e in st.session_state.real_time_data if e['confidence'] > 0.8])
                st.metric("High Confidence", high_conf)
            
            with col_m4:
                avg_snr = np.mean([e['snr'] for e in st.session_state.real_time_data])
                st.metric("Avg SNR", f"{avg_snr:.1f}")
            
            # Live event table
            st.markdown("#### Recent Events")
            
            # Format for display
            display_df = rt_df.copy()
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%H:%M:%S')
            display_df = display_df[['timestamp', 'detector', 'event_type', 'confidence', 'snr']]
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=300
            )
            
            # Real-time sky plot
            if len(rt_df) > 0:
                st.markdown("#### Live Sky Map")
                
                fig = go.Figure()
                
                for event_type in rt_df['event_type'].unique():
                    mask = rt_df['event_type'] == event_type
                    
                    fig.add_trace(
                        go.Scatterpolar(
                            r=90 - rt_df.loc[mask, 'dec'],
                            theta=rt_df.loc[mask, 'ra'],
                            mode='markers',
                            marker=dict(
                                size=rt_df.loc[mask, 'snr'],
                                color=rt_df.loc[mask, 'confidence'],
                                colorscale='Viridis',
                                colorbar=dict(title="Confidence"),
                                line=dict(width=1, color='white')
                            ),
                            name=event_type
                        )
                    )
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 90]),
                        angularaxis=dict(direction="clockwise", period=360)
                    ),
                    title="Real-time Event Sky Map",
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Auto-refresh
        if auto_refresh:
            time.sleep(2)
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
                st.metric("Total Events", len(hist_df))
            
            with col_h2:
                event_types = hist_df['event_type'].value_counts()
                st.write("**Event Types:**")
                for event_type, count in event_types.items():
                    st.write(f"- {event_type}: {count}")
            
            with col_h3:
                detector_stats = hist_df['detector'].value_counts()
                st.write("**Detectors:**")
                for detector, count in detector_stats.items():
                    st.write(f"- {detector}: {count}")

# APIs Tab
with tab6:
    st.markdown("## ⚙️ API Integrations")
    
    # API status overview
    st.markdown("### 🔌 API Status Overview")
    
    api_services = {
        'GraceDB': {'url': 'https://gracedb.ligo.org', 'description': 'Gravitational Wave Event Database'},
        'LIGO Open Science': {'url': 'https://losc.ligo.org', 'description': 'LIGO Strain Data'},
        'Fermi LAT': {'url': 'https://fermi.gsfc.nasa.gov', 'description': 'Gamma-ray Observatory Data'},
        'IceCube': {'url': 'https://icecube.wisc.edu', 'description': 'Neutrino Observatory Data'}
    }
    
    for service_name, info in api_services.items():
        col_api1, col_api2, col_api3 = st.columns([1, 2, 1])
        
        with col_api1:
            # Check API status
            status = api_manager.check_api_status(service_name)
            status_color = "🟢" if status == 'online' else "🔴"
            st.markdown(f"**{status_color} {service_name}**")
        
        with col_api2:
            st.markdown(f"*{info['description']}*")
            st.markdown(f"URL: `{info['url']}`")
        
        with col_api3:
            if st.button(f"Test {service_name}", key=f"test_{service_name}"):
                with st.spinner(f"Testing {service_name}..."):
                    status = api_manager.check_api_status(service_name)
                    if status == 'online':
                        st.success("✅ Connected")
                    else:
                        st.error("❌ Connection failed")
    
    # Data fetching
    st.markdown("### 📡 Fetch External Data")
    
    col_fetch1, col_fetch2 = st.columns(2)
    
    with col_fetch1:
        st.markdown("#### GraceDB Events")
        
        gw_limit = st.number_input("Number of events:", 1, 50, 10, key="gw_limit")
        
        if st.button("🌊 Fetch GW Events", type="primary"):
            with st.spinner("Fetching gravitational wave events..."):
                gw_events = api_manager.fetch_gracedb_events(gw_limit)
                
                if gw_events:
                    st.session_state.api_data['gracedb'] = gw_events
                    
                    # Display events
                    gw_df = pd.DataFrame(gw_events)
                    st.dataframe(gw_df.head(), use_container_width=True)
                    
                    st.success(f"✅ Fetched {len(gw_events)} GW events")
                else:
                    st.error("❌ Failed to fetch GW events")
    
    with col_fetch2:
        st.markdown("#### Fermi Gamma-ray Events")
        
        gamma_limit = st.number_input("Number of events:", 1, 50, 10, key="gamma_limit")
        
        if st.button("🌟 Fetch Gamma Events", type="primary"):
            with st.spinner("Fetching gamma-ray events..."):
                gamma_events = api_manager.fetch_fermi_events(gamma_limit)
                
                if gamma_events:
                    st.session_state.api_data['fermi'] = gamma_events
                    
                    # Display events
                    gamma_df = pd.DataFrame(gamma_events)
                    st.dataframe(gamma_df.head(), use_container_width=True)
                    
                    st.success(f"✅ Fetched {len(gamma_events)} gamma-ray events")
                else:
                    st.error("❌ Failed to fetch gamma-ray events")
    
    # API data integration
    if st.session_state.api_data:
        st.markdown("### 🔀 Integrate API Data")
        
        available_sources = list(st.session_state.api_data.keys())
        selected_sources = st.multiselect("Select data sources to integrate:", available_sources)
        
        if selected_sources and st.button("🔄 Integrate Selected Data", type="primary"):
            integrated_data = []
            
            for source in selected_sources:
                source_data = st.session_state.api_data[source]
                
                # Convert to standard format
                for event in source_data:
                    if source == 'gracedb':
                        std_event = {
                            'event_id': event['event_id'],
                            'ra': event['ra'],
                            'dec': event['dec'],
                            'timestamp': datetime.fromtimestamp(event['gpstime']),
                            'source': 'GraceDB',
                            'event_type': 'GW',
                            'confidence': 1.0 - event['far']  # Convert FAR to confidence
                        }
                    elif source == 'fermi':
                        std_event = {
                            'event_id': event['event_id'],
                            'ra': event['ra'],
                            'dec': event['dec'],
                            'timestamp': event['trigger_time'],
                            'source': 'Fermi',
                            'event_type': 'Gamma',
                            'confidence': min(event['significance'] / 20.0, 1.0)
                        }
                    
                    integrated_data.append(std_event)
            
            if integrated_data:
                integrated_df = pd.DataFrame(integrated_data)
                st.session_state.data = integrated_df
                
                st.success(f"✅ Integrated {len(integrated_data)} events from {len(selected_sources)} sources")
                
                # Show integrated data
                st.markdown("#### Integrated Dataset")
                st.dataframe(integrated_df, use_container_width=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem;">
    <div class="feature-card">
        <h3 style="color: white; margin: 0;">🌌 Ultimate Multimessenger AI Platform</h3>
        <p style="color: rgba(255, 255, 255, 0.8); margin: 0.5rem 0 0 0;">
            Advanced AI • Real-time Data • API Integration • Enhanced Visualizations
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Auto-refresh for real-time data
if st.session_state.live_stream and st.session_state.selected_tab == 'Real-time':
    time.sleep(3)
    st.rerun()