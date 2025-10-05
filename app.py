"""
Streamlit app to simulate real-time IMU data streaming and stride event detection from parquet files.
Based on the streamlit-stream websocket example, but reading from local parquet files.
"""
import asyncio
import streamlit as st
import time

from src.imu_streaming import (
    StreamConfig,
    UIConfig,
    IMUDataLoader,
    IMUStreamProcessor,
    ChartRenderer,
    IMUStreamUI,
    GaitEventDetector,
    calculate_combined_metrics,
    display_overall_metrics,
    display_per_foot_metrics,
    display_empty_metrics,
    calculate_dynamic_x_range
)


# Constants
TOOLTIPS = {
    'cadence': "Steps per minute. Walking: 100-120, Running: 160-180. Higher cadence often improves efficiency and reduces injury risk.",
    'stride_variability': "Consistency of stride timing (CV%). <3% = Excellent, 3-5% = Acceptable, >5% = High variability may indicate fatigue or injury.",
    'stride_symmetry': "Left/right balance (%). <5% = Excellent, 5-10% = Good, >10% = Asymmetric (may indicate injury or compensation).",
    'contact_time': "Ground contact as % of stride. Walking ~60%, Running ~35%. Increasing values may indicate fatigue.",
    'stance_swing_ratio': "Time on ground vs. in air. Walking ~1.5, Running ~0.5-0.7. Lower ratio = faster pace.",
    'stride_time': "Time for one complete gait cycle. Consistency (low SD) is important for injury prevention.",
}

SELECTED_SENSORS = ["Gyro Y"]  # Hardcoded for now, could be made configurable

# Session state keys
SESSION_KEYS = {
    'streaming': 'streaming',
    'last_chart': 'last_chart',
    'last_metrics': 'last_metrics'
}


# Initialize configurations and components
stream_config = StreamConfig()
ui_config = UIConfig()

st.set_page_config(page_title="Movetru stride detection")
ui = IMUStreamUI(ui_config)
data_loader = IMUDataLoader(stream_config.DATA_DIR)
renderer = ChartRenderer(ui_config)

# === UI Setup ===
ui.render_header()

# Player selection
players = data_loader.get_available_players()
selected_player = ui.render_player_selector(players)
selected_sensors = SELECTED_SENSORS

# Stream controls
start_time, start_stream, stop_stream = ui.render_stream_controls()

# === Initialize Processing Components ===
processor = IMUStreamProcessor(stream_config)

# Gait event detector
gait_detector = GaitEventDetector(
    fs=stream_config.SAMPLING_RATE,
    msw_threshold=stream_config.MSW_THRESHOLD,
    zc_threshold=stream_config.ZC_THRESHOLD,
    ma_window=stream_config.FILTER_WINDOW_SIZE,
    max_buffer_size=stream_config.SAMPLING_RATE * 5,
    max_stride_time=stream_config.MAX_STRIDE_TIME,
    min_stride_time=stream_config.MIN_STRIDE_TIME,
)

# === UI Placeholders ===
status = ui.create_status_placeholder()
chart = ui.create_chart_placeholders(selected_sensors)

# Metrics section
st.markdown("---")
st.subheader("Metrics")

# Create tabs for different views
tab1, tab2 = st.tabs(["📈 Recent (Last 10s)", "📊 Overall Session"])

with tab1:
    # Overall metrics (combined)
    st.markdown("#### 🎯 Overall")
    col_recent_1, col_recent_2, col_recent_3 = st.columns(3)
    with col_recent_1:
        recent_total_strides = st.empty()
        recent_cadence = st.empty()
    with col_recent_2:
        recent_stride_variability = st.empty()
        recent_stride_symmetry = st.empty()
    with col_recent_3:
        recent_contact_time = st.empty()
        recent_stance_swing_ratio = st.empty()
    
    # Per-foot details
    st.markdown("#### 🦶 Per Foot")
    col_lf_recent, col_rf_recent = st.columns(2)
    with col_lf_recent:
        st.markdown("**Left Foot**")
        recent_lf_stride = st.empty()
        recent_lf_contact = st.empty()
    with col_rf_recent:
        st.markdown("**Right Foot**")
        recent_rf_stride = st.empty()
        recent_rf_contact = st.empty()

with tab2:
    # Overall metrics (combined)
    st.markdown("#### 🎯 Overall")
    col_overall_1, col_overall_2, col_overall_3 = st.columns(3)
    with col_overall_1:
        overall_total_strides = st.empty()
        overall_cadence = st.empty()
    with col_overall_2:
        overall_stride_variability = st.empty()
        overall_stride_symmetry = st.empty()
    with col_overall_3:
        overall_contact_time = st.empty()
        overall_stance_swing_ratio = st.empty()
    
    # Per-foot details
    st.markdown("#### 🦶 Per Foot")
    col_lf_overall, col_rf_overall = st.columns(2)
    with col_lf_overall:
        st.markdown("**Left Foot**")
        overall_lf_stride = st.empty()
        overall_lf_contact = st.empty()
    with col_rf_overall:
        st.markdown("**Right Foot**")
        overall_rf_stride = st.empty()
        overall_rf_contact = st.empty()

# === Session State Initialization ===
for key, default in [('streaming', False), ('last_chart', None), ('last_metrics', None)]:
    if key not in st.session_state:
        st.session_state[key] = default


# === Main Streaming Function ===
async def stream_imu_data(selected_player: str, sensors: list[str], start_from_time: float = 0.0) -> None:
    """
    Asynchronously stream IMU data from left and right foot parquet files.
    
    Loads player data, processes it in real-time with configurable speed, detects gait events,
    and updates the UI with charts and metrics at regular intervals.
    
    Args:
        selected_player: Player ID to stream data for
        sensors: List of sensor columns to stream (e.g., ["Gyro Y"])
        start_from_time: Time in seconds to start streaming from (default: 0.0)
    """
    # Load the data
    lf_path, rf_path = data_loader.get_file_paths(selected_player)
    status.info(f"Loading {lf_path.name} and {rf_path.name}...")
    df_lf, df_rf = data_loader.load_player_data(selected_player)
    
    # Convert start time to sample index
    start_from = data_loader.time_to_sample_index(df_lf, start_from_time)
    
    # Validate start_from
    valid, error_msg = data_loader.validate_start_position(df_lf, df_rf, start_from)
    if not valid:
        status.error(error_msg)
        return
    
    min_len = min(len(df_lf), len(df_rf))
    
    # Initialize processor windows
    processor.initialize_windows(stream_config.DEFAULT_WINDOW_SIZE, sensors)
    
    # Reset gait detector for new session
    gait_detector.reset()
    
    # Calculate y-axis ranges based on a larger sample to ensure they cover the data
    status.info("Calculating stable y-axis ranges...")
    sample_window_for_range = int(stream_config.SAMPLING_RATE * stream_config.RANGE_CALCULATION_WINDOW)
    range_window_end = min(start_from + sample_window_for_range, min_len)
    processor.calculate_y_ranges(df_lf, df_rf, sensors, start_from, range_window_end)
    
    # Display streaming info
    actual_start_time = df_lf['Time'][start_from]
    status.success(
        f"Streaming {min_len - start_from} samples (starting at {actual_start_time:.2f}s, sample {start_from})"
    )
    
    # Calculate window duration for x-axis range
    window_duration_seconds = stream_config.DEFAULT_WINDOW_SIZE / stream_config.SAMPLING_RATE
    
    # Stream the data
    last_update_time = time.time()
    sample_count = 0
    
    for row_idx in range(start_from, min_len):
        # Check if we should stop
        if not st.session_state.streaming:
            status.warning("Stream stopped by user")
            break
        
        row_lf = df_lf.row(row_idx, named=True)
        row_rf = df_rf.row(row_idx, named=True)
        current_time_lf = row_lf['Time']
        current_time_rf = row_rf['Time']
        
        # Update windows using processor
        processor.update_windows(row_lf, row_rf, sensors)
        
        # Process gait events
        events = gait_detector.process_sample(
            lf_gyro_y=row_lf['Gyro Y'],
            rf_gyro_y=row_rf['Gyro Y'],
            lf_time=current_time_lf,
            rf_time=current_time_rf
        )
        
        sample_count += 1
        
        # Only update charts every UPDATE_INTERVAL samples for better performance
        if sample_count >= stream_config.UPDATE_INTERVAL:
            
            # Get all detected events for display
            events_lf = gait_detector.get_all_events('left')
            events_rf = gait_detector.get_all_events('right')
            
            for sensor in sensors:
                # Get current data from processor
                data = processor.get_current_data(sensor)
                
                # Downsample for display
                times_lf_display, values_lf_display = renderer.downsample_data(
                    data['lf_times'], data['lf_values'], stream_config.DOWNSAMPLE_FACTOR
                )
                times_rf_display, values_rf_display = renderer.downsample_data(
                    data['rf_times'], data['rf_values'], stream_config.DOWNSAMPLE_FACTOR
                )
                
                # Update ranges dynamically if needed
                processor.update_y_ranges(sensor, data['lf_values'], data['rf_values'])
                
                # Use combined y-range (max of both ranges)
                y_range_lf = processor.y_ranges_lf[sensor]
                y_range_rf = processor.y_ranges_rf[sensor]
                y_range_combined = [
                    min(y_range_lf[0], y_range_rf[0]),
                    max(y_range_lf[1], y_range_rf[1])
                ]
                
                # Calculate dynamic x-axis range for scrolling window
                dynamic_x_range = calculate_dynamic_x_range(
                    data['lf_times'], data['rf_times'], 
                    window_duration_seconds, actual_start_time
                )
                
                # Create combined chart with dynamic x-axis range
                fig = renderer.create_combined_chart(
                    times_lf_display, values_lf_display,
                    times_rf_display, values_rf_display,
                    y_range_combined, sensor,
                    events_lf, events_rf,
                    x_range=dynamic_x_range
                )
                
                # Save to session state for freezing
                st.session_state.last_chart = fig
                
                # Update chart
                if chart:
                    chart.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            # Update and display gait metrics
            metrics_lf_recent = gait_detector.get_metrics('left', window_seconds=stream_config.METRICS_WINDOW)
            metrics_rf_recent = gait_detector.get_metrics('right', window_seconds=stream_config.METRICS_WINDOW)
            metrics_lf_overall = gait_detector.get_metrics('left')
            metrics_rf_overall = gait_detector.get_metrics('right')
            symmetry_recent = gait_detector.get_symmetry_metrics(window_seconds=stream_config.METRICS_WINDOW)
            symmetry_overall = gait_detector.get_symmetry_metrics()
            
            # Save to session state for freezing
            st.session_state.last_metrics = {
                'recent_lf': metrics_lf_recent,
                'recent_rf': metrics_rf_recent,
                'overall_lf': metrics_lf_overall,
                'overall_rf': metrics_rf_overall,
                'symmetry_recent': symmetry_recent,
                'symmetry_overall': symmetry_overall
            }
            
            # Calculate and display recent metrics
            combined_recent = calculate_combined_metrics(metrics_lf_recent, metrics_rf_recent, symmetry_recent)
            recent_placeholders = {
                'total_strides': recent_total_strides,
                'cadence': recent_cadence,
                'stride_variability': recent_stride_variability,
                'stride_symmetry': recent_stride_symmetry,
                'contact_time': recent_contact_time,
                'stance_swing_ratio': recent_stance_swing_ratio
            }
            display_overall_metrics(recent_placeholders, combined_recent, TOOLTIPS)
            
            # Display recent per-foot metrics
            display_per_foot_metrics(
                {'stride': recent_lf_stride, 'contact': recent_lf_contact},
                metrics_lf_recent, TOOLTIPS
            )
            display_per_foot_metrics(
                {'stride': recent_rf_stride, 'contact': recent_rf_contact},
                metrics_rf_recent, TOOLTIPS
            )
            
            # Calculate and display overall metrics
            combined_overall = calculate_combined_metrics(metrics_lf_overall, metrics_rf_overall, symmetry_overall)
            overall_placeholders = {
                'total_strides': overall_total_strides,
                'cadence': overall_cadence,
                'stride_variability': overall_stride_variability,
                'stride_symmetry': overall_stride_symmetry,
                'contact_time': overall_contact_time,
                'stance_swing_ratio': overall_stance_swing_ratio
            }
            display_overall_metrics(overall_placeholders, combined_overall, TOOLTIPS)
            
            # Display overall per-foot metrics
            display_per_foot_metrics(
                {'stride': overall_lf_stride, 'contact': overall_lf_contact},
                metrics_lf_overall, TOOLTIPS
            )
            display_per_foot_metrics(
                {'stride': overall_rf_stride, 'contact': overall_rf_contact},
                metrics_rf_overall, TOOLTIPS
            )
            
            # Update status with timing and event information
            elapsed_real_time = time.time() - last_update_time
            actual_speed = (
                (stream_config.UPDATE_INTERVAL / stream_config.SAMPLING_RATE) / elapsed_real_time 
                if elapsed_real_time > 0 else stream_config.DEFAULT_SPEED
            )
            
            all_events_lf = gait_detector.get_all_events('left')
            all_events_rf = gait_detector.get_all_events('right')
            status.info(
                f"Sample {row_idx + 1}/{min_len} | "
                f"LF Time: {current_time_lf:.2f}s | RF Time: {current_time_rf:.2f}s | "
                f"Target Speed: {stream_config.DEFAULT_SPEED}x | Actual: {actual_speed:.1f}x | "
                f"Events: L:{len(all_events_lf['msw'])} R:{len(all_events_rf['msw'])} MSW"
            )
            
            # Calculate sleep time accounting for rendering overhead
            target_update_time = (stream_config.UPDATE_INTERVAL / stream_config.SAMPLING_RATE / 
                                stream_config.DEFAULT_SPEED)
            sleep_time = max(0, target_update_time - elapsed_real_time)
            
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                # Give control back to event loop even if we're behind
                await asyncio.sleep(0)
            
            last_update_time = time.time()
            sample_count = 0
    
    status.success(f"Stream completed! Processed {min_len} samples")
    st.session_state.streaming = False


# Pre-populate UI with frozen/empty state before streaming starts
if not st.session_state.streaming and chart:
    if st.session_state.last_chart is not None:
        # Show frozen chart from last stream
        chart.plotly_chart(st.session_state.last_chart, use_container_width=True, 
                          config={'displayModeBar': False}, key='frozen_chart')
    else:
        # Show empty chart before first stream
        window_duration = stream_config.DEFAULT_WINDOW_SIZE / stream_config.SAMPLING_RATE
        empty_x_range = [start_time, start_time + window_duration]
        empty_fig = renderer.create_combined_chart(
            times_lf=[start_time], values_lf=[0],
            times_rf=[start_time], values_rf=[0],
            y_range=[-200, 200],
            sensor_name=selected_sensors[0],
            events_lf=None,
            events_rf=None,
            x_range=empty_x_range
        )
        chart.plotly_chart(empty_fig, use_container_width=True, 
                          config={'displayModeBar': False}, key='empty_chart')
    
    # Display frozen metrics from last stream
    if st.session_state.last_metrics is not None:
        last = st.session_state.last_metrics
        
        # Calculate and display recent metrics
        combined_recent = calculate_combined_metrics(last['recent_lf'], last['recent_rf'], last['symmetry_recent'])
        recent_placeholders = {
            'total_strides': recent_total_strides,
            'cadence': recent_cadence,
            'stride_variability': recent_stride_variability,
            'stride_symmetry': recent_stride_symmetry,
            'contact_time': recent_contact_time,
            'stance_swing_ratio': recent_stance_swing_ratio
        }
        display_overall_metrics(recent_placeholders, combined_recent, TOOLTIPS)
        display_per_foot_metrics({'stride': recent_lf_stride, 'contact': recent_lf_contact}, 
                                last['recent_lf'], TOOLTIPS)
        display_per_foot_metrics({'stride': recent_rf_stride, 'contact': recent_rf_contact}, 
                                last['recent_rf'], TOOLTIPS)
        
        # Calculate and display overall metrics
        combined_overall = calculate_combined_metrics(last['overall_lf'], last['overall_rf'], last['symmetry_overall'])
        overall_placeholders = {
            'total_strides': overall_total_strides,
            'cadence': overall_cadence,
            'stride_variability': overall_stride_variability,
            'stride_symmetry': overall_stride_symmetry,
            'contact_time': overall_contact_time,
            'stance_swing_ratio': overall_stance_swing_ratio
        }
        display_overall_metrics(overall_placeholders, combined_overall, TOOLTIPS)
        display_per_foot_metrics({'stride': overall_lf_stride, 'contact': overall_lf_contact}, 
                                last['overall_lf'], TOOLTIPS)
        display_per_foot_metrics({'stride': overall_rf_stride, 'contact': overall_rf_contact}, 
                                last['overall_rf'], TOOLTIPS)
    else:
        # Show empty metrics before first stream
        recent_placeholders = {
            'total_strides': recent_total_strides,
            'cadence': recent_cadence,
            'stride_variability': recent_stride_variability,
            'stride_symmetry': recent_stride_symmetry,
            'contact_time': recent_contact_time,
            'stance_swing_ratio': recent_stance_swing_ratio
        }
        overall_placeholders = {
            'total_strides': overall_total_strides,
            'cadence': overall_cadence,
            'stride_variability': overall_stride_variability,
            'stride_symmetry': overall_stride_symmetry,
            'contact_time': overall_contact_time,
            'stance_swing_ratio': overall_stance_swing_ratio
        }
        display_empty_metrics(recent_placeholders, overall_placeholders, TOOLTIPS)
        
        # Empty per-foot metrics
        for stride, contact in [(recent_lf_stride, recent_lf_contact), (recent_rf_stride, recent_rf_contact),
                                (overall_lf_stride, overall_lf_contact), (overall_rf_stride, overall_rf_contact)]:
            stride.metric("Stride Time", value="--", help=TOOLTIPS['stride_time'])
            contact.metric("Contact Time", value="--", help=TOOLTIPS['contact_time'])
    
    if st.session_state.last_chart is None:
        status.info("Ready to stream. Click 'Start Stream' to begin.")


# === Stream Control Logic ===
if start_stream and selected_player and selected_sensors:
    st.session_state.streaming = True
    asyncio.run(stream_imu_data(selected_player, selected_sensors, start_time))

if stop_stream:
    st.session_state.streaming = False
    st.rerun()

if not selected_sensors:
    st.warning("Please select at least one sensor to display")
