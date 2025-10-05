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
    GaitEventDetector
)

# Helper functions for metric display
def get_metric_status(value, metric_type):
    """
    Determine status (good/acceptable/poor) based on metric value and type.
    Returns tuple: (emoji, status_text)
    """
    if value is None:
        return "", ""
    
    if metric_type == 'stride_variability':
        # Lower is better (% CV)
        if value < 3:
            return "🟢", "Excellent"
        elif value < 5:
            return "🟡", "Acceptable"
        else:
            return "🔴", "High"
    
    elif metric_type == 'symmetry':
        # Lower is better (%)
        if value < 5:
            return "🟢", "Excellent"
        elif value < 10:
            return "🟡", "Good"
        else:
            return "🔴", "Asymmetric"
    
    elif metric_type == 'contact_time_walking':
        # Around 60% is typical for walking
        if 55 <= value <= 65:
            return "🟢", "Optimal"
        elif 50 <= value <= 70:
            return "🟡", "Acceptable"
        else:
            return "🔴", "Atypical"
    
    elif metric_type == 'stance_swing_walking':
        # Around 1.5 is typical for walking
        if 1.3 <= value <= 1.7:
            return "🟢", "Optimal"
        elif 1.0 <= value <= 2.0:
            return "🟡", "Acceptable"
        else:
            return "🔴", "Atypical"
    
    elif metric_type == 'cadence_walking':
        # 100-120 for walking, 160-180 for running
        if 100 <= value <= 120:
            return "🟢", "Walking"
        elif 160 <= value <= 180:
            return "🟢", "Running"
        elif 90 <= value <= 130 or 150 <= value <= 190:
            return "🟡", "Moderate"
        else:
            return "🟡", "Variable"
    
    return "", ""

def format_metric_value(value, unit, emoji="", status=""):
    """Format metric value with optional emoji and status."""
    if value is None:
        return "--"
    
    formatted = f"{value}{unit}"
    if emoji and status:
        return f"{emoji} {formatted}"
    return formatted

# Metric tooltips
TOOLTIPS = {
    'cadence': "Steps per minute. Walking: 100-120, Running: 160-180. Higher cadence often improves efficiency and reduces injury risk.",
    'stride_variability': "Consistency of stride timing (CV%). <3% = Excellent, 3-5% = Acceptable, >5% = High variability may indicate fatigue or injury.",
    'stride_symmetry': "Left/right balance (%). <5% = Excellent, 5-10% = Good, >10% = Asymmetric (may indicate injury or compensation).",
    'contact_time': "Ground contact as % of stride. Walking ~60%, Running ~35%. Increasing values may indicate fatigue.",
    'stance_swing_ratio': "Time on ground vs. in air. Walking ~1.5, Running ~0.5-0.7. Lower ratio = faster pace.",
    'stride_time': "Time for one complete gait cycle. Consistency (low SD) is important for injury prevention.",
}

# Initialize configurations
stream_config = StreamConfig()
ui_config = UIConfig()

# Initialize components
st.set_page_config(page_title="Movetru stride detection")
ui = IMUStreamUI(ui_config)
data_loader = IMUDataLoader(stream_config.DATA_DIR)
renderer = ChartRenderer(ui_config)

# Render UI
ui.render_header()

# Get available players and let user select
players = data_loader.get_available_players()
selected_player = ui.render_player_selector(players)

# Sensor selection (hardcoded for now, could be made configurable)
selected_sensors = ["Gyro Y"]

# Stream controls
start_time, start_stream, stop_stream = ui.render_stream_controls()

# Initialize processor
processor = IMUStreamProcessor(stream_config)

# Initialize gait detector
gait_detector = GaitEventDetector(
    fs=stream_config.SAMPLING_RATE,
    msw_threshold=stream_config.MSW_THRESHOLD,
    zc_threshold=stream_config.ZC_THRESHOLD,
    ma_window=stream_config.FILTER_WINDOW_SIZE,
    max_buffer_size=stream_config.SAMPLING_RATE * 5,
    max_stride_time=stream_config.MAX_STRIDE_TIME,
    min_stride_time=stream_config.MIN_STRIDE_TIME,
)

# Status and chart placeholders
status = ui.create_status_placeholder()
chart = ui.create_chart_placeholders(selected_sensors)

# Metrics placeholders
st.markdown("---")  # Add a separator line
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

# Initialize session state
if 'streaming' not in st.session_state:
    st.session_state.streaming = False
if 'last_chart' not in st.session_state:
    st.session_state.last_chart = None
if 'last_metrics' not in st.session_state:
    st.session_state.last_metrics = None


async def stream_imu_data(selected_player: str, sensors: list, start_from_time: float = 0.0):
    """
    Asynchronously stream IMU data from left and right foot parquet files.
    
    Args:
        selected_player: Player ID to stream data for
        sensors: List of sensor columns to stream
        start_from_time: Time in seconds to start streaming from
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
    chart_update_count = 0  # Track chart updates for event marker rendering
    
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
            chart_update_count += 1
            
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
                
                # Calculate dynamic x-axis range based on current data
                # Always show a fixed 10-second window that scrolls with the data
                if data['lf_times'] and data['rf_times']:
                    # Get the maximum time from both feet
                    current_max_time = max(max(data['lf_times']), max(data['rf_times']))
                    # Set range to show window_duration_seconds ending at current_max_time
                    x_min = current_max_time - window_duration_seconds
                    x_max = current_max_time
                    dynamic_x_range = [x_min, x_max]
                else:
                    # Fallback to initial range if no data yet
                    dynamic_x_range = [actual_start_time, actual_start_time + window_duration_seconds]
                
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
            
            # Update gait metrics
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
            
            # Calculate combined metrics
            total_strides_recent = metrics_lf_recent['total_strides'] + metrics_rf_recent['total_strides']
            avg_cadence_recent = None
            if metrics_lf_recent['cadence'] is not None and metrics_rf_recent['cadence'] is not None:
                avg_cadence_recent = (metrics_lf_recent['cadence'] + metrics_rf_recent['cadence']) / 2
            
            avg_stride_cv_recent = None
            if metrics_lf_recent['stride_time_cv'] is not None and metrics_rf_recent['stride_time_cv'] is not None:
                avg_stride_cv_recent = (metrics_lf_recent['stride_time_cv'] + metrics_rf_recent['stride_time_cv']) / 2
            
            avg_contact_recent = None
            if metrics_lf_recent['contact_time_percent'] is not None and metrics_rf_recent['contact_time_percent'] is not None:
                avg_contact_recent = (metrics_lf_recent['contact_time_percent'] + metrics_rf_recent['contact_time_percent']) / 2
            
            avg_stance_swing_recent = None
            if metrics_lf_recent['stance_swing_ratio'] is not None and metrics_rf_recent['stance_swing_ratio'] is not None:
                avg_stance_swing_recent = (metrics_lf_recent['stance_swing_ratio'] + metrics_rf_recent['stance_swing_ratio']) / 2
            
            # Display recent metrics - Overall with colors
            recent_total_strides.metric("Total Strides", value=total_strides_recent)
            
            # Cadence with color
            cadence_emoji, cadence_status = get_metric_status(avg_cadence_recent, 'cadence_walking')
            cadence_display = f"{cadence_emoji} {avg_cadence_recent:.1f}" if avg_cadence_recent is not None else "--"
            recent_cadence.metric(
                "Cadence (steps/min)",
                value=cadence_display,
                help=TOOLTIPS['cadence']
            )
            
            # Stride Variability with color
            cv_emoji, cv_status = get_metric_status(avg_stride_cv_recent, 'stride_variability')
            cv_display = f"{cv_emoji} {avg_stride_cv_recent:.1f}%" if avg_stride_cv_recent is not None else "--"
            recent_stride_variability.metric(
                "Stride Variability (CV)",
                value=cv_display,
                help=TOOLTIPS['stride_variability']
            )
            
            # Stride Symmetry with color
            sym_emoji, sym_status = get_metric_status(symmetry_recent['stride_time_symmetry'], 'symmetry')
            sym_display = f"{sym_emoji} {symmetry_recent['stride_time_symmetry']:.1f}%" if symmetry_recent['stride_time_symmetry'] is not None else "--"
            recent_stride_symmetry.metric(
                "Stride Symmetry",
                value=sym_display,
                help=TOOLTIPS['stride_symmetry']
            )
            
            # Contact Time with color
            contact_emoji, contact_status = get_metric_status(avg_contact_recent, 'contact_time_walking')
            contact_display = f"{contact_emoji} {avg_contact_recent:.1f}%" if avg_contact_recent is not None else "--"
            recent_contact_time.metric(
                "Avg Contact Time",
                value=contact_display,
                help=TOOLTIPS['contact_time']
            )
            
            # Stance/Swing Ratio with color
            ratio_emoji, ratio_status = get_metric_status(avg_stance_swing_recent, 'stance_swing_walking')
            ratio_display = f"{ratio_emoji} {avg_stance_swing_recent:.2f}" if avg_stance_swing_recent is not None else "--"
            recent_stance_swing_ratio.metric(
                "Stance/Swing Ratio",
                value=ratio_display,
                help=TOOLTIPS['stance_swing_ratio']
            )
            
            # Display recent metrics - Per Foot
            recent_lf_stride.metric(
                "Stride Time",
                value=f"{metrics_lf_recent['stride_time_mean']:.3f} s" if metrics_lf_recent['stride_time_mean'] is not None else "--",
                delta=f"± {metrics_lf_recent['stride_time_std']:.3f} s" if metrics_lf_recent['stride_time_std'] is not None else None,
                delta_color="off",
                help=TOOLTIPS['stride_time']
            )
            lf_contact_emoji, _ = get_metric_status(metrics_lf_recent['contact_time_percent'], 'contact_time_walking')
            lf_contact_display = f"{lf_contact_emoji} {metrics_lf_recent['contact_time_percent']:.1f}%" if metrics_lf_recent['contact_time_percent'] is not None else "--"
            recent_lf_contact.metric(
                "Contact Time",
                value=lf_contact_display,
                help=TOOLTIPS['contact_time']
            )
            recent_rf_stride.metric(
                "Stride Time",
                value=f"{metrics_rf_recent['stride_time_mean']:.3f} s" if metrics_rf_recent['stride_time_mean'] is not None else "--",
                delta=f"± {metrics_rf_recent['stride_time_std']:.3f} s" if metrics_rf_recent['stride_time_std'] is not None else None,
                delta_color="off",
                help=TOOLTIPS['stride_time']
            )
            rf_contact_emoji, _ = get_metric_status(metrics_rf_recent['contact_time_percent'], 'contact_time_walking')
            rf_contact_display = f"{rf_contact_emoji} {metrics_rf_recent['contact_time_percent']:.1f}%" if metrics_rf_recent['contact_time_percent'] is not None else "--"
            recent_rf_contact.metric(
                "Contact Time",
                value=rf_contact_display,
                help=TOOLTIPS['contact_time']
            )
            
            # Calculate combined metrics
            total_strides_overall = metrics_lf_overall['total_strides'] + metrics_rf_overall['total_strides']
            avg_cadence_overall = None
            if metrics_lf_overall['cadence'] is not None and metrics_rf_overall['cadence'] is not None:
                avg_cadence_overall = (metrics_lf_overall['cadence'] + metrics_rf_overall['cadence']) / 2
            
            avg_stride_cv_overall = None
            if metrics_lf_overall['stride_time_cv'] is not None and metrics_rf_overall['stride_time_cv'] is not None:
                avg_stride_cv_overall = (metrics_lf_overall['stride_time_cv'] + metrics_rf_overall['stride_time_cv']) / 2
            
            avg_contact_overall = None
            if metrics_lf_overall['contact_time_percent'] is not None and metrics_rf_overall['contact_time_percent'] is not None:
                avg_contact_overall = (metrics_lf_overall['contact_time_percent'] + metrics_rf_overall['contact_time_percent']) / 2
            
            avg_stance_swing_overall = None
            if metrics_lf_overall['stance_swing_ratio'] is not None and metrics_rf_overall['stance_swing_ratio'] is not None:
                avg_stance_swing_overall = (metrics_lf_overall['stance_swing_ratio'] + metrics_rf_overall['stance_swing_ratio']) / 2
            
            # Display overall metrics - Overall with colors
            overall_total_strides.metric("Total Strides", value=total_strides_overall)
            
            # Cadence with color
            cadence_emoji_o, _ = get_metric_status(avg_cadence_overall, 'cadence_walking')
            cadence_display_o = f"{cadence_emoji_o} {avg_cadence_overall:.1f}" if avg_cadence_overall is not None else "--"
            overall_cadence.metric(
                "Cadence (steps/min)",
                value=cadence_display_o,
                help=TOOLTIPS['cadence']
            )
            
            # Stride Variability with color
            cv_emoji_o, _ = get_metric_status(avg_stride_cv_overall, 'stride_variability')
            cv_display_o = f"{cv_emoji_o} {avg_stride_cv_overall:.1f}%" if avg_stride_cv_overall is not None else "--"
            overall_stride_variability.metric(
                "Stride Variability (CV)",
                value=cv_display_o,
                help=TOOLTIPS['stride_variability']
            )
            
            # Stride Symmetry with color
            sym_emoji_o, _ = get_metric_status(symmetry_overall['stride_time_symmetry'], 'symmetry')
            sym_display_o = f"{sym_emoji_o} {symmetry_overall['stride_time_symmetry']:.1f}%" if symmetry_overall['stride_time_symmetry'] is not None else "--"
            overall_stride_symmetry.metric(
                "Stride Symmetry",
                value=sym_display_o,
                help=TOOLTIPS['stride_symmetry']
            )
            
            # Contact Time with color
            contact_emoji_o, _ = get_metric_status(avg_contact_overall, 'contact_time_walking')
            contact_display_o = f"{contact_emoji_o} {avg_contact_overall:.1f}%" if avg_contact_overall is not None else "--"
            overall_contact_time.metric(
                "Avg Contact Time",
                value=contact_display_o,
                help=TOOLTIPS['contact_time']
            )
            
            # Stance/Swing Ratio with color
            ratio_emoji_o, _ = get_metric_status(avg_stance_swing_overall, 'stance_swing_walking')
            ratio_display_o = f"{ratio_emoji_o} {avg_stance_swing_overall:.2f}" if avg_stance_swing_overall is not None else "--"
            overall_stance_swing_ratio.metric(
                "Stance/Swing Ratio",
                value=ratio_display_o,
                help=TOOLTIPS['stance_swing_ratio']
            )
            
            # Display overall metrics - Per Foot
            overall_lf_stride.metric(
                "Stride Time",
                value=f"{metrics_lf_overall['stride_time_mean']:.3f} s" if metrics_lf_overall['stride_time_mean'] is not None else "--",
                delta=f"± {metrics_lf_overall['stride_time_std']:.3f} s" if metrics_lf_overall['stride_time_std'] is not None else None,
                delta_color="off",
                help=TOOLTIPS['stride_time']
            )
            lf_contact_emoji_o, _ = get_metric_status(metrics_lf_overall['contact_time_percent'], 'contact_time_walking')
            lf_contact_display_o = f"{lf_contact_emoji_o} {metrics_lf_overall['contact_time_percent']:.1f}%" if metrics_lf_overall['contact_time_percent'] is not None else "--"
            overall_lf_contact.metric(
                "Contact Time",
                value=lf_contact_display_o,
                help=TOOLTIPS['contact_time']
            )
            overall_rf_stride.metric(
                "Stride Time",
                value=f"{metrics_rf_overall['stride_time_mean']:.3f} s" if metrics_rf_overall['stride_time_mean'] is not None else "--",
                delta=f"± {metrics_rf_overall['stride_time_std']:.3f} s" if metrics_rf_overall['stride_time_std'] is not None else None,
                delta_color="off",
                help=TOOLTIPS['stride_time']
            )
            rf_contact_emoji_o, _ = get_metric_status(metrics_rf_overall['contact_time_percent'], 'contact_time_walking')
            rf_contact_display_o = f"{rf_contact_emoji_o} {metrics_rf_overall['contact_time_percent']:.1f}%" if metrics_rf_overall['contact_time_percent'] is not None else "--"
            overall_rf_contact.metric(
                "Contact Time",
                value=rf_contact_display_o,
                help=TOOLTIPS['contact_time']
            )
            
            # Update status and calculate timing
            current_real_time = time.time()
            elapsed_real_time = current_real_time - last_update_time
            actual_speed = (
                (stream_config.UPDATE_INTERVAL / stream_config.SAMPLING_RATE) / elapsed_real_time 
                if elapsed_real_time > 0 
                else stream_config.DEFAULT_SPEED
            )
            
            # Get event counts for status (always available, not just when rendering)
            all_events_lf = gait_detector.get_all_events('left')
            all_events_rf = gait_detector.get_all_events('right')
            
            status.info(
                f"Sample {row_idx + 1}/{min_len} | "
                f"LF Time: {current_time_lf:.2f}s | RF Time: {current_time_rf:.2f}s | "
                f"Target Speed: {stream_config.DEFAULT_SPEED}x | Actual: {actual_speed:.1f}x | "
                f"Events: L:{len(all_events_lf['msw'])} R:{len(all_events_rf['msw'])} MSW"
            )
            
            # Calculate delay accounting for rendering time
            target_update_time = stream_config.UPDATE_INTERVAL / stream_config.SAMPLING_RATE / stream_config.DEFAULT_SPEED
            rendering_time = elapsed_real_time
            sleep_time = max(0, target_update_time - rendering_time)
            
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                # Give control back to event loop even if we're behind
                await asyncio.sleep(0)
            
            last_update_time = time.time()
            sample_count = 0
    
    status.success(f"Stream completed! Processed {min_len} samples")
    st.session_state.streaming = False


# Pre-populate the UI with frozen/empty state before streaming starts
# This prevents the UI from disappearing when start is pressed
if not st.session_state.streaming:
    if st.session_state.last_chart is not None and chart:
        # Show frozen chart from last stream
        chart.plotly_chart(st.session_state.last_chart, use_container_width=True, config={'displayModeBar': False}, key='frozen_chart')
    elif chart:
        # Show empty chart before first stream with x-range based on start time
        window_duration = stream_config.DEFAULT_WINDOW_SIZE / stream_config.SAMPLING_RATE
        # Use the selected start_time to set the x-axis range
        empty_x_range = [start_time, start_time + window_duration]
        empty_fig = renderer.create_combined_chart(
            times_lf=[start_time], values_lf=[0],
            times_rf=[start_time], values_rf=[0],
            y_range=[-200, 200],
            sensor_name="Gyro Y",
            events_lf=None,
            events_rf=None,
            x_range=empty_x_range
        )
        chart.plotly_chart(empty_fig, use_container_width=True, config={'displayModeBar': False}, key='empty_chart')
    
    # Display metrics
    if st.session_state.last_metrics is not None:
        # Show frozen metrics from last stream
        metrics_lf_recent = st.session_state.last_metrics['recent_lf']
        metrics_rf_recent = st.session_state.last_metrics['recent_rf']
        metrics_lf_overall = st.session_state.last_metrics['overall_lf']
        metrics_rf_overall = st.session_state.last_metrics['overall_rf']
        symmetry_recent = st.session_state.last_metrics['symmetry_recent']
        symmetry_overall = st.session_state.last_metrics['symmetry_overall']
        
        # Calculate combined metrics for recent
        total_strides_recent = metrics_lf_recent['total_strides'] + metrics_rf_recent['total_strides']
        avg_cadence_recent = None
        if metrics_lf_recent['cadence'] is not None and metrics_rf_recent['cadence'] is not None:
            avg_cadence_recent = (metrics_lf_recent['cadence'] + metrics_rf_recent['cadence']) / 2
        avg_stride_cv_recent = None
        if metrics_lf_recent['stride_time_cv'] is not None and metrics_rf_recent['stride_time_cv'] is not None:
            avg_stride_cv_recent = (metrics_lf_recent['stride_time_cv'] + metrics_rf_recent['stride_time_cv']) / 2
        avg_contact_recent = None
        if metrics_lf_recent['contact_time_percent'] is not None and metrics_rf_recent['contact_time_percent'] is not None:
            avg_contact_recent = (metrics_lf_recent['contact_time_percent'] + metrics_rf_recent['contact_time_percent']) / 2
        avg_stance_swing_recent = None
        if metrics_lf_recent['stance_swing_ratio'] is not None and metrics_rf_recent['stance_swing_ratio'] is not None:
            avg_stance_swing_recent = (metrics_lf_recent['stance_swing_ratio'] + metrics_rf_recent['stance_swing_ratio']) / 2
        
        # Display recent metrics with colors
        recent_total_strides.metric("Total Strides", value=total_strides_recent)
        
        cadence_emoji_f, _ = get_metric_status(avg_cadence_recent, 'cadence_walking')
        recent_cadence.metric("Cadence (steps/min)", value=f"{cadence_emoji_f} {avg_cadence_recent:.1f}" if avg_cadence_recent is not None else "--", help=TOOLTIPS['cadence'])
        
        cv_emoji_f, _ = get_metric_status(avg_stride_cv_recent, 'stride_variability')
        recent_stride_variability.metric("Stride Variability (CV)", value=f"{cv_emoji_f} {avg_stride_cv_recent:.1f}%" if avg_stride_cv_recent is not None else "--", help=TOOLTIPS['stride_variability'])
        
        sym_emoji_f, _ = get_metric_status(symmetry_recent['stride_time_symmetry'], 'symmetry')
        recent_stride_symmetry.metric("Stride Symmetry", value=f"{sym_emoji_f} {symmetry_recent['stride_time_symmetry']:.1f}%" if symmetry_recent['stride_time_symmetry'] is not None else "--", help=TOOLTIPS['stride_symmetry'])
        
        contact_emoji_f, _ = get_metric_status(avg_contact_recent, 'contact_time_walking')
        recent_contact_time.metric("Avg Contact Time", value=f"{contact_emoji_f} {avg_contact_recent:.1f}%" if avg_contact_recent is not None else "--", help=TOOLTIPS['contact_time'])
        
        ratio_emoji_f, _ = get_metric_status(avg_stance_swing_recent, 'stance_swing_walking')
        recent_stance_swing_ratio.metric("Stance/Swing Ratio", value=f"{ratio_emoji_f} {avg_stance_swing_recent:.2f}" if avg_stance_swing_recent is not None else "--", help=TOOLTIPS['stance_swing_ratio'])
        
        recent_lf_stride.metric("Stride Time", value=f"{metrics_lf_recent['stride_time_mean']:.3f} s" if metrics_lf_recent['stride_time_mean'] is not None else "--", delta=f"± {metrics_lf_recent['stride_time_std']:.3f} s" if metrics_lf_recent['stride_time_std'] is not None else None, delta_color="off", help=TOOLTIPS['stride_time'])
        
        lf_contact_emoji_f, _ = get_metric_status(metrics_lf_recent['contact_time_percent'], 'contact_time_walking')
        recent_lf_contact.metric("Contact Time", value=f"{lf_contact_emoji_f} {metrics_lf_recent['contact_time_percent']:.1f}%" if metrics_lf_recent['contact_time_percent'] is not None else "--", help=TOOLTIPS['contact_time'])
        
        recent_rf_stride.metric("Stride Time", value=f"{metrics_rf_recent['stride_time_mean']:.3f} s" if metrics_rf_recent['stride_time_mean'] is not None else "--", delta=f"± {metrics_rf_recent['stride_time_std']:.3f} s" if metrics_rf_recent['stride_time_std'] is not None else None, delta_color="off", help=TOOLTIPS['stride_time'])
        
        rf_contact_emoji_f, _ = get_metric_status(metrics_rf_recent['contact_time_percent'], 'contact_time_walking')
        recent_rf_contact.metric("Contact Time", value=f"{rf_contact_emoji_f} {metrics_rf_recent['contact_time_percent']:.1f}%" if metrics_rf_recent['contact_time_percent'] is not None else "--", help=TOOLTIPS['contact_time'])
        
        # Calculate combined metrics for overall
        total_strides_overall = metrics_lf_overall['total_strides'] + metrics_rf_overall['total_strides']
        avg_cadence_overall = None
        if metrics_lf_overall['cadence'] is not None and metrics_rf_overall['cadence'] is not None:
            avg_cadence_overall = (metrics_lf_overall['cadence'] + metrics_rf_overall['cadence']) / 2
        avg_stride_cv_overall = None
        if metrics_lf_overall['stride_time_cv'] is not None and metrics_rf_overall['stride_time_cv'] is not None:
            avg_stride_cv_overall = (metrics_lf_overall['stride_time_cv'] + metrics_rf_overall['stride_time_cv']) / 2
        avg_contact_overall = None
        if metrics_lf_overall['contact_time_percent'] is not None and metrics_rf_overall['contact_time_percent'] is not None:
            avg_contact_overall = (metrics_lf_overall['contact_time_percent'] + metrics_rf_overall['contact_time_percent']) / 2
        avg_stance_swing_overall = None
        if metrics_lf_overall['stance_swing_ratio'] is not None and metrics_rf_overall['stance_swing_ratio'] is not None:
            avg_stance_swing_overall = (metrics_lf_overall['stance_swing_ratio'] + metrics_rf_overall['stance_swing_ratio']) / 2
        
        # Display overall metrics with colors
        overall_total_strides.metric("Total Strides", value=total_strides_overall)
        
        cadence_emoji_o, _ = get_metric_status(avg_cadence_overall, 'cadence_walking')
        overall_cadence.metric("Cadence (steps/min)", value=f"{cadence_emoji_o} {avg_cadence_overall:.1f}" if avg_cadence_overall is not None else "--", help=TOOLTIPS['cadence'])
        
        cv_emoji_o, _ = get_metric_status(avg_stride_cv_overall, 'stride_variability')
        overall_stride_variability.metric("Stride Variability (CV)", value=f"{cv_emoji_o} {avg_stride_cv_overall:.1f}%" if avg_stride_cv_overall is not None else "--", help=TOOLTIPS['stride_variability'])
        
        sym_emoji_o, _ = get_metric_status(symmetry_overall['stride_time_symmetry'], 'symmetry')
        overall_stride_symmetry.metric("Stride Symmetry", value=f"{sym_emoji_o} {symmetry_overall['stride_time_symmetry']:.1f}%" if symmetry_overall['stride_time_symmetry'] is not None else "--", help=TOOLTIPS['stride_symmetry'])
        
        contact_emoji_o, _ = get_metric_status(avg_contact_overall, 'contact_time_walking')
        overall_contact_time.metric("Avg Contact Time", value=f"{contact_emoji_o} {avg_contact_overall:.1f}%" if avg_contact_overall is not None else "--", help=TOOLTIPS['contact_time'])
        
        ratio_emoji_o, _ = get_metric_status(avg_stance_swing_overall, 'stance_swing_walking')
        overall_stance_swing_ratio.metric("Stance/Swing Ratio", value=f"{ratio_emoji_o} {avg_stance_swing_overall:.2f}" if avg_stance_swing_overall is not None else "--", help=TOOLTIPS['stance_swing_ratio'])
        
        overall_lf_stride.metric("Stride Time", value=f"{metrics_lf_overall['stride_time_mean']:.3f} s" if metrics_lf_overall['stride_time_mean'] is not None else "--", delta=f"± {metrics_lf_overall['stride_time_std']:.3f} s" if metrics_lf_overall['stride_time_std'] is not None else None, delta_color="off", help=TOOLTIPS['stride_time'])
        
        lf_contact_emoji_o, _ = get_metric_status(metrics_lf_overall['contact_time_percent'], 'contact_time_walking')
        overall_lf_contact.metric("Contact Time", value=f"{lf_contact_emoji_o} {metrics_lf_overall['contact_time_percent']:.1f}%" if metrics_lf_overall['contact_time_percent'] is not None else "--", help=TOOLTIPS['contact_time'])
        
        overall_rf_stride.metric("Stride Time", value=f"{metrics_rf_overall['stride_time_mean']:.3f} s" if metrics_rf_overall['stride_time_mean'] is not None else "--", delta=f"± {metrics_rf_overall['stride_time_std']:.3f} s" if metrics_rf_overall['stride_time_std'] is not None else None, delta_color="off", help=TOOLTIPS['stride_time'])
        
        rf_contact_emoji_o, _ = get_metric_status(metrics_rf_overall['contact_time_percent'], 'contact_time_walking')
        overall_rf_contact.metric("Contact Time", value=f"{rf_contact_emoji_o} {metrics_rf_overall['contact_time_percent']:.1f}%" if metrics_rf_overall['contact_time_percent'] is not None else "--", help=TOOLTIPS['contact_time'])
    else:
        # Show empty metrics before first stream with tooltips
        # Recent metrics
        recent_total_strides.metric("Total Strides", value="--")
        recent_cadence.metric("Cadence (steps/min)", value="--", help=TOOLTIPS['cadence'])
        recent_stride_variability.metric("Stride Variability (CV)", value="--", help=TOOLTIPS['stride_variability'])
        recent_stride_symmetry.metric("Stride Symmetry", value="--", help=TOOLTIPS['stride_symmetry'])
        recent_contact_time.metric("Avg Contact Time", value="--", help=TOOLTIPS['contact_time'])
        recent_stance_swing_ratio.metric("Stance/Swing Ratio", value="--", help=TOOLTIPS['stance_swing_ratio'])
        recent_lf_stride.metric("Stride Time", value="--", help=TOOLTIPS['stride_time'])
        recent_lf_contact.metric("Contact Time", value="--", help=TOOLTIPS['contact_time'])
        recent_rf_stride.metric("Stride Time", value="--", help=TOOLTIPS['stride_time'])
        recent_rf_contact.metric("Contact Time", value="--", help=TOOLTIPS['contact_time'])
        
        # Overall metrics
        overall_total_strides.metric("Total Strides", value="--")
        overall_cadence.metric("Cadence (steps/min)", value="--", help=TOOLTIPS['cadence'])
        overall_stride_variability.metric("Stride Variability (CV)", value="--", help=TOOLTIPS['stride_variability'])
        overall_stride_symmetry.metric("Stride Symmetry", value="--", help=TOOLTIPS['stride_symmetry'])
        overall_contact_time.metric("Avg Contact Time", value="--", help=TOOLTIPS['contact_time'])
        overall_stance_swing_ratio.metric("Stance/Swing Ratio", value="--", help=TOOLTIPS['stance_swing_ratio'])
        overall_lf_stride.metric("Stride Time", value="--", help=TOOLTIPS['stride_time'])
        overall_lf_contact.metric("Contact Time", value="--", help=TOOLTIPS['contact_time'])
        overall_rf_stride.metric("Stride Time", value="--", help=TOOLTIPS['stride_time'])
        overall_rf_contact.metric("Contact Time", value="--", help=TOOLTIPS['contact_time'])
    
    if st.session_state.last_chart is None:
        status.info("Ready to stream. Click 'Start Stream' to begin.")

# Handle streaming (after displaying initial state)
if start_stream and selected_player and selected_sensors:
    st.session_state.streaming = True
    asyncio.run(stream_imu_data(selected_player, selected_sensors, start_time))

if stop_stream:
    st.session_state.streaming = False
    st.rerun()  # Force rerun to show frozen state immediately

if not selected_sensors:
    st.warning("Please select at least one sensor to display")
