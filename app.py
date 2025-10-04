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
st.subheader("📊 Gait Metrics")

# Create tabs for different views
tab1, tab2 = st.tabs(["📈 Recent (Last 5s)", "📊 Overall Session"])

with tab1:
    col_lf_recent, col_rf_recent = st.columns(2)
    with col_lf_recent:
        st.markdown("#### 🦶 Left Foot")
        recent_lf_strides = st.empty()
        recent_lf_stance = st.empty()
        recent_lf_swing = st.empty()
        recent_lf_stride = st.empty()
    with col_rf_recent:
        st.markdown("#### 🦶 Right Foot")
        recent_rf_strides = st.empty()
        recent_rf_stance = st.empty()
        recent_rf_swing = st.empty()
        recent_rf_stride = st.empty()

with tab2:
    col_lf_overall, col_rf_overall = st.columns(2)
    with col_lf_overall:
        st.markdown("#### 🦶 Left Foot")
        overall_lf_strides = st.empty()
        overall_lf_stance = st.empty()
        overall_lf_swing = st.empty()
        overall_lf_stride = st.empty()
    with col_rf_overall:
        st.markdown("#### 🦶 Right Foot")
        overall_rf_strides = st.empty()
        overall_rf_stance = st.empty()
        overall_rf_swing = st.empty()
        overall_rf_stride = st.empty()

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
                
                # Create combined chart with side-by-side subplots
                fig = renderer.create_combined_chart(
                    times_lf_display, values_lf_display,
                    times_rf_display, values_rf_display,
                    y_range_combined, sensor,
                    events_lf, events_rf
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
            
            # Save to session state for freezing
            st.session_state.last_metrics = {
                'recent_lf': metrics_lf_recent,
                'recent_rf': metrics_rf_recent,
                'overall_lf': metrics_lf_overall,
                'overall_rf': metrics_rf_overall
            }
            
            # Display recent metrics (Left Foot)
            recent_lf_strides.metric(
                "Total Strides",
                value=metrics_lf_recent['total_strides']
            )
            recent_lf_stance.metric(
                "Stance Time",
                value=f"{metrics_lf_recent['stance_time_mean']:.3f} s" if metrics_lf_recent['stance_time_mean'] is not None else "--",
                delta=f"± {metrics_lf_recent['stance_time_std']:.3f} s" if metrics_lf_recent['stance_time_std'] is not None else None,
                delta_color="off"
            )
            recent_lf_swing.metric(
                "Swing Time",
                value=f"{metrics_lf_recent['swing_time_mean']:.3f} s" if metrics_lf_recent['swing_time_mean'] is not None else "--",
                delta=f"± {metrics_lf_recent['swing_time_std']:.3f} s" if metrics_lf_recent['swing_time_std'] is not None else None,
                delta_color="off"
            )
            recent_lf_stride.metric(
                "Stride Time",
                value=f"{metrics_lf_recent['stride_time_mean']:.3f} s" if metrics_lf_recent['stride_time_mean'] is not None else "--",
                delta=f"± {metrics_lf_recent['stride_time_std']:.3f} s" if metrics_lf_recent['stride_time_std'] is not None else None,
                delta_color="off"
            )
            
            # Display recent metrics (Right Foot)
            recent_rf_strides.metric(
                "Total Strides",
                value=metrics_rf_recent['total_strides']
            )
            recent_rf_stance.metric(
                "Stance Time",
                value=f"{metrics_rf_recent['stance_time_mean']:.3f} s" if metrics_rf_recent['stance_time_mean'] is not None else "--",
                delta=f"± {metrics_rf_recent['stance_time_std']:.3f} s" if metrics_rf_recent['stance_time_std'] is not None else None,
                delta_color="off"
            )
            recent_rf_swing.metric(
                "Swing Time",
                value=f"{metrics_rf_recent['swing_time_mean']:.3f} s" if metrics_rf_recent['swing_time_mean'] is not None else "--",
                delta=f"± {metrics_rf_recent['swing_time_std']:.3f} s" if metrics_rf_recent['swing_time_std'] is not None else None,
                delta_color="off"
            )
            recent_rf_stride.metric(
                "Stride Time",
                value=f"{metrics_rf_recent['stride_time_mean']:.3f} s" if metrics_rf_recent['stride_time_mean'] is not None else "--",
                delta=f"± {metrics_rf_recent['stride_time_std']:.3f} s" if metrics_rf_recent['stride_time_std'] is not None else None,
                delta_color="off"
            )
            
            # Display overall metrics (Left Foot)
            overall_lf_strides.metric(
                "Total Strides",
                value=metrics_lf_overall['total_strides']
            )
            overall_lf_stance.metric(
                "Stance Time",
                value=f"{metrics_lf_overall['stance_time_mean']:.3f} s" if metrics_lf_overall['stance_time_mean'] is not None else "--",
                delta=f"± {metrics_lf_overall['stance_time_std']:.3f} s" if metrics_lf_overall['stance_time_std'] is not None else None,
                delta_color="off"
            )
            overall_lf_swing.metric(
                "Swing Time",
                value=f"{metrics_lf_overall['swing_time_mean']:.3f} s" if metrics_lf_overall['swing_time_mean'] is not None else "--",
                delta=f"± {metrics_lf_overall['swing_time_std']:.3f} s" if metrics_lf_overall['swing_time_std'] is not None else None,
                delta_color="off"
            )
            overall_lf_stride.metric(
                "Stride Time",
                value=f"{metrics_lf_overall['stride_time_mean']:.3f} s" if metrics_lf_overall['stride_time_mean'] is not None else "--",
                delta=f"± {metrics_lf_overall['stride_time_std']:.3f} s" if metrics_lf_overall['stride_time_std'] is not None else None,
                delta_color="off"
            )
            
            # Display overall metrics (Right Foot)
            overall_rf_strides.metric(
                "Total Strides",
                value=metrics_rf_overall['total_strides']
            )
            overall_rf_stance.metric(
                "Stance Time",
                value=f"{metrics_rf_overall['stance_time_mean']:.3f} s" if metrics_rf_overall['stance_time_mean'] is not None else "--",
                delta=f"± {metrics_rf_overall['stance_time_std']:.3f} s" if metrics_rf_overall['stance_time_std'] is not None else None,
                delta_color="off"
            )
            overall_rf_swing.metric(
                "Swing Time",
                value=f"{metrics_rf_overall['swing_time_mean']:.3f} s" if metrics_rf_overall['swing_time_mean'] is not None else "--",
                delta=f"± {metrics_rf_overall['swing_time_std']:.3f} s" if metrics_rf_overall['swing_time_std'] is not None else None,
                delta_color="off"
            )
            overall_rf_stride.metric(
                "Stride Time",
                value=f"{metrics_rf_overall['stride_time_mean']:.3f} s" if metrics_rf_overall['stride_time_mean'] is not None else "--",
                delta=f"± {metrics_rf_overall['stride_time_std']:.3f} s" if metrics_rf_overall['stride_time_std'] is not None else None,
                delta_color="off"
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
        # Show empty chart before first stream
        empty_fig = renderer.create_combined_chart(
            times_lf=[0], values_lf=[0],
            times_rf=[0], values_rf=[0],
            y_range=[-200, 200],
            sensor_name="Gyro Y",
            events_lf=None,
            events_rf=None
        )
        chart.plotly_chart(empty_fig, use_container_width=True, config={'displayModeBar': False}, key='empty_chart')
    
    # Display metrics
    if st.session_state.last_metrics is not None:
        # Show frozen metrics from last stream
        metrics_lf_recent = st.session_state.last_metrics['recent_lf']
        metrics_rf_recent = st.session_state.last_metrics['recent_rf']
        metrics_lf_overall = st.session_state.last_metrics['overall_lf']
        metrics_rf_overall = st.session_state.last_metrics['overall_rf']
        
        # Display recent metrics (Left Foot)
        recent_lf_strides.metric("Total Strides", value=metrics_lf_recent['total_strides'])
        recent_lf_stance.metric(
            "Stance Time",
            value=f"{metrics_lf_recent['stance_time_mean']:.3f} s" if metrics_lf_recent['stance_time_mean'] is not None else "--",
            delta=f"± {metrics_lf_recent['stance_time_std']:.3f} s" if metrics_lf_recent['stance_time_std'] is not None else None,
            delta_color="off"
        )
        recent_lf_swing.metric(
            "Swing Time",
            value=f"{metrics_lf_recent['swing_time_mean']:.3f} s" if metrics_lf_recent['swing_time_mean'] is not None else "--",
            delta=f"± {metrics_lf_recent['swing_time_std']:.3f} s" if metrics_lf_recent['swing_time_std'] is not None else None,
            delta_color="off"
        )
        recent_lf_stride.metric(
            "Stride Time",
            value=f"{metrics_lf_recent['stride_time_mean']:.3f} s" if metrics_lf_recent['stride_time_mean'] is not None else "--",
            delta=f"± {metrics_lf_recent['stride_time_std']:.3f} s" if metrics_lf_recent['stride_time_std'] is not None else None,
            delta_color="off"
        )
        
        # Display recent metrics (Right Foot)
        recent_rf_strides.metric("Total Strides", value=metrics_rf_recent['total_strides'])
        recent_rf_stance.metric(
            "Stance Time",
            value=f"{metrics_rf_recent['stance_time_mean']:.3f} s" if metrics_rf_recent['stance_time_mean'] is not None else "--",
            delta=f"± {metrics_rf_recent['stance_time_std']:.3f} s" if metrics_rf_recent['stance_time_std'] is not None else None,
            delta_color="off"
        )
        recent_rf_swing.metric(
            "Swing Time",
            value=f"{metrics_rf_recent['swing_time_mean']:.3f} s" if metrics_rf_recent['swing_time_mean'] is not None else "--",
            delta=f"± {metrics_rf_recent['swing_time_std']:.3f} s" if metrics_rf_recent['swing_time_std'] is not None else None,
            delta_color="off"
        )
        recent_rf_stride.metric(
            "Stride Time",
            value=f"{metrics_rf_recent['stride_time_mean']:.3f} s" if metrics_rf_recent['stride_time_mean'] is not None else "--",
            delta=f"± {metrics_rf_recent['stride_time_std']:.3f} s" if metrics_rf_recent['stride_time_std'] is not None else None,
            delta_color="off"
        )
        
        # Display overall metrics (Left Foot)
        overall_lf_strides.metric("Total Strides", value=metrics_lf_overall['total_strides'])
        overall_lf_stance.metric(
            "Stance Time",
            value=f"{metrics_lf_overall['stance_time_mean']:.3f} s" if metrics_lf_overall['stance_time_mean'] is not None else "--",
            delta=f"± {metrics_lf_overall['stance_time_std']:.3f} s" if metrics_lf_overall['stance_time_std'] is not None else None,
            delta_color="off"
        )
        overall_lf_swing.metric(
            "Swing Time",
            value=f"{metrics_lf_overall['swing_time_mean']:.3f} s" if metrics_lf_overall['swing_time_mean'] is not None else "--",
            delta=f"± {metrics_lf_overall['swing_time_std']:.3f} s" if metrics_lf_overall['swing_time_std'] is not None else None,
            delta_color="off"
        )
        overall_lf_stride.metric(
            "Stride Time",
            value=f"{metrics_lf_overall['stride_time_mean']:.3f} s" if metrics_lf_overall['stride_time_mean'] is not None else "--",
            delta=f"± {metrics_lf_overall['stride_time_std']:.3f} s" if metrics_lf_overall['stride_time_std'] is not None else None,
            delta_color="off"
        )
        
        # Display overall metrics (Right Foot)
        overall_rf_strides.metric("Total Strides", value=metrics_rf_overall['total_strides'])
        overall_rf_stance.metric(
            "Stance Time",
            value=f"{metrics_rf_overall['stance_time_mean']:.3f} s" if metrics_rf_overall['stance_time_mean'] is not None else "--",
            delta=f"± {metrics_rf_overall['stance_time_std']:.3f} s" if metrics_rf_overall['stance_time_std'] is not None else None,
            delta_color="off"
        )
        overall_rf_swing.metric(
            "Swing Time",
            value=f"{metrics_rf_overall['swing_time_mean']:.3f} s" if metrics_rf_overall['swing_time_mean'] is not None else "--",
            delta=f"± {metrics_rf_overall['swing_time_std']:.3f} s" if metrics_rf_overall['swing_time_std'] is not None else None,
            delta_color="off"
        )
        overall_rf_stride.metric(
            "Stride Time",
            value=f"{metrics_rf_overall['stride_time_mean']:.3f} s" if metrics_rf_overall['stride_time_mean'] is not None else "--",
            delta=f"± {metrics_rf_overall['stride_time_std']:.3f} s" if metrics_rf_overall['stride_time_std'] is not None else None,
            delta_color="off"
        )
    else:
        # Show empty metrics before first stream
        # Recent metrics (Left Foot)
        recent_lf_strides.metric("Total Strides", value="--")
        recent_lf_stance.metric("Stance Time", value="--")
        recent_lf_swing.metric("Swing Time", value="--")
        recent_lf_stride.metric("Stride Time", value="--")
        
        # Recent metrics (Right Foot)
        recent_rf_strides.metric("Total Strides", value="--")
        recent_rf_stance.metric("Stance Time", value="--")
        recent_rf_swing.metric("Swing Time", value="--")
        recent_rf_stride.metric("Stride Time", value="--")
        
        # Overall metrics (Left Foot)
        overall_lf_strides.metric("Total Strides", value="--")
        overall_lf_stance.metric("Stance Time", value="--")
        overall_lf_swing.metric("Swing Time", value="--")
        overall_lf_stride.metric("Stride Time", value="--")
        
        # Overall metrics (Right Foot)
        overall_rf_strides.metric("Total Strides", value="--")
        overall_rf_stance.metric("Stance Time", value="--")
        overall_rf_swing.metric("Swing Time", value="--")
        overall_rf_stride.metric("Stride Time", value="--")
    
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
