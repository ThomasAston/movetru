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
    IMUStreamUI
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

# Filter configuration (in sidebar)
filter_config = ui.render_filter_controls(sampling_rate=stream_config.SAMPLING_RATE)

# Initialize processor with filter configuration
processor = IMUStreamProcessor(stream_config, filter_config)

# Status and chart placeholders
status = ui.create_status_placeholder()
chart_lf, chart_rf = ui.create_chart_placeholders(selected_sensors)


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
    
    # Calculate y-axis ranges based on a larger sample to ensure they cover the data
    status.info("Calculating stable y-axis ranges...")
    sample_window_for_range = int(stream_config.SAMPLING_RATE * stream_config.RANGE_CALCULATION_WINDOW)
    range_window_end = min(start_from + sample_window_for_range, min_len)
    processor.calculate_y_ranges(df_lf, df_rf, sensors, start_from, range_window_end)
    
    # Display streaming info
    actual_start_time = df_lf['Time'][start_from]
    filter_info = processor.get_filter_info()
    
    if filter_info:
        filter_desc = filter_info.get('description', 'Unknown filter')
        status.success(
            f"Streaming {min_len - start_from} samples (starting at {actual_start_time:.2f}s, sample {start_from})\n\n"
            f"🎛️ Filter: {filter_desc}"
        )
    else:
        status.success(
            f"Streaming {min_len - start_from} samples (starting at {actual_start_time:.2f}s, sample {start_from})\n\n"
            f"🎛️ Filter: None (raw data)"
        )
    
    # Stream the data
    last_update_time = time.time()
    sample_count = 0
    
    for row_idx in range(start_from, min_len):
        # Check if we should stop
        if 'stop_streaming' in st.session_state and st.session_state.stop_streaming:
            status.warning("Stream stopped by user")
            break
        
        row_lf = df_lf.row(row_idx, named=True)
        row_rf = df_rf.row(row_idx, named=True)
        current_time_lf = row_lf['Time']
        current_time_rf = row_rf['Time']
        
        # Update windows using processor
        processor.update_windows(row_lf, row_rf, sensors)
        sample_count += 1
        
        # Only update charts every UPDATE_INTERVAL samples for better performance
        if sample_count >= stream_config.UPDATE_INTERVAL:
            # Prepare all figures first, then update both charts together for better sync
            figs_lf = []
            figs_rf = []
            
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
                
                # Get updated ranges
                y_range_lf = processor.y_ranges_lf[sensor]
                y_range_rf = processor.y_ranges_rf[sensor]
                
                # Create charts using renderer
                fig_lf = renderer.create_sensor_chart(
                    times_lf_display, values_lf_display, y_range_lf, sensor, 'LF'
                )
                fig_rf = renderer.create_sensor_chart(
                    times_rf_display, values_rf_display, y_range_rf, sensor, 'RF'
                )
                
                figs_lf.append(fig_lf)
                figs_rf.append(fig_rf)
            
            # Update both charts together for better synchronization
            if chart_lf and figs_lf:
                for fig in figs_lf:
                    chart_lf.plotly_chart(fig, use_container_width=True)
            
            if chart_rf and figs_rf:
                for fig in figs_rf:
                    chart_rf.plotly_chart(fig, use_container_width=True)
            
            # Update status and calculate timing
            current_real_time = time.time()
            elapsed_real_time = current_real_time - last_update_time
            actual_speed = (
                (stream_config.UPDATE_INTERVAL / stream_config.SAMPLING_RATE) / elapsed_real_time 
                if elapsed_real_time > 0 
                else stream_config.DEFAULT_SPEED
            )
            
            status.info(
                f"Sample {row_idx + 1}/{min_len} | "
                f"LF Time: {current_time_lf:.2f}s | RF Time: {current_time_rf:.2f}s | "
                f"Target Speed: {stream_config.DEFAULT_SPEED}x | Actual: {actual_speed:.1f}x"
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


# Handle streaming
if start_stream and selected_player and selected_sensors:
    st.session_state.stop_streaming = False
    asyncio.run(stream_imu_data(selected_player, selected_sensors, start_time))

if stop_stream:
    st.session_state.stop_streaming = True
    status.warning("Stopping stream...")

if not selected_sensors:
    st.warning("Please select at least one sensor to display")
