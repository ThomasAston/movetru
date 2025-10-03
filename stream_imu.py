"""
Streamlit app to simulate real-time IMU data streaming from parquet files.
Based on the streamlit-stream websocket example, but reading from local parquet files.
"""
import asyncio
import streamlit as st
import polars as pl
from pathlib import Path
from collections import deque, defaultdict
from functools import partial
import time
from PIL import Image
import plotly.graph_objects as go

# Configuration
DATA_DIR = Path("data/raw/imu")
UPDATE_INTERVAL = 25  # Update chart every N samples for smoother performance
SAMPLING_RATE = 256  # Hz
DOWNSAMPLE_FACTOR = 2  # Display every Nth point to reduce rendering load

st.set_page_config(page_title="Movetru stride detection")

# UI Elements
logo_path = Path("data/images/logo.webp")
logo_img = Image.open(logo_path)
st.image(logo_img, width=logo_img.width // 2)
st.title("Real-time stride event detection")

# Player selection - extract unique player IDs from LF files
available_files = sorted(DATA_DIR.glob("*_LF.parquet"))
player_ids = [f.stem.replace("_LF", "") for f in available_files]
selected_player = st.selectbox("Select Player", player_ids, index=0 if player_ids else None)

selected_sensors = ["Gyro Y"]

# Window size
window_size = 2000

# Speed control
speed_multiplier = 1

# Start position
start_time = st.number_input("Start from time (seconds)", min_value=0.0, value=2170.0, step=10.0, 
                             help="Choose which time (in seconds) to start streaming from")

# Control buttons
col1, col2 = st.columns([1, 1])
with col1:
    start_stream = st.button("▶ Start Stream")
with col2:
    stop_stream = st.button("⏹ Stop Stream")

# Status
status = st.empty()

# Create placeholder charts for left and right foot
if selected_sensors:
    st.subheader("Live Data")
    st.markdown("**Left Foot (LF)**")
    chart_lf = st.empty()
    st.markdown("**Right Foot (RF)**")
    chart_rf = st.empty()
else:
    chart_lf = None
    chart_rf = None


async def stream_imu_data(lf_path: Path, rf_path: Path, sensors: list, window_size: int, speed: float, start_from_time: float = 0.0):
    """
    Asynchronously stream IMU data from left and right foot parquet files.
    
    Args:
        lf_path: Path to the left foot parquet file
        rf_path: Path to the right foot parquet file
        sensors: List of sensor columns to stream
        window_size: Number of samples to display in the rolling window
        speed: Speed multiplier for playback
        start_from_time: Time in seconds to start streaming from
    """
    # Load the data
    status.info(f"Loading {lf_path.name} and {rf_path.name}...")
    df_lf = pl.read_parquet(lf_path)
    df_rf = pl.read_parquet(rf_path)
    
    # Convert start time to sample index
    # Find the first sample where time >= start_from_time
    start_from = 0
    for idx in range(len(df_lf)):
        if df_lf['Time'][idx] >= start_from_time:
            start_from = idx
            break
    
    # Validate start_from
    min_len = min(len(df_lf), len(df_rf))
    if start_from >= min_len:
        status.error(f"Start time {start_from_time}s is beyond available data (max time: {df_lf['Time'][-1]:.2f}s)")
        return
    
    # Initialize rolling windows for each foot and sensor
    windows_lf = defaultdict(partial(deque, maxlen=window_size))
    windows_rf = defaultdict(partial(deque, maxlen=window_size))
    time_window_lf = deque(maxlen=window_size)
    time_window_rf = deque(maxlen=window_size)
    
    # Calculate y-axis ranges based on a larger sample to ensure they cover the data
    # Use a 30-second window (or available data) to calculate stable ranges
    status.info("Calculating stable y-axis ranges...")
    sample_window_for_range = int(SAMPLING_RATE * 30)  # 30 seconds of data
    range_window_end = min(start_from + sample_window_for_range, min_len)
    y_ranges_lf = {}
    y_ranges_rf = {}
    
    for sensor in sensors:
        # Left foot ranges - sample from larger window
        sample_values_lf = df_lf[sensor][start_from:range_window_end].to_list()
        if sample_values_lf:
            data_min = min(sample_values_lf)
            data_max = max(sample_values_lf)
            data_range = data_max - data_min
            
            # Set minimum range based on sensor type
            min_range = 400 if sensor.startswith('Gyro') else 2.0
            
            if data_range < min_range:
                center = (data_max + data_min) / 2
                y_ranges_lf[sensor] = [center - min_range / 2, center + min_range / 2]
            else:
                padding = data_range * 0.3  # Increased padding to 30%
                y_ranges_lf[sensor] = [data_min - padding, data_max + padding]
        
        # Right foot ranges - sample from larger window
        sample_values_rf = df_rf[sensor][start_from:range_window_end].to_list()
        if sample_values_rf:
            data_min = min(sample_values_rf)
            data_max = max(sample_values_rf)
            data_range = data_max - data_min
            
            min_range = 400 if sensor.startswith('Gyro') else 2.0
            
            if data_range < min_range:
                center = (data_max + data_min) / 2
                y_ranges_rf[sensor] = [center - min_range / 2, center + min_range / 2]
            else:
                padding = data_range * 0.3  # Increased padding to 30%
                y_ranges_rf[sensor] = [data_min - padding, data_max + padding]
    
    actual_start_time = df_lf['Time'][start_from]
    status.success(f"Streaming {min_len - start_from} samples (starting at {actual_start_time:.2f}s, sample {start_from})")
    
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
        
        # Update windows
        time_window_lf.append(current_time_lf)
        time_window_rf.append(current_time_rf)
        for sensor in sensors:
            windows_lf[sensor].append(row_lf[sensor])
            windows_rf[sensor].append(row_rf[sensor])
        
        sample_count += 1
        
        # Only update charts every UPDATE_INTERVAL samples for better performance
        if sample_count >= UPDATE_INTERVAL:
            # Pre-convert deques to lists once per update cycle
            times_lf = list(time_window_lf)
            times_rf = list(time_window_rf)
            
            # Downsample for display to reduce rendering overhead
            times_lf_display = times_lf[::DOWNSAMPLE_FACTOR]
            times_rf_display = times_rf[::DOWNSAMPLE_FACTOR]
            
            # Prepare all figures first, then update both charts together for better sync
            figs_lf = []
            figs_rf = []
            
            for sensor in sensors:
                # Prepare left foot chart data
                values_lf = list(windows_lf[sensor])
                values_lf_display = values_lf[::DOWNSAMPLE_FACTOR]
                
                # Get current ranges and expand if needed
                y_min_lf, y_max_lf = y_ranges_lf.get(sensor, [min(values_lf), max(values_lf)])
                current_min = min(values_lf)
                current_max = max(values_lf)
                
                # Dynamically expand ranges if current data exceeds them
                if current_min < y_min_lf:
                    y_min_lf = current_min - abs(current_min) * 0.1
                    y_ranges_lf[sensor][0] = y_min_lf
                if current_max > y_max_lf:
                    y_max_lf = current_max + abs(current_max) * 0.1
                    y_ranges_lf[sensor][1] = y_max_lf
                
                fig_lf = go.Figure()
                fig_lf.add_trace(go.Scatter(
                    x=times_lf_display,
                    y=values_lf_display,
                    mode='lines',
                    line=dict(color='#d68032', width=1.5),  # Orange for left foot
                    name=f"LF {sensor}"
                ))
                fig_lf.update_layout(
                    height=200,
                    margin=dict(l=20, r=20, t=10, b=20),
                    xaxis_title="Time (s)",
                    yaxis=dict(range=[y_min_lf, y_max_lf], fixedrange=True),
                    showlegend=False,
                    transition={'duration': 0},
                    uirevision='constant'
                )
                figs_lf.append(fig_lf)
                
                # Prepare right foot chart data
                values_rf = list(windows_rf[sensor])
                values_rf_display = values_rf[::DOWNSAMPLE_FACTOR]
                
                # Get current ranges and expand if needed
                y_min_rf, y_max_rf = y_ranges_rf.get(sensor, [min(values_rf), max(values_rf)])
                current_min = min(values_rf)
                current_max = max(values_rf)
                
                # Dynamically expand ranges if current data exceeds them
                if current_min < y_min_rf:
                    y_min_rf = current_min - abs(current_min) * 0.1
                    y_ranges_rf[sensor][0] = y_min_rf
                if current_max > y_max_rf:
                    y_max_rf = current_max + abs(current_max) * 0.1
                    y_ranges_rf[sensor][1] = y_max_rf
                
                fig_rf = go.Figure()
                fig_rf.add_trace(go.Scatter(
                    x=times_rf_display,
                    y=values_rf_display,
                    mode='lines',
                    line=dict(color='#2a9d8f', width=1.5),  # Teal for right foot
                    name=f"RF {sensor}"
                ))
                fig_rf.update_layout(
                    height=200,
                    margin=dict(l=20, r=20, t=10, b=20),
                    xaxis_title="Time (s)",
                    yaxis=dict(range=[y_min_rf, y_max_rf], fixedrange=True),
                    showlegend=False,
                    transition={'duration': 0},
                    uirevision='constant'
                )
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
            actual_speed = (UPDATE_INTERVAL / SAMPLING_RATE) / elapsed_real_time if elapsed_real_time > 0 else speed
            
            status.info(
                f"Sample {row_idx + 1}/{min_len} | "
                f"LF Time: {current_time_lf:.2f}s | RF Time: {current_time_rf:.2f}s | "
                f"Target Speed: {speed}x | Actual: {actual_speed:.1f}x"
            )
            
            # Calculate delay accounting for rendering time
            target_update_time = UPDATE_INTERVAL / SAMPLING_RATE / speed
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
    lf_path = DATA_DIR / f"{selected_player}_LF.parquet"
    rf_path = DATA_DIR / f"{selected_player}_RF.parquet"
    asyncio.run(stream_imu_data(lf_path, rf_path, selected_sensors, window_size, speed_multiplier, start_time))

if stop_stream:
    st.session_state.stop_streaming = True
    status.warning("Stopping stream...")

if not selected_sensors:
    st.warning("Please select at least one sensor to display")
