# Gait Detection Integration Summary

## Overview

The real-time gait event detection algorithm from `demo_realtime.ipynb` has been successfully integrated into the Streamlit app. The algorithm detects stride events (Mid-Swing, Foot Strike, Foot Off) from gyroscope data in near real-time.

## Changes Made

### 1. New Module: `src/imu_streaming/gait_detector.py`

Created a production-ready `GaitEventDetector` class that implements:

- **Real-time Mid-Swing (MSW) Detection**: Detects local minima after descending zero-crossings
- **Retrospective Foot Strike (FS) Detection**: Maximum angular velocity between MSW and mid-stance
- **Retrospective Foot Off (FO) Detection**: Maximum angular velocity between mid-stance and next MSW
- **Metrics Calculation**: Computes stance time, swing time, and stride time with statistics

**Key Features:**
- Causal filtering (moving average)
- State management for both feet independently
- Windowed metrics (last 5s) and overall session metrics
- Proper timestamp tracking

### 2. Updated `src/imu_streaming/chart_renderer.py`

Enhanced the `create_sensor_chart` method to:
- Accept optional `events` parameter
- Plot MSW events as black X markers
- Plot FS events as black circle markers
- Plot FO events as black triangle-up markers
- Automatically match event times to signal values within the current window

### 3. Updated `src/imu_streaming/config.py`

Added gait detection parameters to `StreamConfig`:
```python
MSW_THRESHOLD: float = -115.0      # Mid-swing threshold (deg/s)
ZC_THRESHOLD: float = 0.0          # Zero-crossing threshold (deg/s)
MAX_STRIDE_TIME: float = 2.5       # Maximum valid stride duration (seconds)
MIN_STRIDE_TIME: float = 0.1       # Minimum valid stride duration (seconds)
METRICS_WINDOW: float = 5.0        # Time window for recent metrics (seconds)
```

### 4. Updated `app.py`

Integrated gait detection into the streaming loop:

**Initialization:**
- Create `GaitEventDetector` instance with configured parameters
- Add metrics display placeholders (recent + overall for both feet)

**Streaming Loop:**
- Process each sample through `gait_detector.process_sample()`
- Retrieve all detected events with `get_all_events()`
- Pass events to chart renderer for visualization
- Calculate and display metrics (recent window + overall session)
- Show event counts in status bar

### 5. Updated `src/imu_streaming/__init__.py`

Added `GaitEventDetector` to exports.

### 6. Documentation in Notebook

Added a new markdown cell at the end of `demo_realtime.ipynb` documenting the parameters used in the Streamlit app.

## Parameters Adapted for 256 Hz Sampling Rate

The notebook uses 100 Hz CSV data, while the Streamlit app uses 256 Hz parquet data. Parameters have been adjusted accordingly:

| Parameter | Notebook (100 Hz) | Streamlit (256 Hz) | Notes |
|-----------|-------------------|-------------------|-------|
| MSW Threshold | -115 deg/s | -115 deg/s | Same threshold |
| ZC Threshold | 0 deg/s | 0 deg/s | Same threshold |
| MA Window | 5 samples | 15 samples | Adjusted for sampling rate |
| Max Buffer | 500 samples (5s) | 1280 samples (5s) | 5 seconds worth |
| Max Stride Time | 2.5 s | 2.5 s | Same |
| Min Stride Time | 0.05 s | 0.1 s | Slightly adjusted |

## Visualization Features

### Real-Time Events (MSW)
- Mid-swing events are detected immediately and plotted as black X markers
- Appear on the chart as soon as they're detected

### Retrospective Events (FS, FO)
- Foot strike and foot off events require two consecutive MSWs to detect
- Slight delay (~1 stride) before appearing on the chart
- Plotted at the correct timestamp once detected

### Metrics Display

Two columns showing:

**Recent Metrics (Last 5 seconds):**
- Left foot: Strides, Stance time, Swing time, Stride time
- Right foot: Strides, Stance time, Swing time, Stride time

**Overall Session Metrics:**
- Left foot: Complete session statistics
- Right foot: Complete session statistics

All time metrics show mean ± std deviation.

## Algorithm Details

The detection algorithm follows this sequence:

1. **Sample-by-sample Processing:**
   - Apply moving average filter (causal)
   - Track zero-crossings (positive to negative)
   - Detect local minima after zero-crossings

2. **Mid-Swing Detection:**
   - After descending zero-crossing
   - Find local minimum below threshold
   - Immediately report event

3. **Retrospective Event Detection:**
   - When new MSW detected, search previous MSW→current MSW window
   - Detect mid-stance (30-60% of cycle, minimum of |signal|)
   - Detect FS (max between previous MSW and mid-stance)
   - Detect FO (max between mid-stance and current MSW)

4. **Metrics Calculation:**
   - Stance time = FO time - FS time
   - Swing time = next FS time - FO time
   - Stride time = MSW to next MSW

## Testing

To test the implementation:

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Select a player and start the stream

3. Verify:
   - MSW events appear in real-time (black X)
   - FS and FO events appear after ~1 stride (circles and triangles)
   - Metrics update continuously
   - Recent metrics show last 5 seconds
   - Overall metrics show cumulative statistics

## Future Enhancements

Potential improvements:
- Add mid-stance visualization (currently used only for detection)
- Add event count summary per foot
- Export detected events to CSV
- Add configurable thresholds in UI
- Add cadence calculation
- Add asymmetry metrics (left vs right comparison)
