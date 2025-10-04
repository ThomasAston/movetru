# movetru
Repository for Stage 2 Interview – Sports Biomechanics Data Scientist Task

All files relating to the task are contained in this repository:
- `data`: Contains raw and processed data including: IMU, motion capture, event data, as well as other files used for submission such as images and reading materials.
- `pages`: Streamlit app files for interactive data visualization and analysis.
- `presentation`: slides and materials for the presentation.
- `src`: Source code for the IMU streaming and gait detection modules.
- `utils`: basic Python scripts for data processing and analysis.
- `workbooks`: Jupyter notebooks used for data exploration and analysis.

A working demo of the Streamlit app can be found at: [movetru.streamlit.app](https://movetru.streamlit.app)

## Real-Time Gait Detection

The main Streamlit app (`app.py`) now includes real-time stride event detection:

### Features
- **Real-time Mid-Swing (MSW) detection**: Black X markers appear as events are detected
- **Retrospective Foot Strike (FS) detection**: Black circle markers (slight delay)
- **Retrospective Foot Off (FO) detection**: Black triangle-up markers (slight delay)
- **Live metrics display**: Shows recent (5s window) and overall session statistics
  - Stride count
  - Stance time (mean ± std)
  - Swing time (mean ± std)
  - Stride time (mean ± std)

### Running the App

```bash
# Activate the conda environment
conda activate movetru

# Run the Streamlit app
streamlit run app.py
```

### Testing

Test the gait detector independently:

```bash
python test_gait_detector.py
```

### Algorithm Details

See `GAIT_DETECTION_INTEGRATION.md` for complete documentation of the algorithm implementation, parameters, and integration details.

The algorithm is based on:
- Brasiliano et al., 2023 - Mid-swing detection using gyroscope data
- Hsu et al., 2014 - Stride event detection methodology