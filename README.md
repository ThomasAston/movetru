# movetru
Repository for Stage 2 Interview – Sports Biomechanics Data Scientist Task

All files relating to the task are contained in this repository:
- `data`: Contains raw and processed data including: IMU, motion capture, event data, as well as other files used for submission such as images and reading materials.
- `presentation`: slides and materials for the presentation.
- `src`: Source code for the IMU streaming and gait detection modules.
- `utils`: basic Python scripts for initial data processing and analysis.
- `workbooks`: Jupyter notebooks used for data exploration and analysis.

A working demo of the Streamlit app can be found at: [movetru-stride.streamlit.app](https://movetru-stride.streamlit.app/)

## Real-Time Gait Detection

The main Streamlit app (`app.py`) includes real-time stride event detection:

### Running the app locally

The online version can be slow, so to run the app locally, clone the repository and run the following commands in your terminal:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

### Algorithm Details

The sample data used throughout this work comes from the dataset provided by [Grouvel et al., 2023](https://www-nature-com.eux.idm.oclc.org/articles/s41597-023-02077-3). 

The event detection algorithms are inspired by the works of:
- [Falbriard et al., 2018](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2018.00610/full)
- [Gouda and Andrysek, 2022](https://www.mdpi.com/1424-8220/22/22/8888)