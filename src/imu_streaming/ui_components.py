"""UI components for the Streamlit IMU streaming application."""

import streamlit as st
from PIL import Image
from typing import Tuple, Optional

from .config import UIConfig
from .signal_filters import FilterConfig, FilterType


class IMUStreamUI:
    """Handles rendering of UI components for the IMU streaming app."""
    
    def __init__(self, ui_config: UIConfig):
        """
        Initialize the UI component manager.
        
        Args:
            ui_config: UI configuration object
        """
        self.config = ui_config
    
    def render_header(self):
        """Render app header with logo and title."""
        logo_img = Image.open(self.config.LOGO_PATH)
        st.image(logo_img, width=logo_img.width // 2)
        st.title("Real-time stride event detection")
    
    def render_player_selector(self, players: list) -> Optional[str]:
        """
        Render player selection dropdown.
        
        Args:
            players: List of available player IDs
            
        Returns:
            Selected player ID or None
        """
        return st.selectbox(
            "Select Player", 
            players, 
            index=0 if players else None
        )
    
    def render_stream_controls(
        self, 
        default_start_time: Optional[float] = None
    ) -> Tuple[float, bool, bool]:
        """
        Render streaming control inputs.
        
        Args:
            default_start_time: Default start time in seconds
            
        Returns:
            Tuple of (start_time, start_clicked, stop_clicked)
        """
        if default_start_time is None:
            default_start_time = self.config.DEFAULT_START_TIME
        
        start_time = st.number_input(
            "Start from time (seconds)", 
            min_value=0.0, 
            value=default_start_time, 
            step=self.config.TIME_STEP,
            help="Choose which time (in seconds) to start streaming from"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            start = st.button("▶ Start Stream")
        with col2:
            stop = st.button("⏹ Stop Stream")
        
        return start_time, start, stop
    
    def create_chart_placeholders(
        self, 
        sensors: list
    ) -> Tuple[Optional[st.delta_generator.DeltaGenerator], 
               Optional[st.delta_generator.DeltaGenerator]]:
        """
        Create placeholders for charts.
        
        Args:
            sensors: List of selected sensors
            
        Returns:
            Tuple of (left_foot_chart, right_foot_chart) or (None, None)
        """
        if not sensors:
            st.warning("Please select at least one sensor to display")
            return None, None
        
        st.subheader("Live Data")
        st.markdown("**Left Foot (LF)**")
        chart_lf = st.empty()
        st.markdown("**Right Foot (RF)**")
        chart_rf = st.empty()
        
        return chart_lf, chart_rf
    
    def create_status_placeholder(self) -> st.delta_generator.DeltaGenerator:
        """
        Create a placeholder for status messages.
        
        Returns:
            Streamlit empty placeholder
        """
        return st.empty()
    
    def render_filter_controls(self, sampling_rate: int = 256) -> Optional[FilterConfig]:
        """
        Render filter configuration controls.
        
        Args:
            sampling_rate: Sampling rate in Hz
            
        Returns:
            FilterConfig object or None if no filtering
        """
        enable_filter = st.checkbox(
            "Enable signal smoothing",
            value=True,
            help="4th order low-pass Butterworth filter (based on FFT analysis)"
        )
        
        if not enable_filter:
            return FilterConfig(filter_type='none')
        
        # Use the optimal parameters from your notebook analysis
        return FilterConfig(
            filter_type='butterworth',
            cutoff_freq=12.6,  # Hz - preserves top 3 harmonics
            filter_order=4,
            sampling_rate=sampling_rate
        )
