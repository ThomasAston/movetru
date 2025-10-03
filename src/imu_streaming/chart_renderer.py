"""Chart rendering utilities for IMU data visualization."""

import plotly.graph_objects as go
from typing import List, Tuple

from .config import UIConfig


class ChartRenderer:
    """Handles creation and styling of Plotly charts for IMU data."""
    
    def __init__(self, ui_config: UIConfig):
        """
        Initialize the chart renderer.
        
        Args:
            ui_config: UI configuration object
        """
        self.config = ui_config
    
    def create_sensor_chart(
        self, 
        times: List[float], 
        values: List[float],
        y_range: List[float], 
        sensor_name: str,
        foot: str
    ) -> go.Figure:
        """
        Create a Plotly chart for a single sensor.
        
        Args:
            times: List of time values (x-axis)
            values: List of sensor values (y-axis)
            y_range: Y-axis range [min, max]
            sensor_name: Name of the sensor
            foot: Foot identifier ('LF' or 'RF')
            
        Returns:
            Plotly Figure object
        """
        color = (
            self.config.CHART_COLORS['left_foot'] 
            if foot == 'LF' 
            else self.config.CHART_COLORS['right_foot']
        )
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times,
            y=values,
            mode='lines',
            line=dict(color=color, width=self.config.CHART_LINE_WIDTH),
            name=f"{foot} {sensor_name}"
        ))
        
        fig.update_layout(
            height=self.config.CHART_HEIGHT,
            margin=self.config.CHART_MARGIN,
            xaxis_title="Time (s)",
            yaxis=dict(range=y_range, fixedrange=True),
            showlegend=False,
            transition={'duration': 0},
            uirevision='constant'
        )
        
        return fig
    
    def downsample_data(
        self, 
        times: List[float], 
        values: List[float], 
        factor: int
    ) -> Tuple[List[float], List[float]]:
        """
        Downsample data for display performance.
        
        Args:
            times: List of time values
            values: List of sensor values
            factor: Downsampling factor (keep every Nth point)
            
        Returns:
            Tuple of (downsampled_times, downsampled_values)
        """
        return times[::factor], values[::factor]
