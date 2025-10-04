"""Chart rendering utilities for IMU data visualization."""

import plotly.graph_objects as go
from typing import List, Tuple, Dict, Optional

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
        foot: str,
        events: Optional[Dict[str, List[float]]] = None
    ) -> go.Figure:
        """
        Create a Plotly chart for a single sensor with optional gait events.
        
        Args:
            times: List of time values (x-axis)
            values: List of sensor values (y-axis)
            y_range: Y-axis range [min, max]
            sensor_name: Name of the sensor
            foot: Foot identifier ('LF' or 'RF')
            events: Optional dictionary of event times {'msw': [...], 'fs': [...], 'fo': [...]}
            
        Returns:
            Plotly Figure object
        """
        color = (
            self.config.CHART_COLORS['left_foot'] 
            if foot == 'LF' 
            else self.config.CHART_COLORS['right_foot']
        )
        
        fig = go.Figure()
        
        # Add main signal trace
        fig.add_trace(go.Scatter(
            x=times,
            y=values,
            mode='lines',
            line=dict(color=color, width=self.config.CHART_LINE_WIDTH),
            name=f"{foot} {sensor_name}"
        ))
        
        # Add gait event markers if provided
        if events and times:
            time_min, time_max = min(times), max(times)
            
            # Helper function to get y-values at event times
            def get_y_at_times(event_times):
                # Find corresponding y-values for event times within the current window
                y_vals = []
                x_vals = []
                for t in event_times:
                    if time_min <= t <= time_max:
                        # Find closest time index
                        idx = min(range(len(times)), key=lambda i: abs(times[i] - t))
                        x_vals.append(times[idx])
                        y_vals.append(values[idx])
                return x_vals, y_vals
            
            # Mid-swing events (black X)
            if 'msw' in events and events['msw']:
                msw_x, msw_y = get_y_at_times(events['msw'])
                if msw_x:
                    fig.add_trace(go.Scatter(
                        x=msw_x,
                        y=msw_y,
                        mode='markers',
                        marker=dict(symbol='x', size=10, color='black', line=dict(width=2)),
                        name='MSW',
                        showlegend=False
                    ))
            
            # Foot strike events (black circle)
            if 'fs' in events and events['fs']:
                fs_x, fs_y = get_y_at_times(events['fs'])
                if fs_x:
                    fig.add_trace(go.Scatter(
                        x=fs_x,
                        y=fs_y,
                        mode='markers',
                        marker=dict(symbol='circle', size=8, color='black'),
                        name='FS',
                        showlegend=False
                    ))
            
            # Foot off events (black triangle-up)
            if 'fo' in events and events['fo']:
                fo_x, fo_y = get_y_at_times(events['fo'])
                if fo_x:
                    fig.add_trace(go.Scatter(
                        x=fo_x,
                        y=fo_y,
                        mode='markers',
                        marker=dict(symbol='triangle-up', size=8, color='black'),
                        name='FO',
                        showlegend=False
                    ))
        
        fig.update_layout(
            height=self.config.CHART_HEIGHT,
            margin=self.config.CHART_MARGIN,
            xaxis_title="Time (s)",
            yaxis=dict(range=y_range, fixedrange=True),
            showlegend=False,
            transition={'duration': 0},
            uirevision='constant',
            # Reduce animations that can cause flickering
            hovermode=False,
            dragmode=False
        )
        
        # Disable animations for smoother updates
        fig.update_xaxes(fixedrange=True)
        fig.update_yaxes(fixedrange=True)
        
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
