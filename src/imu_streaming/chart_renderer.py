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
            
            # Mid-swing events (small black X)
            if 'msw' in events and events['msw']:
                msw_x, msw_y = get_y_at_times(events['msw'])
                if msw_x:
                    fig.add_trace(go.Scatter(
                        x=msw_x,
                        y=msw_y,
                        mode='markers',
                        marker=dict(
                            symbol='x', 
                            size=6,  # Smaller size
                            color='rgba(0, 0, 0, 0.5)',  # More transparent
                            line=dict(width=1)  # Thinner line
                        ),
                        name='MSW',
                        showlegend=False,
                        hoverinfo='skip'
                    ))
            
            # Foot strike events (small black circle)
            if 'fs' in events and events['fs']:
                fs_x, fs_y = get_y_at_times(events['fs'])
                if fs_x:
                    fig.add_trace(go.Scatter(
                        x=fs_x,
                        y=fs_y,
                        mode='markers',
                        marker=dict(
                            symbol='circle', 
                            size=5,  # Smaller size
                            color='rgba(0, 0, 0, 0.5)',  # More transparent
                            line=dict(width=0)
                        ),
                        name='FS',
                        showlegend=False,
                        hoverinfo='skip'
                    ))
            
            # Foot off events (small black triangle-up)
            if 'fo' in events and events['fo']:
                fo_x, fo_y = get_y_at_times(events['fo'])
                if fo_x:
                    fig.add_trace(go.Scatter(
                        x=fo_x,
                        y=fo_y,
                        mode='markers',
                        marker=dict(
                            symbol='triangle-up', 
                            size=5,  # Smaller size
                            color='rgba(0, 0, 0, 0.5)',  # More transparent
                            line=dict(width=0)
                        ),
                        name='FO',
                        showlegend=False,
                        hoverinfo='skip'
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
            dragmode=False,
            # Additional settings to reduce juddering
            plot_bgcolor='white',
            paper_bgcolor='white',
            # Disable all animations
            newshape=dict(line_color='black'),
        )
        
        # Disable animations and transitions for smoother updates
        fig.update_xaxes(
            fixedrange=True,
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='lightgray',
            # Disable axis animations
            type='linear'
        )
        fig.update_yaxes(
            fixedrange=True,
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='lightgray',
            # Disable axis animations
            type='linear'
        )
        
        # Disable trace animations
        for trace in fig.data:
            trace.update(showlegend=False)
        
        return fig

    def create_combined_chart(
        self,
        times_lf: List[float],
        values_lf: List[float],
        times_rf: List[float],
        values_rf: List[float],
        y_range: List[float],
        sensor_name: str,
        events_lf: Optional[Dict[str, List[float]]] = None,
        events_rf: Optional[Dict[str, List[float]]] = None
    ) -> go.Figure:
        """
        Create a stacked subplot figure for left and right foot with shared y-axis.
        
        Args:
            times_lf: Left foot time values
            values_lf: Left foot sensor values
            times_rf: Right foot time values
            values_rf: Right foot sensor values
            y_range: Shared y-axis range [min, max]
            sensor_name: Name of the sensor
            events_lf: Optional left foot events
            events_rf: Optional right foot events
            
        Returns:
            Plotly Figure with subplots
        """
        from plotly.subplots import make_subplots
        
        # Create stacked subplots with shared x-axis
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=("Left Foot", "Right Foot"),
            vertical_spacing=0.12,
            row_heights=[0.5, 0.5]
        )
        
        # Add left foot signal
        fig.add_trace(
            go.Scatter(
                x=times_lf,
                y=values_lf,
                mode='lines',
                line=dict(color=self.config.CHART_COLORS['left_foot'], width=self.config.CHART_LINE_WIDTH),
                name="Signal",
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add right foot signal
        fig.add_trace(
            go.Scatter(
                x=times_rf,
                y=values_rf,
                mode='lines',
                line=dict(color=self.config.CHART_COLORS['right_foot'], width=self.config.CHART_LINE_WIDTH),
                name="Signal",
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add placeholder legend traces (invisible, just for legend)
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(symbol='x-thin', size=8, color='rgba(0, 0, 0, 0.4)', line=dict(width=1)),
                name='Mid-Swing (MSW)',
                legendgroup='msw',
                showlegend=True
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(symbol='circle', size=7, color='rgba(0, 0, 0, 0.4)'),
                name='Foot Strike (FS)',
                legendgroup='fs',
                showlegend=True
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(symbol='triangle-up', size=7, color='rgba(0, 0, 0, 0.4)'),
                name='Foot Off (FO)',
                legendgroup='fo',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Add left foot event markers (don't show in legend, use legendgroup)
        if events_lf and times_lf:
            time_min_lf, time_max_lf = min(times_lf), max(times_lf)
            self._add_event_markers(fig, events_lf, times_lf, values_lf, time_min_lf, time_max_lf, row=1, col=1, show_legend=False)
        
        # Add right foot event markers (don't show in legend, use legendgroup)
        if events_rf and times_rf:
            time_min_rf, time_max_rf = min(times_rf), max(times_rf)
            self._add_event_markers(fig, events_rf, times_rf, values_rf, time_min_rf, time_max_rf, row=2, col=1, show_legend=False)
        
        # Update layout
        fig.update_layout(
            height=self.config.CHART_HEIGHT,
            margin=self.config.CHART_MARGIN,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.08,
                xanchor="left",
                x=0,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(200, 200, 200, 0.5)",
                borderwidth=1
            ),
            transition={'duration': 0},
            uirevision='constant',
            hovermode=False,
            dragmode=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
        )
        
        # Update x-axes (only show title on bottom plot)
        fig.update_xaxes(
            fixedrange=True,
            showgrid=True,
            gridcolor='rgba(220, 220, 220, 0.3)',  # Very light grid
            zeroline=False,
            type='linear'
        )
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        
        # Update y-axes with shared range
        fig.update_yaxes(
            title_text="Ang. velocity (deg/s)",
            range=y_range,
            fixedrange=True,
            showgrid=True,
            gridcolor='rgba(220, 220, 220, 0.3)',  # Very light grid
            zeroline=True,
            zerolinecolor='rgba(200, 200, 200, 0.5)',
            zerolinewidth=1,
            type='linear'
        )
        
        return fig
    
    def _add_event_markers(
        self,
        fig: go.Figure,
        events: Dict[str, List[float]],
        times: List[float],
        values: List[float],
        time_min: float,
        time_max: float,
        row: int,
        col: int,
        show_legend: bool = False
    ):
        """Helper to add event markers to a subplot."""
        
        def get_y_at_times(event_times):
            y_vals = []
            x_vals = []
            for t in event_times:
                if time_min <= t <= time_max:
                    idx = min(range(len(times)), key=lambda i: abs(times[i] - t))
                    x_vals.append(times[idx])
                    y_vals.append(values[idx])
            return x_vals, y_vals
        
        # Mid-swing events (refined X marker)
        if 'msw' in events and events['msw']:
            msw_x, msw_y = get_y_at_times(events['msw'])
            if msw_x:
                fig.add_trace(
                    go.Scatter(
                        x=msw_x,
                        y=msw_y,
                        mode='markers',
                        marker=dict(
                            symbol='x-thin',  # Thinner X
                            size=8,
                            color='rgba(0, 0, 0, 0.4)',
                            line=dict(width=1)
                        ),
                        name='Mid-Swing (MSW)',
                        legendgroup='msw',
                        showlegend=show_legend,
                        hoverinfo='skip'
                    ),
                    row=row, col=col
                )
        
        # Foot strike events (small circle)
        if 'fs' in events and events['fs']:
            fs_x, fs_y = get_y_at_times(events['fs'])
            if fs_x:
                fig.add_trace(
                    go.Scatter(
                        x=fs_x,
                        y=fs_y,
                        mode='markers',
                        marker=dict(
                            symbol='circle',
                            size=7,
                            color='rgba(0, 0, 0, 0.4)',
                            line=dict(width=0)
                        ),
                        name='Foot Strike (FS)',
                        legendgroup='fs',
                        showlegend=show_legend,
                        hoverinfo='skip'
                    ),
                    row=row, col=col
                )
        
        # Foot off events (small triangle)
        if 'fo' in events and events['fo']:
            fo_x, fo_y = get_y_at_times(events['fo'])
            if fo_x:
                fig.add_trace(
                    go.Scatter(
                        x=fo_x,
                        y=fo_y,
                        mode='markers',
                        marker=dict(
                            symbol='triangle-up',
                            size=7,
                            color='rgba(0, 0, 0, 0.4)',
                            line=dict(width=0)
                        ),
                        name='Foot Off (FO)',
                        legendgroup='fo',
                        showlegend=show_legend,
                        hoverinfo='skip'
                    ),
                    row=row, col=col
                )

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
