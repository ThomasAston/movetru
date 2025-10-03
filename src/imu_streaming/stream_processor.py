"""Core streaming logic for processing IMU data."""

from collections import deque, defaultdict
from functools import partial
from typing import Dict, List, Optional
import polars as pl
import numpy as np

from .config import StreamConfig
from .signal_filters import StreamingFilter, FilterConfig


class IMUStreamProcessor:
    """Processes IMU data streams with rolling windows and dynamic range calculation."""
    
    def __init__(self, config: StreamConfig, filter_config: Optional[FilterConfig] = None):
        """
        Initialize the stream processor.
        
        Args:
            config: Stream configuration object
            filter_config: Optional filter configuration. If None, no filtering is applied.
        """
        self.config = config
        self.windows_lf = None
        self.windows_rf = None
        self.time_window_lf = None
        self.time_window_rf = None
        self.y_ranges_lf = {}
        self.y_ranges_rf = {}
        
        # Filtering support
        self.filter_config = filter_config
        self.filters_lf = {}  # Dict of sensor -> filter
        self.filters_rf = {}  # Dict of sensor -> filter
        self.filtering_enabled = filter_config is not None and filter_config.filter_type != 'none'
    
    def initialize_windows(self, window_size: int, sensors: List[str]):
        """
        Initialize rolling windows for each sensor.
        
        Args:
            window_size: Maximum number of samples in each window
            sensors: List of sensor names to track
        """
        self.windows_lf = defaultdict(partial(deque, maxlen=window_size))
        self.windows_rf = defaultdict(partial(deque, maxlen=window_size))
        self.time_window_lf = deque(maxlen=window_size)
        self.time_window_rf = deque(maxlen=window_size)
        
        # Initialize filters for each sensor if filtering is enabled
        if self.filtering_enabled:
            for sensor in sensors:
                self.filters_lf[sensor] = StreamingFilter(self.filter_config)
                self.filters_rf[sensor] = StreamingFilter(self.filter_config)
    
    def calculate_y_ranges(
        self, 
        df_lf: pl.DataFrame, 
        df_rf: pl.DataFrame,
        sensors: List[str], 
        start_idx: int, 
        end_idx: int
    ):
        """
        Calculate stable y-axis ranges from a data sample.
        
        Args:
            df_lf: Left foot DataFrame
            df_rf: Right foot DataFrame
            sensors: List of sensor names
            start_idx: Starting sample index
            end_idx: Ending sample index
        """
        for sensor in sensors:
            # Left foot
            values_lf = df_lf[sensor][start_idx:end_idx].to_list()
            self.y_ranges_lf[sensor] = self._calculate_sensor_range(values_lf, sensor)
            
            # Right foot
            values_rf = df_rf[sensor][start_idx:end_idx].to_list()
            self.y_ranges_rf[sensor] = self._calculate_sensor_range(values_rf, sensor)
    
    def _calculate_sensor_range(self, values: List[float], sensor: str) -> List[float]:
        """
        Calculate range for a single sensor with padding.
        
        Args:
            values: List of sensor values
            sensor: Sensor name (used to determine min range)
            
        Returns:
            List of [min_value, max_value] for y-axis range
        """
        if not values:
            return [0.0, 1.0]
        
        data_min, data_max = min(values), max(values)
        data_range = data_max - data_min
        
        # Determine minimum range based on sensor type
        min_range = (
            self.config.GYRO_MIN_RANGE 
            if sensor.startswith('Gyro') 
            else self.config.ACCEL_MIN_RANGE
        )
        
        if data_range < min_range:
            # Center the range if data range is too small
            center = (data_max + data_min) / 2
            return [center - min_range / 2, center + min_range / 2]
        
        # Add padding around the data range
        padding = data_range * self.config.Y_AXIS_PADDING
        return [data_min - padding, data_max + padding]
    
    def update_windows(self, row_lf: dict, row_rf: dict, sensors: List[str]):
        """
        Update rolling windows with new data.
        
        Args:
            row_lf: Left foot data row (dict with sensor values)
            row_rf: Right foot data row (dict with sensor values)
            sensors: List of sensor names to update
        """
        self.time_window_lf.append(row_lf['Time'])
        self.time_window_rf.append(row_rf['Time'])
        
        for sensor in sensors:
            # Apply filtering if enabled
            if self.filtering_enabled:
                lf_value = self.filters_lf[sensor].filter_sample(row_lf[sensor])
                rf_value = self.filters_rf[sensor].filter_sample(row_rf[sensor])
            else:
                lf_value = row_lf[sensor]
                rf_value = row_rf[sensor]
            
            self.windows_lf[sensor].append(lf_value)
            self.windows_rf[sensor].append(rf_value)
    
    def get_current_data(self, sensor: str) -> Dict:
        """
        Get current windowed data for a sensor.
        
        Args:
            sensor: Sensor name
            
        Returns:
            Dictionary with times, values, and ranges for both feet
        """
        lf_values = list(self.windows_lf[sensor])
        rf_values = list(self.windows_rf[sensor])
        
        return {
            'lf_times': list(self.time_window_lf),
            'lf_values': lf_values,
            'rf_times': list(self.time_window_rf),
            'rf_values': rf_values,
            'lf_range': self.y_ranges_lf[sensor].copy(),
            'rf_range': self.y_ranges_rf[sensor].copy()
        }
    
    def update_y_ranges(self, sensor: str, values_lf: List[float], values_rf: List[float]):
        """
        Dynamically expand y-axis ranges if current data exceeds them.
        
        Args:
            sensor: Sensor name
            values_lf: Current left foot values
            values_rf: Current right foot values
        """
        # Left foot range adjustment
        current_min = min(values_lf)
        current_max = max(values_lf)
        y_min_lf, y_max_lf = self.y_ranges_lf[sensor]
        
        if current_min < y_min_lf:
            expansion = abs(current_min) * self.config.DYNAMIC_RANGE_EXPANSION
            self.y_ranges_lf[sensor][0] = current_min - expansion
        if current_max > y_max_lf:
            expansion = abs(current_max) * self.config.DYNAMIC_RANGE_EXPANSION
            self.y_ranges_lf[sensor][1] = current_max + expansion
        
        # Right foot range adjustment
        current_min = min(values_rf)
        current_max = max(values_rf)
        y_min_rf, y_max_rf = self.y_ranges_rf[sensor]
        
        if current_min < y_min_rf:
            expansion = abs(current_min) * self.config.DYNAMIC_RANGE_EXPANSION
            self.y_ranges_rf[sensor][0] = current_min - expansion
        if current_max > y_max_rf:
            expansion = abs(current_max) * self.config.DYNAMIC_RANGE_EXPANSION
            self.y_ranges_rf[sensor][1] = current_max + expansion
    
    def get_filter_info(self) -> Optional[dict]:
        """
        Get information about the currently applied filter.
        
        Returns:
            Dictionary with filter details, or None if no filtering
        """
        if not self.filtering_enabled:
            return None
        
        # Get info from any filter (they're all configured the same)
        if self.filters_lf:
            sensor = list(self.filters_lf.keys())[0]
            return self.filters_lf[sensor].get_info()
        
        return None