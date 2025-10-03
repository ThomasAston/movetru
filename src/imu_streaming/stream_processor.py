"""Core streaming logic for processing IMU data."""

from collections import deque, defaultdict
from functools import partial
from typing import Dict, List

import polars as pl

from .config import StreamConfig


class IMUStreamProcessor:
    """Processes IMU data streams with rolling windows and dynamic range calculation."""

    def __init__(self, config: StreamConfig):
        """Initialize the stream processor."""
        self.config = config
        self.windows_lf = None
        self.windows_rf = None
        self.filtered_windows_lf = None
        self.filtered_windows_rf = None
        self.filter_buffers_lf = None
        self.filter_buffers_rf = None
        self.time_window_lf = None
        self.time_window_rf = None
        self.y_ranges_lf: Dict[str, List[float]] = {}
        self.y_ranges_rf: Dict[str, List[float]] = {}

    def initialize_windows(self, window_size: int, sensors: List[str]):
        """Initialize rolling windows for each sensor."""
        self.windows_lf = defaultdict(partial(deque, maxlen=window_size))
        self.windows_rf = defaultdict(partial(deque, maxlen=window_size))
        self.filtered_windows_lf = defaultdict(partial(deque, maxlen=window_size))
        self.filtered_windows_rf = defaultdict(partial(deque, maxlen=window_size))
        self.filter_buffers_lf = defaultdict(partial(deque, maxlen=self.config.FILTER_WINDOW_SIZE))
        self.filter_buffers_rf = defaultdict(partial(deque, maxlen=self.config.FILTER_WINDOW_SIZE))
        self.time_window_lf = deque(maxlen=window_size)
        self.time_window_rf = deque(maxlen=window_size)

        # Reset y-ranges for the sensors we're tracking during this session
        for sensor in sensors:
            self.y_ranges_lf[sensor] = [0.0, 1.0]
            self.y_ranges_rf[sensor] = [0.0, 1.0]

    def calculate_y_ranges(
        self,
        df_lf: pl.DataFrame,
        df_rf: pl.DataFrame,
        sensors: List[str],
        start_idx: int,
        end_idx: int,
    ):
        """Calculate stable y-axis ranges using the filtered signal."""

        for sensor in sensors:
            values_lf = df_lf[sensor][start_idx:end_idx].to_list()
            filtered_lf = self._preview_filtered_series(values_lf)
            self.y_ranges_lf[sensor] = self._calculate_sensor_range(filtered_lf, sensor)

            values_rf = df_rf[sensor][start_idx:end_idx].to_list()
            filtered_rf = self._preview_filtered_series(values_rf)
            self.y_ranges_rf[sensor] = self._calculate_sensor_range(filtered_rf, sensor)

    def _calculate_sensor_range(self, values: List[float], sensor: str) -> List[float]:
        """Calculate range for a single sensor with padding."""
        if not values:
            return [0.0, 1.0]

        data_min, data_max = min(values), max(values)
        data_range = data_max - data_min

        min_range = (
            self.config.GYRO_MIN_RANGE
            if sensor.startswith("Gyro")
            else self.config.ACCEL_MIN_RANGE
        )

        if data_range < min_range:
            center = (data_max + data_min) / 2
            return [center - min_range / 2, center + min_range / 2]

        padding = data_range * self.config.Y_AXIS_PADDING
        return [data_min - padding, data_max + padding]

    def update_windows(self, row_lf: dict, row_rf: dict, sensors: List[str]):
        """Update rolling windows with new data and filtered values."""

        self.time_window_lf.append(row_lf["Time"])
        self.time_window_rf.append(row_rf["Time"])

        for sensor in sensors:
            lf_value = row_lf[sensor]
            rf_value = row_rf[sensor]

            self.windows_lf[sensor].append(lf_value)
            self.windows_rf[sensor].append(rf_value)

            filtered_lf = self._filter_sample(self.filter_buffers_lf[sensor], lf_value)
            filtered_rf = self._filter_sample(self.filter_buffers_rf[sensor], rf_value)
            self.filtered_windows_lf[sensor].append(filtered_lf)
            self.filtered_windows_rf[sensor].append(filtered_rf)

    def get_current_data(self, sensor: str) -> Dict:
        """Get current time-aligned, filtered data for a sensor."""

        return {
            "lf_times": list(self.time_window_lf),
            "lf_values": list(self.filtered_windows_lf[sensor]),
            "rf_times": list(self.time_window_rf),
            "rf_values": list(self.filtered_windows_rf[sensor]),
            "lf_range": self.y_ranges_lf[sensor].copy(),
            "rf_range": self.y_ranges_rf[sensor].copy(),
        }

    def update_y_ranges(self, sensor: str, values_lf: List[float], values_rf: List[float]):
        """Dynamically expand y-axis ranges if current data exceeds them."""

        if not values_lf or not values_rf:
            return

        # Left foot range adjustment
        current_min_lf = min(values_lf)
        current_max_lf = max(values_lf)
        y_min_lf, y_max_lf = self.y_ranges_lf[sensor]

        if current_min_lf < y_min_lf:
            expansion = abs(current_min_lf) * self.config.DYNAMIC_RANGE_EXPANSION
            self.y_ranges_lf[sensor][0] = current_min_lf - expansion
        if current_max_lf > y_max_lf:
            expansion = abs(current_max_lf) * self.config.DYNAMIC_RANGE_EXPANSION
            self.y_ranges_lf[sensor][1] = current_max_lf + expansion

        # Right foot range adjustment
        current_min_rf = min(values_rf)
        current_max_rf = max(values_rf)
        y_min_rf, y_max_rf = self.y_ranges_rf[sensor]

        if current_min_rf < y_min_rf:
            expansion = abs(current_min_rf) * self.config.DYNAMIC_RANGE_EXPANSION
            self.y_ranges_rf[sensor][0] = current_min_rf - expansion
        if current_max_rf > y_max_rf:
            expansion = abs(current_max_rf) * self.config.DYNAMIC_RANGE_EXPANSION
            self.y_ranges_rf[sensor][1] = current_max_rf + expansion

    def _filter_sample(self, buffer: deque, new_value: float) -> float:
        """Apply a causal moving average filter to the incoming sample."""

        buffer.append(new_value)

        if not buffer:
            return new_value

        return float(sum(buffer) / len(buffer))

    def _preview_filtered_series(self, values: List[float]) -> List[float]:
        """Apply the moving average filter to a list of values for preview calculations."""

        if not values:
            return []

        buffer = deque(maxlen=self.config.FILTER_WINDOW_SIZE)
        filtered: List[float] = []

        for value in values:
            buffer.append(value)
            filtered.append(float(sum(buffer) / len(buffer)))

        return filtered
    