"""Configuration settings for IMU streaming application."""

from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class StreamConfig:
    """Configuration for data streaming and processing."""
    
    DATA_DIR: Path = Path("data/raw/imu")
    UPDATE_INTERVAL: int = 25  # Update chart every N samples
    SAMPLING_RATE: int = 256  # Hz
    DOWNSAMPLE_FACTOR: int = 2  # Display every Nth point
    DEFAULT_WINDOW_SIZE: int = 2000  # Number of samples in rolling window
    DEFAULT_SPEED: float = 1.0  # Playback speed multiplier
    RANGE_CALCULATION_WINDOW: int = 30  # seconds - for y-axis range calculation
    Y_AXIS_PADDING: float = 0.3  # Padding around data range (30%)
    GYRO_MIN_RANGE: float = 400.0  # Minimum y-axis range for gyroscope
    ACCEL_MIN_RANGE: float = 2.0  # Minimum y-axis range for accelerometer
    DYNAMIC_RANGE_EXPANSION: float = 0.1  # Expand range by 10% when exceeded
    FILTER_WINDOW_SIZE: int = 15  # Samples used in moving average filter


@dataclass
class UIConfig:
    """Configuration for UI elements and styling."""
    
    LOGO_PATH: Path = Path("data/images/logo.webp")
    CHART_HEIGHT: int = 200
    CHART_LINE_WIDTH: float = 1.5
    CHART_MARGIN: dict = field(default_factory=lambda: dict(l=20, r=20, t=10, b=20))
    CHART_COLORS: dict = field(default_factory=lambda: {
        'left_foot': '#d68032',   # Orange
        'right_foot': '#2a9d8f'   # Teal
    })
    DEFAULT_START_TIME: float = 2170.0  # Default start time in seconds
    TIME_STEP: float = 10.0  # Time step for number input
