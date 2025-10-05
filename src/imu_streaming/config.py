"""Configuration settings for IMU streaming application."""

from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class StreamConfig:
    """Configuration for data streaming and processing."""
    
    # Data and sampling
    DATA_DIR: Path = Path("data/raw/imu")
    SAMPLING_RATE: int = 256  # Hz
    
    # Display and performance
    UPDATE_INTERVAL: int = SAMPLING_RATE // 10  # Update chart every N samples
    DOWNSAMPLE_FACTOR: int = 5  # Display every Nth point
    DEFAULT_WINDOW_SIZE: int = SAMPLING_RATE * 10  # 10 seconds
    DEFAULT_SPEED: float = 1.0  # Playback speed multiplier (1 = real-time)
    
    # Y-axis range configuration
    RANGE_CALCULATION_WINDOW: int = 30  # Seconds for y-axis range calculation
    Y_AXIS_PADDING: float = 0.3  # Padding around data range (30%)
    GYRO_MIN_RANGE: float = 400.0  # Minimum y-axis range for gyroscope
    ACCEL_MIN_RANGE: float = 2.0  # Minimum y-axis range for accelerometer
    DYNAMIC_RANGE_EXPANSION: float = 0.1  # Expand range by 10% when exceeded
    
    # Signal processing
    FILTER_WINDOW_SIZE: int = 15  # Samples used in moving average filter
    
    # Gait detection parameters
    MSW_THRESHOLD: float = -115.0  # Mid-swing threshold (deg/s)
    ZC_THRESHOLD: float = 0.0  # Zero-crossing threshold (deg/s)
    MAX_STRIDE_TIME: float = 2.5  # Maximum valid stride duration (seconds)
    MIN_STRIDE_TIME: float = 0.1  # Minimum valid stride duration (seconds)
    METRICS_WINDOW: float = 10.0  # Time window for recent metrics (seconds)


@dataclass
class UIConfig:
    """Configuration for UI elements and styling."""
    
    LOGO_PATH: Path = Path("data/images/logo.webp")
    CHART_HEIGHT: int = 500  # Taller for stacked layout with legend
    CHART_LINE_WIDTH: float = 1.5
    CHART_MARGIN: dict = field(default_factory=lambda: dict(l=50, r=20, t=60, b=50))  # Extra top margin for legend
    CHART_COLORS: dict = field(default_factory=lambda: {
        'left_foot': '#d68032',   # Orange
        'right_foot': '#2a9d8f'   # Teal
    })
    DEFAULT_START_TIME: float = 2400.0  # Default start time in seconds
    TIME_STEP: float = 10.0  # Time step for number input
