"""Real-time gait event detection module."""

from collections import deque
from typing import Optional, Dict, List
import numpy as np


class GaitEventDetector:
    """
    Detects gait events (MSW, FS, MS, FO) in real-time from gyroscope data.
    Based on the algorithm from Brasiliano et al., 2023 and Hsu et al., 2014.
    """

    def __init__(
        self,
        fs: int = 256,
        msw_threshold: float = -115.0,
        zc_threshold: float = 0.0,
        ma_window: int = 15,
        max_buffer_size: int = None,
        max_stride_time: float = 2.5,
        min_stride_time: float = 0.1,
    ):
        """
        Initialize the gait event detector.

        Args:
            fs: Sampling frequency in Hz
            msw_threshold: Threshold for mid-swing detection (deg/s)
            zc_threshold: Threshold for zero-crossing detection (deg/s)
            ma_window: Moving average window size (samples)
            max_buffer_size: Maximum buffer size for smoothing (samples), defaults to 5 seconds
            max_stride_time: Maximum valid stride duration (seconds)
            min_stride_time: Minimum valid stride duration (seconds)
        """
        self.fs = fs
        self.msw_threshold = msw_threshold
        self.zc_threshold = zc_threshold
        self.ma_window = ma_window
        self.max_buffer_size = max_buffer_size or (fs * 5)  # Default: 5 seconds
        self.max_stride_time = max_stride_time
        self.min_stride_time = min_stride_time

        # State for both feet
        self.feet_state = {
            'left': self._init_foot_state(),
            'right': self._init_foot_state()
        }

    def _init_foot_state(self) -> Dict:
        """Initialize state for a single foot."""
        return {
            # Smoothing buffers
            'raw_buffer': deque(maxlen=self.max_buffer_size),
            'smoothed_history': [],  # Last 3 samples for MSW detection
            'all_smoothed': [],  # All smoothed values
            'all_times': [],  # Store all timestamps for mapping

            # MSW detection state
            'zc_prev': None,
            'after_desc_zc': False,
            'msw_indices': [],
            'msw_times': [],

            # Gait event detection state
            'previous_msw': None,
            'previous_msw_time': None,
            'mid_stance_indices': [],
            'mid_stance_times': [],
            'foot_strike_indices': [],
            'foot_strike_times': [],
            'foot_off_indices': [],
            'foot_off_times': [],

            # Sample counter
            'sample_idx': 0,
        }

    def reset(self):
        """Reset all internal state."""
        self.feet_state = {
            'left': self._init_foot_state(),
            'right': self._init_foot_state()
        }

    def process_sample(
        self, 
        lf_gyro_y: float, 
        rf_gyro_y: float,
        lf_time: float,
        rf_time: float
    ) -> Dict[str, Dict]:
        """
        Process a new sample and return any newly detected events.

        Args:
            lf_gyro_y: Left foot gyroscope Y value (deg/s)
            rf_gyro_y: Right foot gyroscope Y value (deg/s)
            lf_time: Left foot timestamp (seconds)
            rf_time: Right foot timestamp (seconds)

        Returns:
            Dictionary with 'left' and 'right' keys, each containing:
                - 'msw': list of newly detected MSW times
                - 'fs': list of newly detected FS times
                - 'fo': list of newly detected FO times
                - 'ms': list of newly detected MS times
        """
        events = {
            'left': {'msw': [], 'fs': [], 'fo': [], 'ms': []},
            'right': {'msw': [], 'fs': [], 'fo': [], 'ms': []}
        }

        foot_data = [
            ('left', lf_gyro_y, lf_time),
            ('right', rf_gyro_y, rf_time)
        ]

        for foot_name, raw_value, timestamp in foot_data:
            foot_state = self.feet_state[foot_name]

            # STEP 1: Update smoothing buffer
            foot_state['raw_buffer'].append(raw_value)

            # STEP 2: Compute smoothed value
            smoothed_value = self._moving_average(foot_state['raw_buffer'])
            foot_state['all_smoothed'].append(smoothed_value)
            foot_state['all_times'].append(timestamp)

            # STEP 3: MSW Detection
            foot_state['smoothed_history'].append(smoothed_value)
            if len(foot_state['smoothed_history']) > 3:
                foot_state['smoothed_history'].pop(0)

            # Detect zero-crossing
            foot_state['zc_prev'], is_desc_zc = self._detect_zero_crossing_descending(
                smoothed_value, foot_state['zc_prev']
            )

            if is_desc_zc:
                foot_state['after_desc_zc'] = True

            # Detect local minimum after descending zero-crossing
            if foot_state['after_desc_zc']:
                if self._detect_local_minimum(foot_state['smoothed_history']):
                    current_msw_idx = foot_state['sample_idx'] - 1
                    current_msw_time = timestamp
                    foot_state['msw_indices'].append(current_msw_idx)
                    foot_state['msw_times'].append(current_msw_time)
                    foot_state['after_desc_zc'] = False

                    # Add to events
                    events[foot_name]['msw'].append(current_msw_time)

                    # STEP 4: Retrospective FS/MS/FO Detection
                    if foot_state['previous_msw'] is not None:
                        prev_msw_idx = foot_state['previous_msw']
                        prev_msw_time = foot_state['previous_msw_time']

                        # Check if this forms a valid stride
                        if self._is_valid_stride(prev_msw_time, current_msw_time):
                            smoothed_signal = np.array(foot_state['all_smoothed'])
                            signal_norm = np.abs(smoothed_signal)

                            # Detect mid-stance
                            ms_idx = self._detect_mid_stance(
                                signal_norm, prev_msw_idx, current_msw_idx
                            )

                            if ms_idx is not None:
                                # Calculate time for mid-stance
                                ms_time = self._index_to_time(ms_idx, foot_state)
                                foot_state['mid_stance_indices'].append(ms_idx)
                                foot_state['mid_stance_times'].append(ms_time)
                                events[foot_name]['ms'].append(ms_time)

                                # Detect foot strike
                                fs_idx = self._detect_foot_strike(
                                    smoothed_signal, prev_msw_idx, ms_idx
                                )
                                if fs_idx is not None:
                                    fs_time = self._index_to_time(fs_idx, foot_state)
                                    foot_state['foot_strike_indices'].append(fs_idx)
                                    foot_state['foot_strike_times'].append(fs_time)
                                    events[foot_name]['fs'].append(fs_time)

                                # Detect foot off
                                fo_idx = self._detect_foot_off(
                                    smoothed_signal, ms_idx, current_msw_idx
                                )
                                if fo_idx is not None:
                                    fo_time = self._index_to_time(fo_idx, foot_state)
                                    foot_state['foot_off_indices'].append(fo_idx)
                                    foot_state['foot_off_times'].append(fo_time)
                                    events[foot_name]['fo'].append(fo_time)

                    # Update previous MSW
                    foot_state['previous_msw'] = current_msw_idx
                    foot_state['previous_msw_time'] = current_msw_time

            # Increment sample counter
            foot_state['sample_idx'] += 1

        return events

    def _moving_average(self, buffer: deque) -> float:
        """Compute the moving average of the buffer."""
        if len(buffer) < self.ma_window:
            return np.mean(buffer) if len(buffer) > 0 else 0.0
        return np.mean(list(buffer)[-self.ma_window:])

    def _detect_zero_crossing_descending(
        self, current_value: float, previous_state: Optional[bool]
    ) -> tuple[bool, bool]:
        """Detect descending zero-crossing (positive to negative)."""
        current_state = current_value >= self.zc_threshold
        is_descending_zc = False

        if previous_state is not None:
            if previous_state and not current_state:
                is_descending_zc = True

        return current_state, is_descending_zc

    def _detect_local_minimum(self, history: List[float]) -> bool:
        """Detect local minimum using 3-sample window."""
        if len(history) < 3:
            return False

        prev_sample = history[0]
        curr_sample = history[1]
        next_sample = history[2]

        is_local_min = (prev_sample >= curr_sample) and (curr_sample <= next_sample)
        below_threshold = curr_sample < self.msw_threshold

        return is_local_min and below_threshold

    def _is_valid_stride(self, msw1_time: float, msw2_time: float) -> bool:
        """Check if two consecutive mid-swings form a valid stride."""
        stride_time = msw2_time - msw1_time
        return self.min_stride_time < stride_time < self.max_stride_time

    def _detect_mid_stance(
        self, signal: np.ndarray, msw_start_idx: int, msw_end_idx: int
    ) -> Optional[int]:
        """Detect mid-stance as minimum in the 30-60% range between MSWs."""
        cycle_length = msw_end_idx - msw_start_idx
        start_search = msw_start_idx + int(0.3 * cycle_length)
        end_search = msw_start_idx + int(0.6 * cycle_length)

        if start_search >= end_search or end_search > msw_end_idx:
            return None

        search_segment = signal[start_search:end_search]
        min_idx_relative = np.argmin(search_segment)

        return start_search + min_idx_relative

    def _detect_foot_strike(
        self, signal: np.ndarray, msw_idx: int, mid_stance_idx: int
    ) -> Optional[int]:
        """Detect foot strike as maximum between MSW and mid-stance."""
        if msw_idx >= mid_stance_idx:
            return None

        search_segment = signal[msw_idx:mid_stance_idx]
        max_idx_relative = np.argmax(search_segment)

        return msw_idx + max_idx_relative

    def _detect_foot_off(
        self, signal: np.ndarray, mid_stance_idx: int, next_msw_idx: int
    ) -> Optional[int]:
        """Detect foot off as maximum between mid-stance and next MSW."""
        if mid_stance_idx >= next_msw_idx:
            return None

        search_segment = signal[mid_stance_idx:next_msw_idx]
        max_idx_relative = np.argmax(search_segment)

        return mid_stance_idx + max_idx_relative

    def _index_to_time(self, idx: int, foot_state: Dict) -> float:
        """
        Convert sample index to timestamp using stored timestamps.
        """
        if idx < len(foot_state['all_times']):
            return foot_state['all_times'][idx]
        # Fallback if index is out of range (shouldn't happen)
        return idx / self.fs

    def get_metrics(self, foot: str, window_seconds: Optional[float] = None) -> Dict:
        """
        Calculate gait metrics for a given foot.

        Args:
            foot: 'left' or 'right'
            window_seconds: If provided, only calculate metrics for the last N seconds.
                          If None, calculate for all data.

        Returns:
            Dictionary containing stride metrics
        """
        foot_state = self.feet_state[foot]

        fs_times = np.array(foot_state['foot_strike_times'])
        fo_times = np.array(foot_state['foot_off_times'])
        msw_times = np.array(foot_state['msw_times'])

        # Apply time window filter if requested
        # Use the most recent data timestamp, not the last event time
        # This ensures metrics go to zero when person is stationary
        if window_seconds is not None and len(foot_state['all_times']) > 0:
            current_time = foot_state['all_times'][-1]
            cutoff_time = current_time - window_seconds

            # Filter events that fall within the time window
            # Important: MSW events define stride boundaries, so we filter based on MSW times
            # to maintain proper event alignment
            msw_mask = msw_times >= cutoff_time
            
            # Find which stride index corresponds to the cutoff time
            # FS and FO events at index i belong to stride i (between MSW[i] and MSW[i+1])
            if np.any(msw_mask):
                first_kept_stride = np.where(msw_mask)[0][0]
                # Keep MSW events from cutoff onwards
                msw_times = msw_times[msw_mask]
                # Keep FS/FO events from the first kept stride onwards
                # This ensures FS and FO are properly paired within valid strides
                fs_times = fs_times[first_kept_stride:]
                fo_times = fo_times[first_kept_stride:]
            else:
                # No events in window
                msw_times = np.array([])
                fs_times = np.array([])
                fo_times = np.array([])

        metrics = {
            'total_strides': len(msw_times),
            'stance_time_mean': None,
            'stance_time_std': None,
            'swing_time_mean': None,
            'swing_time_std': None,
            'stride_time_mean': None,
            'stride_time_std': None,
            'cadence': None,
            'stride_time_cv': None,
            'stance_swing_ratio': None,
            'contact_time_percent': None,
        }

        # Calculate stance time (FS to FO)
        stance_times = None
        if len(fs_times) > 0 and len(fo_times) > 0:
            min_len = min(len(fs_times), len(fo_times))
            stance_times = fo_times[:min_len] - fs_times[:min_len]
            # Filter out any negative stance times (these indicate misaligned events)
            stance_times = stance_times[stance_times > 0]
            if len(stance_times) > 0:
                metrics['stance_time_mean'] = float(np.mean(stance_times))
                metrics['stance_time_std'] = float(np.std(stance_times))

        # Calculate swing time (FO to next FS)
        swing_times = None
        if len(fs_times) > 1 and len(fo_times) > 0:
            min_len = min(len(fs_times) - 1, len(fo_times))
            swing_times = fs_times[1:min_len + 1] - fo_times[:min_len]
            # Filter out any negative swing times (these indicate misaligned events)
            swing_times = swing_times[swing_times > 0]
            if len(swing_times) > 0:
                metrics['swing_time_mean'] = float(np.mean(swing_times))
                metrics['swing_time_std'] = float(np.std(swing_times))

        # Calculate stride time (MSW to next MSW)
        stride_times = None
        if len(msw_times) > 1:
            stride_times = np.diff(msw_times)
            metrics['stride_time_mean'] = float(np.mean(stride_times))
            metrics['stride_time_std'] = float(np.std(stride_times))
            
            # Cadence (steps per minute) - each stride = 2 steps
            metrics['cadence'] = float(60.0 / np.mean(stride_times) * 2)
            
            # Stride time variability (Coefficient of Variation)
            # Higher CV indicates less consistent stride pattern (injury risk indicator)
            if metrics['stride_time_mean'] > 0:
                metrics['stride_time_cv'] = float((np.std(stride_times) / metrics['stride_time_mean']) * 100)

        # Stance/Swing ratio - important for gait efficiency and balance
        # Typical ratio is around 60/40 (stance/swing) for walking
        if metrics['stance_time_mean'] is not None and metrics['swing_time_mean'] is not None:
            metrics['stance_swing_ratio'] = float(metrics['stance_time_mean'] / metrics['swing_time_mean'])
        
        # Contact time as percentage of stride time
        # Useful for injury prevention - higher contact times can indicate fatigue
        if metrics['stance_time_mean'] is not None and metrics['stride_time_mean'] is not None:
            metrics['contact_time_percent'] = float((metrics['stance_time_mean'] / metrics['stride_time_mean']) * 100)

        return metrics

    def get_all_events(self, foot: str) -> Dict[str, List[float]]:
        """
        Get all detected event times for a given foot.

        Args:
            foot: 'left' or 'right'

        Returns:
            Dictionary with event types as keys and lists of times as values
        """
        foot_state = self.feet_state[foot]
        return {
            'msw': foot_state['msw_times'].copy(),
            'fs': foot_state['foot_strike_times'].copy(),
            'fo': foot_state['foot_off_times'].copy(),
            'ms': foot_state['mid_stance_times'].copy(),
        }
    
    def get_symmetry_metrics(self, window_seconds: Optional[float] = None) -> Dict:
        """
        Calculate gait symmetry metrics between left and right feet.
        
        Asymmetry is a key indicator for injury risk and rehabilitation progress.
        
        Args:
            window_seconds: If provided, only calculate metrics for the last N seconds.
                          If None, calculate for all data.
        
        Returns:
            Dictionary containing symmetry metrics
        """
        metrics_left = self.get_metrics('left', window_seconds)
        metrics_right = self.get_metrics('right', window_seconds)
        
        symmetry = {
            'stride_time_symmetry': None,
            'stance_time_symmetry': None,
            'swing_time_symmetry': None,
        }
        
        # Calculate Gait Symmetry Index (GSI) for each metric
        # GSI = |Left - Right| / (0.5 * (Left + Right)) * 100
        # Values closer to 0% indicate better symmetry
        # Values > 10% may indicate pathological gait or injury
        
        if metrics_left['stride_time_mean'] is not None and metrics_right['stride_time_mean'] is not None:
            left_val = metrics_left['stride_time_mean']
            right_val = metrics_right['stride_time_mean']
            symmetry['stride_time_symmetry'] = float(
                abs(left_val - right_val) / (0.5 * (left_val + right_val)) * 100
            )
        
        if metrics_left['stance_time_mean'] is not None and metrics_right['stance_time_mean'] is not None:
            left_val = metrics_left['stance_time_mean']
            right_val = metrics_right['stance_time_mean']
            symmetry['stance_time_symmetry'] = float(
                abs(left_val - right_val) / (0.5 * (left_val + right_val)) * 100
            )
        
        if metrics_left['swing_time_mean'] is not None and metrics_right['swing_time_mean'] is not None:
            left_val = metrics_left['swing_time_mean']
            right_val = metrics_right['swing_time_mean']
            symmetry['swing_time_symmetry'] = float(
                abs(left_val - right_val) / (0.5 * (left_val + right_val)) * 100
            )
        
        return symmetry
