"""
Signal filtering for real-time IMU data processing.

This module provides various filtering approaches optimized for live streaming:
1. Butterworth low-pass filter (causal, for real-time)
2. Moving average filter (simple smoothing)
3. Savitzky-Golay filter (local polynomial regression)
4. Adaptive frequency-based filtering with FFT analysis
"""

from collections import deque
from typing import List, Tuple, Optional, Literal
import numpy as np
from scipy.signal import butter, sosfilt, savgol_filter
from dataclasses import dataclass


FilterType = Literal['butterworth', 'moving_average', 'savgol', 'none']


@dataclass
class FilterConfig:
    """Configuration for signal filtering."""
    filter_type: FilterType = 'butterworth'
    cutoff_freq: float = 12.6  # Hz - default from your notebook analysis
    filter_order: int = 4
    sampling_rate: int = 256  # Hz
    
    # Moving average parameters
    window_size: int = 5
    
    # Savitzky-Golay parameters
    savgol_window: int = 11  # Must be odd
    savgol_polyorder: int = 2
    
    # Adaptive filtering
    use_adaptive_cutoff: bool = False
    n_harmonics: int = 3
    fft_window_size: int = 2048  # Samples for FFT analysis
    cutoff_update_interval: int = 1000  # Samples between cutoff updates


class ButterworthFilter:
    """
    Real-time Butterworth low-pass filter using sosfilt (causal).
    
    Uses second-order sections (SOS) for numerical stability and maintains
    internal state (zi) for sample-by-sample processing.
    
    Unlike filtfilt (zero-phase, non-causal), sosfilt processes data causally
    which is essential for live streaming.
    
    Note: Introduces phase lag, but this is acceptable for stride detection 
    where relative timing between peaks matters more than absolute timing.
    """
    
    def __init__(self, cutoff: float, fs: int, order: int = 4):
        """
        Initialize Butterworth filter.
        
        Args:
            cutoff: Cutoff frequency in Hz
            fs: Sampling rate in Hz
            order: Filter order (default 4, as in your notebook)
        """
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        
        # Design filter using second-order sections (more numerically stable)
        self.sos = butter(order, cutoff, btype='low', fs=fs, output='sos')
        
        # Initialize filter state (stores filter's internal state between calls)
        self.zi = None
        self._reset_state()
    
    def _reset_state(self):
        """Reset filter state to initial conditions."""
        from scipy.signal import sosfilt_zi
        self.zi = sosfilt_zi(self.sos)
    
    def filter_sample(self, sample: float) -> float:
        """
        Filter a single sample in real-time.
        
        Args:
            sample: Input sample value
            
        Returns:
            Filtered sample value
        """
        # Process single sample through filter
        filtered, self.zi = sosfilt(self.sos, [sample], zi=self.zi)
        return filtered[0]
    
    def filter_batch(self, samples: np.ndarray) -> np.ndarray:
        """
        Filter a batch of samples, maintaining state.
        
        Args:
            samples: Array of input samples
            
        Returns:
            Array of filtered samples
        """
        filtered, self.zi = sosfilt(self.sos, samples, zi=self.zi)
        return filtered
    
    def reset(self):
        """Reset filter state (useful when starting a new stream)."""
        self._reset_state()
    
    def update_cutoff(self, new_cutoff: float):
        """
        Update cutoff frequency and reset filter.
        
        Args:
            new_cutoff: New cutoff frequency in Hz
        """
        self.cutoff = new_cutoff
        self.sos = butter(self.order, new_cutoff, btype='low', fs=self.fs, output='sos')
        self._reset_state()


class MovingAverageFilter:
    """
    Simple moving average filter for smoothing.
    
    Very fast, no phase distortion issues, but less effective at 
    frequency-selective filtering compared to Butterworth.
    """
    
    def __init__(self, window_size: int):
        """
        Initialize moving average filter.
        
        Args:
            window_size: Number of samples to average
        """
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        self.sum = 0.0
    
    def filter_sample(self, sample: float) -> float:
        """
        Filter a single sample.
        
        Args:
            sample: Input sample value
            
        Returns:
            Filtered sample value (moving average)
        """
        # Remove oldest value from sum if buffer is full
        if len(self.buffer) == self.window_size:
            self.sum -= self.buffer[0]
        
        # Add new sample
        self.buffer.append(sample)
        self.sum += sample
        
        # Return average
        return self.sum / len(self.buffer)
    
    def filter_batch(self, samples: np.ndarray) -> np.ndarray:
        """Filter batch of samples."""
        return np.array([self.filter_sample(s) for s in samples])
    
    def reset(self):
        """Reset filter state."""
        self.buffer.clear()
        self.sum = 0.0


class SavitzkyGolayFilter:
    """
    Savitzky-Golay filter for smoothing with polynomial fitting.
    
    Good compromise between smoothing and preserving peaks.
    Requires buffering (window_length samples), so has inherent delay.
    """
    
    def __init__(self, window_length: int, polyorder: int):
        """
        Initialize Savitzky-Golay filter.
        
        Args:
            window_length: Length of filter window (must be odd)
            polyorder: Order of polynomial fit
        """
        if window_length % 2 == 0:
            window_length += 1  # Must be odd
        
        self.window_length = window_length
        self.polyorder = polyorder
        self.buffer = deque(maxlen=window_length)
    
    def filter_sample(self, sample: float) -> Optional[float]:
        """
        Filter a single sample.
        
        Args:
            sample: Input sample value
            
        Returns:
            Filtered sample value, or None if buffer not yet full
        """
        self.buffer.append(sample)
        
        if len(self.buffer) < self.window_length:
            return None  # Not enough data yet
        
        # Apply Savitzky-Golay filter to buffered data
        filtered = savgol_filter(list(self.buffer), self.window_length, self.polyorder)
        
        # Return the center point (current filtered value)
        return filtered[self.window_length // 2]
    
    def filter_batch(self, samples: np.ndarray) -> np.ndarray:
        """Filter batch of samples."""
        results = []
        for s in samples:
            result = self.filter_sample(s)
            if result is not None:
                results.append(result)
        return np.array(results)
    
    def reset(self):
        """Reset filter state."""
        self.buffer.clear()


class AdaptiveFrequencyFilter:
    """
    Adaptive Butterworth filter with dynamic cutoff based on FFT analysis.
    
    Periodically analyzes frequency content to adjust cutoff frequency,
    similar to your notebook approach but adapted for streaming.
    """
    
    def __init__(self, config: FilterConfig):
        """
        Initialize adaptive filter.
        
        Args:
            config: Filter configuration
        """
        self.config = config
        self.butterworth = ButterworthFilter(
            config.cutoff_freq, 
            config.sampling_rate, 
            config.filter_order
        )
        
        # Buffer for FFT analysis
        self.fft_buffer = deque(maxlen=config.fft_window_size)
        self.sample_count = 0
        self.current_cutoff = config.cutoff_freq
    
    def filter_sample(self, sample: float) -> float:
        """
        Filter sample and periodically update cutoff.
        
        Args:
            sample: Input sample value
            
        Returns:
            Filtered sample value
        """
        # Add to FFT buffer
        self.fft_buffer.append(sample)
        self.sample_count += 1
        
        # Periodically update cutoff based on FFT
        if (self.sample_count % self.config.cutoff_update_interval == 0 and 
            len(self.fft_buffer) == self.config.fft_window_size):
            self._update_cutoff_from_fft()
        
        # Apply filter
        return self.butterworth.filter_sample(sample)
    
    def _update_cutoff_from_fft(self):
        """Analyze frequency content and update cutoff frequency."""
        # Convert buffer to numpy array
        signal = np.array(self.fft_buffer)
        
        # Compute FFT
        fft_result = np.fft.fft(signal)
        magnitude = np.abs(fft_result[:len(fft_result)//2])
        freqs = np.fft.fftfreq(len(signal), 1/self.config.sampling_rate)
        positive_freqs = freqs[:len(freqs)//2]
        
        # Find peaks in frequency domain
        from scipy.signal import find_peaks
        min_peak_height = np.max(magnitude) * 0.05  # 5% of max
        peaks, _ = find_peaks(magnitude, height=min_peak_height)
        
        if len(peaks) == 0:
            return  # No peaks found, keep current cutoff
        
        # Get peak frequencies and magnitudes
        peak_freqs = positive_freqs[peaks]
        peak_mags = magnitude[peaks]
        
        # Sort by magnitude
        sorted_idx = np.argsort(peak_mags)[::-1]
        
        # Take top N harmonics
        n_keep = min(self.config.n_harmonics, len(peaks))
        top_freqs = peak_freqs[sorted_idx[:n_keep]]
        
        # Set cutoff to max frequency + buffer
        new_cutoff = np.max(top_freqs) + 1.0
        
        # Constrain cutoff to reasonable range (1-50 Hz)
        new_cutoff = np.clip(new_cutoff, 1.0, 50.0)
        
        # Only update if significantly different (>10% change)
        if abs(new_cutoff - self.current_cutoff) / self.current_cutoff > 0.1:
            print(f"📊 Adaptive filter: updating cutoff {self.current_cutoff:.1f} Hz → {new_cutoff:.1f} Hz")
            self.butterworth.update_cutoff(new_cutoff)
            self.current_cutoff = new_cutoff
    
    def reset(self):
        """Reset filter state."""
        self.butterworth.reset()
        self.fft_buffer.clear()
        self.sample_count = 0


class StreamingFilter:
    """
    Main streaming filter interface with multiple filter options.
    
    This is the primary class you'll use in your Streamlit app.
    """
    
    def __init__(self, config: FilterConfig):
        """
        Initialize streaming filter.
        
        Args:
            config: Filter configuration
        """
        self.config = config
        self.filter_type = config.filter_type
        
        # Initialize appropriate filter
        if config.filter_type == 'butterworth':
            if config.use_adaptive_cutoff:
                self.filter = AdaptiveFrequencyFilter(config)
            else:
                self.filter = ButterworthFilter(
                    config.cutoff_freq, 
                    config.sampling_rate, 
                    config.filter_order
                )
        elif config.filter_type == 'moving_average':
            self.filter = MovingAverageFilter(config.window_size)
        elif config.filter_type == 'savgol':
            self.filter = SavitzkyGolayFilter(config.savgol_window, config.savgol_polyorder)
        else:  # 'none'
            self.filter = None
    
    def filter_sample(self, sample: float) -> float:
        """
        Filter a single sample.
        
        Args:
            sample: Input sample value
            
        Returns:
            Filtered sample value (or original if no filter)
        """
        if self.filter is None:
            return sample
        return self.filter.filter_sample(sample)
    
    def filter_batch(self, samples: np.ndarray) -> np.ndarray:
        """
        Filter a batch of samples.
        
        Args:
            samples: Array of input samples
            
        Returns:
            Array of filtered samples
        """
        if self.filter is None:
            return samples
        
        if hasattr(self.filter, 'filter_batch'):
            return self.filter.filter_batch(samples)
        else:
            return np.array([self.filter.filter_sample(s) for s in samples])
    
    def reset(self):
        """Reset filter state."""
        if self.filter is not None:
            self.filter.reset()
    
    def get_info(self) -> dict:
        """
        Get filter information for display.
        
        Returns:
            Dictionary with filter details
        """
        if self.config.filter_type == 'none':
            return {'type': 'None', 'description': 'No filtering'}
        elif self.config.filter_type == 'butterworth':
            adaptive = " (Adaptive)" if self.config.use_adaptive_cutoff else ""
            return {
                'type': f'Butterworth{adaptive}',
                'order': self.config.filter_order,
                'cutoff': f'{self.config.cutoff_freq:.1f} Hz',
                'description': f'{self.config.filter_order}th order low-pass @ {self.config.cutoff_freq:.1f} Hz'
            }
        elif self.config.filter_type == 'moving_average':
            return {
                'type': 'Moving Average',
                'window': self.config.window_size,
                'description': f'Moving average (window={self.config.window_size})'
            }
        elif self.config.filter_type == 'savgol':
            return {
                'type': 'Savitzky-Golay',
                'window': self.config.savgol_window,
                'order': self.config.savgol_polyorder,
                'description': f'Savitzky-Golay (window={self.config.savgol_window}, poly={self.config.savgol_polyorder})'
            }


def compare_filters(signal: np.ndarray, fs: int = 256, cutoff: float = 12.6) -> dict:
    """
    Compare different filters on a signal sample.
    
    Useful for testing and visualization.
    
    Args:
        signal: Input signal array
        fs: Sampling rate in Hz
        cutoff: Cutoff frequency in Hz
        
    Returns:
        Dictionary with filtered signals for each filter type
    """
    results = {'original': signal}
    
    # Butterworth
    config_butter = FilterConfig(filter_type='butterworth', cutoff_freq=cutoff, sampling_rate=fs)
    filter_butter = StreamingFilter(config_butter)
    results['butterworth'] = filter_butter.filter_batch(signal)
    
    # Moving Average
    config_ma = FilterConfig(filter_type='moving_average', window_size=5)
    filter_ma = StreamingFilter(config_ma)
    results['moving_average'] = filter_ma.filter_batch(signal)
    
    # Savitzky-Golay
    config_sg = FilterConfig(filter_type='savgol', savgol_window=11, savgol_polyorder=2)
    filter_sg = StreamingFilter(config_sg)
    results['savgol'] = filter_sg.filter_batch(signal)
    
    return results
