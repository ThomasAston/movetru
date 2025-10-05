"""Complete gait event detection pipeline."""

import numpy as np
from collections import deque

from .signal_processing import moving_average
from .event_detection import (
    detect_zero_crossing_descending,
    detect_local_minimum,
    detect_mid_stance,
    detect_foot_strike,
    detect_foot_off,
    is_valid_stride
)


def process_gait_data_realtime(left_foot, right_foot, fs, 
                               msw_threshold=-115, 
                               zc_threshold=0,
                               ma_window=5,
                               max_stride_time=2.5):
    """
    Process gait data in real-time simulation to detect all gait events.
    
    This function simulates real-time streaming by processing samples one at a time,
    applying smoothing, detecting mid-swing events, and then retrospectively detecting
    foot strike, mid-stance, and foot off events between consecutive mid-swings.
    
    Parameters
    ----------
    left_foot : array-like
        Left foot gyroscope Y-axis data
    right_foot : array-like
        Right foot gyroscope Y-axis data
    fs : float
        Sampling frequency (Hz)
    msw_threshold : float, optional
        Threshold for mid-swing detection in deg/s (default: -115)
    zc_threshold : float, optional
        Threshold for zero-crossing detection (default: 0)
    ma_window : int, optional
        Moving average window size (default: 5)
    max_stride_time : float, optional
        Maximum stride time in seconds (default: 2.5)
    
    Returns
    -------
    dict
        Dictionary containing results for both feet with keys:
        - 'left' and 'right', each containing:
            - 'smoothed': smoothed signal
            - 'msw_indices': mid-swing event indices
            - 'zc_desc_indices': descending zero-crossing indices
            - 'mid_stance_indices': mid-stance event indices
            - 'foot_strike_indices': foot strike event indices
            - 'foot_off_indices': foot off event indices
    """
    max_buffer_size = int(fs * 5)  # Keep last 5 seconds for smoothing
    
    # Initialize state for both feet
    feet_data = {
        'left': {'raw_signal': left_foot},
        'right': {'raw_signal': right_foot}
    }
    
    for foot_name, foot_state in feet_data.items():
        # Smoothing buffers
        foot_state['raw_buffer'] = deque(maxlen=max_buffer_size)
        foot_state['smoothed_history'] = []  # Last 3 samples for MSW detection
        foot_state['all_smoothed'] = []  # All smoothed values
        
        # MSW detection state
        foot_state['zc_prev'] = None
        foot_state['after_desc_zc'] = False
        foot_state['msw_indices'] = []
        foot_state['zc_desc_indices'] = []
        
        # Gait event detection state
        foot_state['previous_msw'] = None
        foot_state['mid_stance_indices'] = []
        foot_state['foot_strike_indices'] = []
        foot_state['foot_off_indices'] = []
    
    # Main streaming loop: process samples one by one
    for i in range(len(left_foot)):
        for foot_name, foot_state in feet_data.items():
            # Get new raw sample
            raw_sample = foot_state['raw_signal'][i]
            
            # Update smoothing buffer
            foot_state['raw_buffer'].append(raw_sample)
            
            # Compute smoothed value
            smoothed_sample = moving_average(
                np.array(foot_state['raw_buffer']), 
                window=ma_window
            )
            foot_state['all_smoothed'].append(smoothed_sample)
            
            # Update history for local minima detection
            foot_state['smoothed_history'].append(smoothed_sample)
            if len(foot_state['smoothed_history']) > 3:
                foot_state['smoothed_history'].pop(0)
            
            # Detect zero-crossing
            foot_state['zc_prev'], is_desc_zc = detect_zero_crossing_descending(
                smoothed_sample, foot_state['zc_prev'], zc_threshold
            )
            
            if is_desc_zc:
                foot_state['after_desc_zc'] = True
                foot_state['zc_desc_indices'].append(i)
            
            # Detect local minimum after descending zero-crossing
            if foot_state['after_desc_zc']:
                if detect_local_minimum(foot_state['smoothed_history'], msw_threshold):
                    current_msw = i - 1  # The local min is at i-1
                    foot_state['msw_indices'].append(current_msw)
                    foot_state['after_desc_zc'] = False
                    
                    # Retrospective FS/MS/FO detection
                    if foot_state['previous_msw'] is not None:
                        prev_msw = foot_state['previous_msw']
                        
                        # Check if this forms a valid stride
                        if is_valid_stride(prev_msw, current_msw, fs, max_stride_time):
                            # Convert smoothed data to numpy for processing
                            smoothed_signal = np.array(foot_state['all_smoothed'])
                            signal_norm = np.abs(smoothed_signal)
                            
                            # Detect mid-stance
                            mid_stance_idx = detect_mid_stance(
                                signal_norm, prev_msw, current_msw
                            )
                            
                            if mid_stance_idx is not None:
                                foot_state['mid_stance_indices'].append(mid_stance_idx)
                                
                                # Detect foot strike
                                fs_idx = detect_foot_strike(
                                    smoothed_signal, prev_msw, mid_stance_idx
                                )
                                if fs_idx is not None:
                                    foot_state['foot_strike_indices'].append(fs_idx)
                                
                                # Detect foot off
                                fo_idx = detect_foot_off(
                                    smoothed_signal, mid_stance_idx, current_msw
                                )
                                if fo_idx is not None:
                                    foot_state['foot_off_indices'].append(fo_idx)
                    
                    # Update previous MSW for next iteration
                    foot_state['previous_msw'] = current_msw
    
    # Convert to numpy arrays and prepare output
    results = {}
    for foot_name in ['left', 'right']:
        results[foot_name] = {
            'smoothed': np.array(feet_data[foot_name]['all_smoothed']),
            'msw_indices': np.array(feet_data[foot_name]['msw_indices']),
            'zc_desc_indices': np.array(feet_data[foot_name]['zc_desc_indices']),
            'mid_stance_indices': np.array(feet_data[foot_name]['mid_stance_indices']),
            'foot_strike_indices': np.array(feet_data[foot_name]['foot_strike_indices']),
            'foot_off_indices': np.array(feet_data[foot_name]['foot_off_indices'])
        }
    
    return results