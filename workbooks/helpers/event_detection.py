"""Event detection algorithms for gait analysis."""

import numpy as np


def detect_zero_crossing_descending(current_value, previous_state, threshold=0):
    """
    Detect descending zero-crossing (transition from positive to negative).
    
    Parameters
    ----------
    current_value : float
        Current signal value
    previous_state : bool or None
        Previous state (True = above threshold, False = below)
    threshold : float, optional
        Zero-crossing threshold (default: 0)
    
    Returns
    -------
    tuple
        (current_state, is_descending_zc)
    """
    current_state = current_value >= threshold
    is_descending_zc = False
    
    if previous_state is not None:
        # Descending ZC: was positive (True), now negative (False)
        if previous_state and not current_state:
            is_descending_zc = True
    
    return current_state, is_descending_zc


def detect_local_minimum(history, threshold):
    """
    Detect local minimum using 3-sample window.
    
    Parameters
    ----------
    history : list
        Last 3 samples [i-2, i-1, i]
    threshold : float
        Minimum threshold value
    
    Returns
    -------
    bool
        True if local minimum detected
    """
    if len(history) < 3:
        return False
    
    prev_sample = history[0]   # i-2
    curr_sample = history[1]   # i-1 (candidate for local minimum)
    next_sample = history[2]   # i (current)
    
    # Check if middle sample is a local minimum
    is_local_min = (prev_sample >= curr_sample) and (curr_sample <= next_sample)
    below_threshold = curr_sample < threshold
    
    return is_local_min and below_threshold


def detect_mid_stance(signal, msw_start_idx, msw_end_idx):
    """
    Detect mid-stance as the minimum angular velocity norm in the 30-60% 
    time-range between consecutive mid-swings.
    
    Parameters
    ----------
    signal : array-like
        Angular velocity signal
    msw_start_idx : int
        Index of the first mid-swing
    msw_end_idx : int
        Index of the second mid-swing
    
    Returns
    -------
    int or None
        Index of mid-stance event, or None if not found
    """
    cycle_length = msw_end_idx - msw_start_idx
    
    # Define 30-60% time range
    start_search = msw_start_idx + int(0.3 * cycle_length)
    end_search = msw_start_idx + int(0.6 * cycle_length)
    
    # Ensure valid range
    if start_search >= end_search or end_search > msw_end_idx:
        return None
    
    # Find minimum in this range
    search_segment = signal[start_search:end_search]
    min_idx_relative = np.argmin(search_segment)
    
    return start_search + min_idx_relative


def detect_foot_strike(signal, msw_idx, mid_stance_idx):
    """
    Detect foot strike as the maximum angular velocity between 
    mid-swing and mid-stance.
    
    Parameters
    ----------
    signal : array-like
        Angular velocity signal
    msw_idx : int
        Index of mid-swing
    mid_stance_idx : int
        Index of mid-stance
    
    Returns
    -------
    int or None
        Index of foot strike event, or None if not found
    """
    if msw_idx >= mid_stance_idx:
        return None
    
    # Search between MSW and mid-stance
    search_segment = signal[msw_idx:mid_stance_idx]
    max_idx_relative = np.argmax(search_segment)
    
    return msw_idx + max_idx_relative


def detect_foot_off(signal, mid_stance_idx, next_msw_idx):
    """
    Detect foot off as the maximum angular velocity between 
    mid-stance and next mid-swing.
    
    Parameters
    ----------
    signal : array-like
        Angular velocity signal
    mid_stance_idx : int
        Index of mid-stance
    next_msw_idx : int
        Index of next mid-swing
    
    Returns
    -------
    int or None
        Index of foot off event, or None if not found
    """
    if mid_stance_idx >= next_msw_idx:
        return None
    
    # Search between mid-stance and next MSW
    search_segment = signal[mid_stance_idx:next_msw_idx]
    max_idx_relative = np.argmax(search_segment)
    
    return mid_stance_idx + max_idx_relative


def is_valid_stride(msw1_idx, msw2_idx, fs, max_stride_time=2.5):
    """
    Check if two consecutive mid-swings are within a reasonable time range
    to be considered part of the same stride pattern.
    
    Parameters
    ----------
    msw1_idx : int
        Index of first mid-swing
    msw2_idx : int
        Index of second mid-swing
    fs : float
        Sampling frequency (Hz)
    max_stride_time : float, optional
        Maximum stride time in seconds (default: 2.5)
    
    Returns
    -------
    bool
        True if stride is valid
    """
    stride_time = (msw2_idx - msw1_idx) / fs
    return 0.05 < stride_time < max_stride_time
