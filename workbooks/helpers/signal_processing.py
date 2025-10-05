"""Signal processing utilities for gait analysis."""

import numpy as np


def moving_average(x, window=5):
    """
    Compute the moving average live.
    
    Parameters
    ----------
    x : array-like
        Input signal
    window : int, optional
        Window size for moving average (default: 5)
    
    Returns
    -------
    float
        Moving average of the last 'window' samples
    """
    if len(x) < window:
        # Not enough data yet, return mean of what we have
        return np.mean(x) if len(x) > 0 else 0
    
    # Compute moving average of last 'window' samples
    return np.mean(x[-window:])


def compute_angular_velocity_norm(gyro_x, gyro_y, gyro_z):
    """
    Compute the norm of the angular velocity vector.
    
    Parameters
    ----------
    gyro_x, gyro_y, gyro_z : array-like
        Angular velocity components
    
    Returns
    -------
    array-like
        Norm of the angular velocity vector
    """
    return np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
