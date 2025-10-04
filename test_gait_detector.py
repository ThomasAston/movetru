"""
Quick test script to verify gait detection logic.
Run this to ensure the GaitEventDetector works correctly before testing in Streamlit.
"""

import numpy as np
from src.imu_streaming import GaitEventDetector


def test_gait_detector():
    """Test the gait detector with synthetic data."""
    
    print("Testing GaitEventDetector...")
    print("=" * 60)
    
    # Initialize detector with 256 Hz sampling rate (matching Streamlit app)
    detector = GaitEventDetector(
        fs=256,
        msw_threshold=-115.0,
        zc_threshold=0.0,
        ma_window=15,
        max_buffer_size=1280,  # 5 seconds
        max_stride_time=2.5,
        min_stride_time=0.1,
    )
    
    # Generate synthetic gait data (simplified sine wave pattern)
    # Simulate 5 seconds at 256 Hz
    duration = 5.0
    fs = 256
    num_samples = int(duration * fs)
    t = np.linspace(0, duration, num_samples)
    
    # Simulate gyroscope Y data for walking at ~1.5 Hz cadence
    # Pattern: positive peak (FO) -> zero crossing -> negative peak (MSW) -> positive peak (FS)
    cadence_hz = 1.5
    signal_lf = 200 * np.sin(2 * np.pi * cadence_hz * t - np.pi/2)
    signal_rf = 200 * np.sin(2 * np.pi * cadence_hz * (t - 0.3) - np.pi/2)  # Offset for alternating gait
    
    print(f"Simulating {duration}s of gait data at {fs} Hz")
    print(f"Expected cadence: {cadence_hz} Hz ({cadence_hz * 60:.1f} steps/min)")
    print(f"Total samples: {num_samples}")
    print()
    
    # Process samples
    event_counts = {'left': 0, 'right': 0}
    
    for i in range(num_samples):
        events = detector.process_sample(
            lf_gyro_y=signal_lf[i],
            rf_gyro_y=signal_rf[i],
            lf_time=t[i],
            rf_time=t[i]
        )
        
        # Count new events
        if events['left']['msw']:
            event_counts['left'] += len(events['left']['msw'])
        if events['right']['msw']:
            event_counts['right'] += len(events['right']['msw'])
    
    print("Processing complete!")
    print()
    
    # Get final metrics
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    for foot in ['left', 'right']:
        print(f"\n{foot.upper()} FOOT:")
        print("-" * 40)
        
        # Get all events
        all_events = detector.get_all_events(foot)
        print(f"  MSW events: {len(all_events['msw'])}")
        print(f"  FS events:  {len(all_events['fs'])}")
        print(f"  FO events:  {len(all_events['fo'])}")
        print(f"  MS events:  {len(all_events['ms'])}")
        
        # Get overall metrics
        metrics = detector.get_metrics(foot)
        print(f"\n  Overall Session Metrics:")
        print(f"    Total strides: {metrics['total_strides']}")
        
        if metrics['stance_time_mean'] is not None:
            print(f"    Stance time:   {metrics['stance_time_mean']:.3f}s ± {metrics['stance_time_std']:.3f}s")
        
        if metrics['swing_time_mean'] is not None:
            print(f"    Swing time:    {metrics['swing_time_mean']:.3f}s ± {metrics['swing_time_std']:.3f}s")
        
        if metrics['stride_time_mean'] is not None:
            print(f"    Stride time:   {metrics['stride_time_mean']:.3f}s ± {metrics['stride_time_std']:.3f}s")
            cadence = 1.0 / metrics['stride_time_mean'] if metrics['stride_time_mean'] > 0 else 0
            print(f"    Cadence:       {cadence:.2f} Hz ({cadence * 60:.1f} steps/min)")
        
        # Get recent metrics (last 2 seconds)
        metrics_recent = detector.get_metrics(foot, window_seconds=2.0)
        print(f"\n  Recent Metrics (last 2s):")
        print(f"    Total strides: {metrics_recent['total_strides']}")
        
        if metrics_recent['stride_time_mean'] is not None:
            print(f"    Stride time:   {metrics_recent['stride_time_mean']:.3f}s ± {metrics_recent['stride_time_std']:.3f}s")
    
    print()
    print("=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_gait_detector()
