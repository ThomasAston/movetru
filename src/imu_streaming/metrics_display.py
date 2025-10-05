"""Metrics display helpers for gait analysis UI."""

from typing import Optional, Tuple, Dict, Any


def get_metric_status(value: Optional[float], metric_type: str) -> Tuple[str, str]:
    """
    Determine status (good/acceptable/poor) based on metric value and type.
    
    Args:
        value: The metric value to evaluate
        metric_type: The type of metric being evaluated
        
    Returns:
        Tuple of (emoji, status_text)
    """
    if value is None:
        return "", ""
    
    status_ranges = {
        'stride_variability': [
            (lambda v: v < 3, "🟢", "Excellent"),
            (lambda v: v < 5, "🟡", "Acceptable"),
            (lambda v: True, "🔴", "High")
        ],
        'symmetry': [
            (lambda v: v < 5, "🟢", "Excellent"),
            (lambda v: v < 10, "🟡", "Good"),
            (lambda v: True, "🔴", "Asymmetric")
        ],
        'contact_time_walking': [
            (lambda v: 55 <= v <= 65, "🟢", "Optimal"),
            (lambda v: 50 <= v <= 70, "🟡", "Acceptable"),
            (lambda v: True, "🔴", "Atypical")
        ],
        'stance_swing_walking': [
            (lambda v: 1.3 <= v <= 1.7, "🟢", "Optimal"),
            (lambda v: 1.0 <= v <= 2.0, "🟡", "Acceptable"),
            (lambda v: True, "🔴", "Atypical")
        ],
        'cadence_walking': [
            (lambda v: 100 <= v <= 120, "🟢", "Walking"),
            (lambda v: 160 <= v <= 180, "🟢", "Running"),
            (lambda v: 90 <= v <= 130 or 150 <= v <= 190, "🟡", "Moderate"),
            (lambda v: True, "🟡", "Variable")
        ]
    }
    
    ranges = status_ranges.get(metric_type, [])
    for condition, emoji, status in ranges:
        if condition(value):
            return emoji, status
    
    return "", ""


def format_metric_value(value: Optional[float], unit: str, emoji: str = "", status: str = "") -> str:
    """
    Format metric value with optional emoji and status.
    
    Args:
        value: The metric value to format
        unit: The unit string to append
        emoji: Optional emoji indicator
        status: Optional status text
        
    Returns:
        Formatted metric string
    """
    if value is None:
        return "--"
    
    formatted = f"{value}{unit}"
    if emoji and status:
        return f"{emoji} {formatted}"
    return formatted


def calculate_combined_metrics(metrics_lf: Dict[str, Any], metrics_rf: Dict[str, Any], 
                               symmetry: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Calculate combined metrics from left and right foot data.
    
    Args:
        metrics_lf: Left foot metrics dictionary
        metrics_rf: Right foot metrics dictionary
        symmetry: Symmetry metrics dictionary
        
    Returns:
        Dictionary of combined metrics
    """
    def avg_metrics(key: str, lf: Dict, rf: Dict) -> Optional[float]:
        """Calculate average of a metric from left and right foot."""
        if lf[key] is not None and rf[key] is not None:
            return (lf[key] + rf[key]) / 2
        return None
    
    return {
        'total_strides': metrics_lf['total_strides'] + metrics_rf['total_strides'],
        'avg_cadence': avg_metrics('cadence', metrics_lf, metrics_rf),
        'avg_stride_cv': avg_metrics('stride_time_cv', metrics_lf, metrics_rf),
        'avg_contact': avg_metrics('contact_time_percent', metrics_lf, metrics_rf),
        'avg_stance_swing': avg_metrics('stance_swing_ratio', metrics_lf, metrics_rf),
        'stride_symmetry': symmetry['stride_time_symmetry']
    }


def display_overall_metrics(placeholders: Dict[str, Any], combined: Dict[str, Optional[float]], 
                           tooltips: Dict[str, str]):
    """
    Display overall combined metrics with color-coded status indicators.
    
    Args:
        placeholders: Dictionary of Streamlit placeholder objects
        combined: Combined metrics dictionary
        tooltips: Tooltip text dictionary
    """
    placeholders['total_strides'].metric("Total Strides", value=combined['total_strides'])
    
    # Cadence
    emoji, _ = get_metric_status(combined['avg_cadence'], 'cadence_walking')
    display = f"{emoji} {combined['avg_cadence']:.1f}" if combined['avg_cadence'] is not None else "--"
    placeholders['cadence'].metric("Cadence (steps/min)", value=display, help=tooltips['cadence'])
    
    # Stride Variability
    emoji, _ = get_metric_status(combined['avg_stride_cv'], 'stride_variability')
    display = f"{emoji} {combined['avg_stride_cv']:.1f}%" if combined['avg_stride_cv'] is not None else "--"
    placeholders['stride_variability'].metric("Stride Variability (CV)", value=display, 
                                             help=tooltips['stride_variability'])
    
    # Stride Symmetry
    emoji, _ = get_metric_status(combined['stride_symmetry'], 'symmetry')
    display = f"{emoji} {combined['stride_symmetry']:.1f}%" if combined['stride_symmetry'] is not None else "--"
    placeholders['stride_symmetry'].metric("Stride Symmetry", value=display, help=tooltips['stride_symmetry'])
    
    # Contact Time
    emoji, _ = get_metric_status(combined['avg_contact'], 'contact_time_walking')
    display = f"{emoji} {combined['avg_contact']:.1f}%" if combined['avg_contact'] is not None else "--"
    placeholders['contact_time'].metric("Avg Contact Time", value=display, help=tooltips['contact_time'])
    
    # Stance/Swing Ratio
    emoji, _ = get_metric_status(combined['avg_stance_swing'], 'stance_swing_walking')
    display = f"{emoji} {combined['avg_stance_swing']:.2f}" if combined['avg_stance_swing'] is not None else "--"
    placeholders['stance_swing_ratio'].metric("Stance/Swing Ratio", value=display, 
                                             help=tooltips['stance_swing_ratio'])


def display_per_foot_metrics(placeholders: Dict[str, Any], metrics: Dict[str, Any], 
                             tooltips: Dict[str, str]):
    """
    Display per-foot metrics.
    
    Args:
        placeholders: Dictionary of Streamlit placeholder objects
        metrics: Metrics dictionary for one foot
        tooltips: Tooltip text dictionary
    """
    placeholders['stride'].metric(
        "Stride Time",
        value=f"{metrics['stride_time_mean']:.3f} s" if metrics['stride_time_mean'] is not None else "--",
        delta=f"± {metrics['stride_time_std']:.3f} s" if metrics['stride_time_std'] is not None else None,
        delta_color="off",
        help=tooltips['stride_time']
    )
    
    emoji, _ = get_metric_status(metrics['contact_time_percent'], 'contact_time_walking')
    display = f"{emoji} {metrics['contact_time_percent']:.1f}%" if metrics['contact_time_percent'] is not None else "--"
    placeholders['contact'].metric("Contact Time", value=display, help=tooltips['contact_time'])


def display_empty_metrics(placeholders_recent: Dict[str, Any], placeholders_overall: Dict[str, Any], 
                         tooltips: Dict[str, str]):
    """
    Display empty metric placeholders before first stream.
    
    Args:
        placeholders_recent: Dictionary of recent metric placeholders
        placeholders_overall: Dictionary of overall metric placeholders
        tooltips: Tooltip text dictionary
    """
    for placeholders in [placeholders_recent, placeholders_overall]:
        placeholders['total_strides'].metric("Total Strides", value="--")
        placeholders['cadence'].metric("Cadence (steps/min)", value="--", help=tooltips['cadence'])
        placeholders['stride_variability'].metric("Stride Variability (CV)", value="--", 
                                                  help=tooltips['stride_variability'])
        placeholders['stride_symmetry'].metric("Stride Symmetry", value="--", help=tooltips['stride_symmetry'])
        placeholders['contact_time'].metric("Avg Contact Time", value="--", help=tooltips['contact_time'])
        placeholders['stance_swing_ratio'].metric("Stance/Swing Ratio", value="--", 
                                                  help=tooltips['stance_swing_ratio'])


def calculate_dynamic_x_range(lf_times: list, rf_times: list, window_duration: float, 
                              fallback_start: float) -> Tuple[float, float]:
    """
    Calculate dynamic x-axis range for scrolling window display.
    
    Args:
        lf_times: Left foot time values
        rf_times: Right foot time values
        window_duration: Duration of the display window in seconds
        fallback_start: Fallback start time if no data available
        
    Returns:
        Tuple of (x_min, x_max)
    """
    if lf_times and rf_times:
        current_max_time = max(max(lf_times), max(rf_times))
        return current_max_time - window_duration, current_max_time
    return fallback_start, fallback_start + window_duration
