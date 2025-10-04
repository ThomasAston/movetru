"""
Adaptive, Speed-Aware Gait Event Detector (Causal, NumPy-only)
----------------------------------------------------------------

What this module does (very short):
- Tracks cadence online and scales timing with speed (walk → run → sprint).
- Uses adaptive, rolling z-score thresholds + hysteresis for robust zero-crossings.
- Fuses sagittal gyro with vertical accel to decide Foot-Strike (FS) and Foot-Off (FO).
- Emits events with confidence scores and stride-normalized timing.

How to use (minimal):
---------------------
1) Create the detector:

    det = AdaptiveGaitDetector(fs=200.0)  # set your IMU sample rate (Hz)

2) Stream samples (t in seconds, gyro [gx,gy,gz], accel [ax,ay,az]):

    out = det.update(t, gyro, accel)
    if out.events:  # list of (event_type, t_event, confidence)
        for ev in out.events:
            print(ev)

3) Optionally, poll state summaries:

    cadence_hz = det.cadence_hz
    stride_time = det.stride_time
    stance_fraction = det.expected_stance_fraction

Notes:
- For best results, map gyroscope Y to sagittal plane rotation. If unsure, try your best axis (often gyroy on shank/foot). The code also works reasonably with |gyro|.
- All logic is causal and suitable for real-time.
- No SciPy required; just NumPy.

"""
from dataclasses import dataclass
from typing import List, Tuple, Optional, Deque
from collections import deque
import numpy as np

# ---------------------------
# Small utilities (causal filters)
# ---------------------------
class EWMA:
    """Exponentially weighted moving average/variance for rolling stats."""
    def __init__(self, alpha_mean: float = 0.02, alpha_var: float = 0.02):
        self.alpha_m = float(alpha_mean)
        self.alpha_v = float(alpha_var)
        self._m = None
        self._v = None

    def update(self, x: float) -> Tuple[float, float]:
        if self._m is None:
            # initialize conservatively
            self._m = float(x)
            self._v = 1e-6
        else:
            dm = x - self._m
            self._m += self.alpha_m * dm
            # EW variance of residual
            self._v = (1 - self.alpha_v) * self._v + self.alpha_v * (dm * dm)
        return self._m, max(self._v, 1e-12)

class OnePole:
    """Simple one-pole IIR low-pass, y[n] = y[n-1] + a*(x-y[n-1])"""
    def __init__(self, alpha: float):
        self.a = float(alpha)
        self.y = None

    def update(self, x: float) -> float:
        if self.y is None:
            self.y = float(x)
        else:
            self.y = self.y + self.a * (x - self.y)
        return self.y

# ---------------------------
# Cadence tracker (online)
# ---------------------------
class CadenceTracker:
    """Track step frequency using inter-peak timing on |gyro_sagittal|.
    Robust EWMA of step interval → cadence. Works for walk→sprint.
    """
    def __init__(self, fs: float, min_f: float = 0.5, max_f: float = 4.5,
                 ew_alpha: float = 0.15):
        self.fs = fs
        self.min_s = 1.0 / max_f  # shortest step interval
        self.max_s = 1.0 / min_f  # longest step interval
        self.ew_alpha = float(ew_alpha)
        self._last_peak_t: Optional[float] = None
        self._cadence_hz: Optional[float] = None
        # Peak detection helpers
        self.prev = None
        self.rising = False
        self.prom_lpf = OnePole(alpha=0.05)  # adaptive prominence baseline

    @property
    def cadence_hz(self) -> Optional[float]:
        return self._cadence_hz

    def update(self, t: float, x: float) -> Optional[float]:
        """Feed absolute sagittal gyro (or magnitude). Returns cadence Hz if updated."""
        # Simple local peak logic with hysteresis via slope flips and prominence
        if self.prev is None:
            self.prev = x
            return self._cadence_hz

        slope = x - self.prev
        self.prev = x
        # Track whether we're in a rising segment
        if slope > 0:
            self.rising = True
        elif slope < 0 and self.rising:
            # Potential peak at current sample (approximate). Use x as peak magnitude.
            self.rising = False
            prom_base = self.prom_lpf.update(x)
            prominence_ok = (x - prom_base) > 0.15 * max(1.0, prom_base)
            if prominence_ok and self._last_peak_t is not None:
                dt = t - self._last_peak_t
                if self.min_s <= dt <= self.max_s:
                    step_hz = 1.0 / dt
                    if self._cadence_hz is None:
                        self._cadence_hz = step_hz
                    else:
                        self._cadence_hz = (1 - self.ew_alpha) * self._cadence_hz + self.ew_alpha * step_hz
            self._last_peak_t = t
        return self._cadence_hz

# ---------------------------
# Event container
# ---------------------------
@dataclass
class DetectorOutput:
    events: List[Tuple[str, float, float]]  # ("FS"/"FO", time, confidence)
    cadence_hz: Optional[float]
    stride_time: Optional[float]
    expected_stance_fraction: float

# ---------------------------
# Main detector
# ---------------------------
class AdaptiveGaitDetector:
    """Causal, speed-aware FS/FO detector using adaptive thresholds and hysteresis.

    Inputs: timestamp t (s), gyro (gx,gy,gz) [rad/s], accel (ax,ay,az) [m/s^2].
    Assumes gy[1] ~ sagittal rotation. If not, set use_abs_gyro=True.
    """
    def __init__(self, fs: float,
                 use_abs_gyro: bool = False,
                 z_hyst_sigma: float = 1.0,
                 accel_z_gate_sigma: float = 1.5,
                 ewma_mean_alpha: float = 0.02,
                 ewma_var_alpha: float = 0.02):
        self.fs = float(fs)
        self.use_abs = bool(use_abs_gyro)
        # Rolling stats for gyro channel (sagittal) and vertical accel
        self.gy_stats = EWMA(alpha_mean=ewma_mean_alpha, alpha_var=ewma_var_alpha)
        self.az_stats = EWMA(alpha_mean=ewma_mean_alpha, alpha_var=ewma_var_alpha)
        self.cad = CadenceTracker(fs)
        # Hysteresis scale (in sigmas)
        self.z_hyst_sigma = float(z_hyst_sigma)
        self.accel_z_gate_sigma = float(accel_z_gate_sigma)
        # State machine
        self._state = "INIT"  # INIT → SWING → STANCE
        self._last_fs_t: Optional[float] = None
        self._last_fo_t: Optional[float] = None
        self._last_event_t: Optional[float] = None
        self._events: Deque[Tuple[str, float, float]] = deque(maxlen=8)
        # Hysteresis levels update each sample
        self._h = 0.0
        self._below = False
        self._above = False
        # Expected stance fraction model params
        # Rough mapping: stance% ~ a - b * cadence (Hz), clamped [0.2, 0.65]
        self._a = 0.62
        self._b = 0.05
        # Filters for vertical accel baseline (gravity removal)
        self.az_lp = OnePole(alpha=0.01)

    # -----------------------
    # Public properties
    # -----------------------
    @property
    def cadence_hz(self) -> Optional[float]:
        return self.cad.cadence_hz

    @property
    def stride_time(self) -> Optional[float]:
        c = self.cadence_hz
        return (2.0 / c) if (c and c > 1e-6) else None

    @property
    def expected_stance_fraction(self) -> float:
        c = self.cadence_hz or 1.8  # default moderate jog
        stance = self._a - self._b * c
        return float(min(0.65, max(0.20, stance)))

    # -----------------------
    # Internal helpers
    # -----------------------
    def _timing_params(self) -> Tuple[float, float, float, float]:
        """Return (min_stride, max_stride, fs_refractory, fo_refractory) in seconds, cadence-scaled."""
        c = self.cadence_hz or 1.8
        Tstep = 1.0 / c
        Tstride = 2.0 * Tstep
        # plausible stride bounds
        min_stride = 0.6 * Tstride
        max_stride = 1.6 * Tstride
        # refractory windows scale with stride
        fs_refrac = 0.08 * Tstride  # avoids double FS on impact ringing
        fo_refrac = 0.06 * Tstride
        return min_stride, max_stride, fs_refrac, fo_refrac

    def _update_hyst(self, gy_f: float) -> Tuple[bool, bool]:
        """Update hysteresis bands around zero using rolling sigma. Returns (zc_up, zc_down)."""
        m, v = self.gy_stats._m or 0.0, self.gy_stats._v or 1e-6
        sigma = np.sqrt(v)
        self._h = self.z_hyst_sigma * max(1e-4, sigma)
        # Track region
        zc_up = False
        zc_down = False
        if gy_f > self._h:
            if self._below:
                zc_up = True
            self._above, self._below = True, False
        elif gy_f < -self._h:
            if self._above:
                zc_down = True
            self._above, self._below = False, True
        # else in-between band; keep flags but no crossing
        return zc_up, zc_down

    def _confidence(self, t: float, event_type: str, gy_prom: float, az_gate_ok: bool) -> float:
        # Base on (prominence proxy), accel gate, timing consistency
        c1 = np.tanh(gy_prom)  # [0,1)
        c2 = 0.25 if az_gate_ok else 0.0
        # timing
        conf_t = 0.25
        if self._last_fs_t and self._last_fo_t and self.stride_time:
            # check stance fraction plausibility
            if event_type == "FO":
                # estimate tentative stance so far
                stance_est = (t - self._last_fs_t) / max(1e-6, self.stride_time)
                err = abs(stance_est - self.expected_stance_fraction)
                conf_t = max(0.05, 0.25 - 0.5 * err)
            elif event_type == "FS" and self._last_fs_t:
                stride_est = t - self._last_fs_t
                min_s, max_s, *_ = self._timing_params()
                if min_s <= stride_est <= max_s:
                    conf_t = 0.25
                else:
                    conf_t = 0.05
        return float(min(1.0, 0.5 * c1 + c2 + conf_t))

    # -----------------------
    # Main update
    # -----------------------
    def update(self, t: float, gyro: np.ndarray, accel: np.ndarray) -> DetectorOutput:
        gx, gy, gz = float(gyro[0]), float(gyro[1]), float(gyro[2])
        ax, ay, az = float(accel[0]), float(accel[1]), float(accel[2])

        # Choose gyro channel: sagittal if available, else magnitude
        gy_raw = abs(gy) if self.use_abs else gy

        # Remove gravity baseline from az (very light LPF to follow slow gravity drift)
        az_baseline = self.az_lp.update(az)
        az_hf = az - az_baseline

        # Update rolling stats (for adaptive sigma)
        gy_m, gy_v = self.gy_stats.update(gy_raw)
        az_m, az_v = self.az_stats.update(az_hf)
        gy_sigma = np.sqrt(gy_v)
        az_sigma = np.sqrt(az_v)

        # Update cadence from |gyro| peaks (robust against sign)
        self.cad.update(t, abs(gy_raw))

        # Hysteretic zero-cross detection around zero on gyro
        zc_up, zc_down = self._update_hyst(gy_raw)

        # Cadence-scaled timing
        min_s, max_s, fs_refrac, fo_refrac = self._timing_params()

        # Accel gate: require vertical impulse near FS
        az_gate = az_hf > (self.accel_z_gate_sigma * max(1e-4, az_sigma))

        events: List[Tuple[str, float, float]] = []

        # State machine
        if self._state == "INIT":
            # Wait for first plausible FS to sync
            if zc_up and az_gate:
                conf = self._confidence(t, "FS", gy_prom=abs(gy_raw - gy_m) / max(1e-6, gy_sigma), az_gate_ok=True)
                self._events.append(("FS", t, conf))
                events.append(("FS", t, conf))
                self._last_fs_t = t
                self._last_event_t = t
                self._state = "STANCE"

        elif self._state == "SWING":
            # Look for FS: upward ZC + accel gate, respect refractory
            if zc_up and (self._last_event_t is None or (t - self._last_event_t) > fs_refrac):
                if az_gate:
                    conf = self._confidence(t, "FS", gy_prom=abs(gy_raw - gy_m) / max(1e-6, gy_sigma), az_gate_ok=True)
                    self._events.append(("FS", t, conf))
                    events.append(("FS", t, conf))
                    # stride plausibility: if previous FS exists, optionally snap to local extremum (omitted for causality simplicity)
                    self._last_fs_t = t
                    self._last_event_t = t
                    self._state = "STANCE"

        elif self._state == "STANCE":
            # Look for FO: downward ZC and no strong vertical impact needed
            if zc_down and (self._last_event_t is None or (t - self._last_event_t) > fo_refrac):
                conf = self._confidence(t, "FO", gy_prom=abs(gy_raw - gy_m) / max(1e-6, gy_sigma), az_gate_ok=False)
                self._events.append(("FO", t, conf))
                events.append(("FO", t, conf))
                self._last_fo_t = t
                self._last_event_t = t
                self._state = "SWING"

        # (Optional) timeouts to re-sync if state stalls can be added here

        return DetectorOutput(
            events=events,
            cadence_hz=self.cadence_hz,
            stride_time=self.stride_time,
            expected_stance_fraction=self.expected_stance_fraction,
        )


# ---------------------------
# Tiny demo stub (non-executing): how you might use it
# ---------------------------
if __name__ == "__main__":
    # This is just illustrative; keep it commented or adapt it to your data stream.
    import numpy as _np

    fs = 200.0
    det = AdaptiveGaitDetector(fs=fs, use_abs_gyro=False)

    # Example: loop over your buffered samples
    # for i in range(N):
    #     t = i / fs
    #     gyro = your_gyro[i]  # shape (3,)
    #     accel = your_accel[i]  # shape (3,)
    #     out = det.update(t, gyro, accel)
    #     for ev in out.events:
    #         print(ev)
    pass
