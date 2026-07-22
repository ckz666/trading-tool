"""
Cycle-strength feature, guarded against the classic DFT artifact.

Running an FFT on a finite price/return window implicitly treats it as one
period of an infinitely repeating signal. If the window's last value doesn't
match its first, that seam is a discontinuity — a sawtooth — and a sawtooth
is built entirely out of low-frequency harmonics. A plain FFT reads that seam
as a "cycle" even on pure white noise; on a random walk it shows up as a
dominant, statistically confident-looking multi-day cycle that carries no
real predictive power (out-of-sample it forecasts worse than a naive
tomorrow-equals-today baseline — that's the entire failure mode this guards
against).

Three fixes applied before any power is measured, in order of how much
each one actually matters:
1. Difference first. Price is an integrated (random-walk) process — its
   power spectrum is inherently tilted toward low frequencies (~k^-2) even
   with zero real structure, purely because integration accumulates noise.
   That tilt alone makes raw price FFT report a strong "cycle" on pure noise;
   detrending or windowing does NOT fix this, because the problem isn't the
   endpoint seam, it's the input being non-stationary. Returns are
   stationary (flat spectrum under the null), so cycle detection has to run
   on the differenced series, never on the level.
2. Detrend the differenced series (removes residual linear drift).
3. Taper with a Hann window, so whatever seam discontinuity remains at the
   window edges is suppressed rather than dumped into the low-frequency
   bins as spurious cycle power.
"""

import numpy as np
import pandas as pd

# Empirical p95 of cycle_strength() on pure random-walk price input at
# window=128 (see tests) — readings at or below this are noise-floor, not signal.
NOISE_FLOOR = 0.10


def cycle_strength(series: pd.Series, window: int = 128) -> float:
    """
    Dominant-cycle power as a fraction of total spectral power, computed on
    the *differenced* trailing `window` bars (see module docstring for why
    that's not optional). Returns 0.0 if there's not enough data or the
    input is degenerate. Values are bounded [0, 1). Calibrated empirically
    at window=128 on differenced random-walk price input: mean ≈0.07,
    p95 ≈0.10 (order-statistics artifact of taking a max over ~64 bins, not
    zero, but well clear of real structure) — a genuine 16-bar sine embedded
    in price scores ≈0.6. NOISE_FLOOR below is that empirical p95; treat
    readings under it as not distinguishable from chance.
    """
    if len(series) < window + 1:
        return 0.0
    x = series.iloc[-(window + 1):].to_numpy(dtype=float)
    if np.allclose(x, x[0]):
        return 0.0

    # 1) difference: removes the random-walk / integration spectral tilt
    diffed = np.diff(x)  # length == window

    # 2) detrend: remove residual linear drift so endpoints roughly match
    t = np.arange(window)
    slope, intercept = np.polyfit(t, diffed, 1)
    detrended = diffed - (slope * t + intercept)

    # 3) taper: Hann window suppresses whatever seam discontinuity remains
    tapered = detrended * np.hanning(window)

    spectrum = np.abs(np.fft.rfft(tapered)) ** 2
    if len(spectrum) < 2:
        return 0.0
    ac_power = spectrum[1:]  # drop DC bin (index 0)
    total = ac_power.sum()
    if total <= 0:
        return 0.0
    dominant = ac_power.max()
    return float(dominant / total)


def rolling_cycle_strength(series: pd.Series, window: int = 128) -> pd.Series:
    """
    Per-bar cycle_strength() over a trailing window, for use as an ML
    feature column. Bars before `window`+1 history exists are filled 0.0
    (noise-floor default) rather than dropped, same warmup convention as
    the rest of build_features().
    """
    n = len(series)
    out = np.zeros(n)
    values = series.to_numpy(dtype=float)
    for i in range(window, n):
        out[i] = cycle_strength(pd.Series(values[i - window: i + 1]), window=window)
    return pd.Series(out, index=series.index)
