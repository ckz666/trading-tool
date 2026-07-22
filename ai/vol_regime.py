"""
Volatility regime detection: 2-state sticky Markov chain (CALM / STORM).

Big moves cluster — a calm market tends to stay calm, a stormy one tends to
stay stormy. That persistence is what a naive "size by today's ATR" approach
throws away. This fits two volatility clusters on the recent window (cheap,
recomputed fresh each call — no persisted model) and runs a forward filter
with a fixed sticky transition matrix to get a smoothed regime probability,
instead of flip-flopping on every noisy bar.

Not a full Baum-Welch HMM fit — the transition matrix is a fixed prior
(calm stays calm ~97%, storm stays storm ~90%), only the two emission
clusters are estimated from data. That's a deliberate simplification: sizing
only needs a stable regime read, not a maximum-likelihood model.
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

# Sticky transition matrix: rows = from-state, cols = to-state, order [calm, storm]
P_TRANSITION = np.array([
    [0.97, 0.03],   # calm -> calm, calm -> storm
    [0.10, 0.90],   # storm -> calm, storm -> storm
])

REGIME_RISK_MULTIPLIER = {"calm": 1.0, "storm": 0.5}


def _fit_two_clusters(x: np.ndarray, iters: int = 15) -> tuple[float, float, float, float]:
    """Cheap 1D 2-means split with per-cluster std — stand-in for Gaussian-mixture
    emission fitting, good enough for a two-bucket calm/storm split."""
    lo, hi = np.percentile(x, 25), np.percentile(x, 75)
    for _ in range(iters):
        d_lo = np.abs(x - lo)
        d_hi = np.abs(x - hi)
        assign_hi = d_hi < d_lo
        if assign_hi.all() or (~assign_hi).all():
            break
        new_lo = x[~assign_hi].mean()
        new_hi = x[assign_hi].mean()
        if np.isclose(new_lo, lo) and np.isclose(new_hi, hi):
            lo, hi = new_lo, new_hi
            break
        lo, hi = new_lo, new_hi
    assign_hi = np.abs(x - hi) < np.abs(x - lo)
    std_lo = x[~assign_hi].std() if (~assign_hi).sum() > 1 else x.std()
    std_hi = x[assign_hi].std() if assign_hi.sum() > 1 else x.std()
    return float(lo), float(max(std_lo, 1e-9)), float(hi), float(max(std_hi, 1e-9))


def _gaussian_pdf(x: float, mean: float, std: float) -> float:
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))


def _bimodality_coefficient(x: np.ndarray) -> float:
    """Sarle's bimodality coefficient. A 2-means split on any unimodal
    distribution still returns two clusters (it's forced to) — that split is
    only trustworthy if the *shape* of the distribution itself is bimodal,
    which this measures directly rather than inferring it from how separated
    the forced clusters happen to look. Uniform ≈ 0.555; well-separated
    bimodal data pushes notably higher; unimodal (incl. Gaussian) sits lower.
    """
    n = len(x)
    if n < 10:
        return 0.0
    g1 = skew(x, bias=False)
    g2 = kurtosis(x, fisher=True, bias=False)  # excess kurtosis, normal = 0
    correction = 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return float((g1 ** 2 + 1) / (g2 + correction))


def classify_vol_regime(df: pd.DataFrame, vol_window: int = 24, lookback: int = 300) -> dict:
    """
    Classify the current bar's volatility regime as calm/storm.

    df: OHLCV with a 'close' column, at least vol_window + ~30 rows.
    vol_window: bars used for the rolling realised-vol measure (24 ≈ 1 day on 1h candles).
    lookback: how much history to use for fitting the two emission clusters.

    Returns {"regime", "prob_storm", "risk_multiplier"}, or a calm/neutral
    default if there isn't enough data yet.
    """
    if len(df) < vol_window + 30:
        return {"regime": "calm", "prob_storm": 0.0, "risk_multiplier": 1.0}

    returns = df["close"].pct_change()
    realised_vol = returns.rolling(vol_window).std().dropna()
    window = realised_vol.iloc[-lookback:]
    if len(window) < 30:
        return {"regime": "calm", "prob_storm": 0.0, "risk_multiplier": 1.0}

    x = window.to_numpy()

    # Bimodality guard: with only one true regime present (e.g. an extended
    # quiet stretch), a forced 2-means split still returns two clusters, and
    # the sticky transition matrix then locks onto whichever half of that
    # single distribution came last and reports it as "storm" with near-
    # certainty — the same artifact-as-structure trap flagged elsewhere in
    # this project (see the spurious-cycle note on FFT-based features).
    # Sarle's bimodality coefficient checks the distribution's actual shape
    # instead of the forced split, so a genuinely unimodal window is
    # rejected before it ever reaches the clustering step.
    if _bimodality_coefficient(x) < 0.6:
        return {"regime": "calm", "prob_storm": 0.0, "risk_multiplier": 1.0}

    calm_mean, calm_std, storm_mean, storm_std = _fit_two_clusters(x)
    if storm_mean < calm_mean:
        calm_mean, calm_std, storm_mean, storm_std = storm_mean, storm_std, calm_mean, calm_std

    # Forward filter (2-state HMM forward algorithm) over the window
    prob = np.array([0.5, 0.5])  # [P(calm), P(storm)] prior at window start
    for v in x:
        prob = prob @ P_TRANSITION
        lik = np.array([
            _gaussian_pdf(v, calm_mean, calm_std),
            _gaussian_pdf(v, storm_mean, storm_std),
        ])
        prob = prob * lik
        total = prob.sum()
        prob = prob / total if total > 0 else np.array([0.5, 0.5])

    prob_storm = float(prob[1])
    regime = "storm" if prob_storm > 0.5 else "calm"
    return {
        "regime": regime,
        "prob_storm": round(prob_storm, 3),
        "risk_multiplier": REGIME_RISK_MULTIPLIER[regime],
    }


def rolling_prob_storm(df: pd.DataFrame, vol_window: int = 24, lookback: int = 300, min_history: int = 60) -> pd.Series:
    """
    Per-bar prob_storm over trailing history, for use as an ML feature column
    (2026-07-22: currently sizing-only via classify_vol_regime; this lets the
    model itself learn regime-conditional patterns instead of just having its
    output de-rated after the fact). Bars before min_history exist are filled
    0.0 (calm default), same warmup convention as the rest of build_features().
    """
    n = len(df)
    out = np.zeros(n)
    close = df["close"]
    for i in range(min_history, n):
        window = df.iloc[max(0, i - lookback + 1): i + 1]
        out[i] = classify_vol_regime(window, vol_window=vol_window, lookback=lookback)["prob_storm"]
    return pd.Series(out, index=df.index)
