import numpy as np
import pandas as pd
import pickle
import os
from collections import Counter
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import ta

from ai.patterns import detect_patterns


MODEL_KEYS = ["gbm", "rf", "et"]


def _tag(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


def _paths(symbol: str):
    tag = _tag(symbol)
    model_paths = {k: f"ai/model_{tag}_{k}.pkl" for k in MODEL_KEYS}
    scaler_path  = f"ai/scaler_{tag}.pkl"
    di_path      = f"ai/di_{tag}.pkl"          # DI Score training stats
    return model_paths, scaler_path

def _di_path(symbol: str) -> str:
    return f"ai/di_{_tag(symbol)}.pkl"


def _legacy_path(symbol: str):
    """Old single-model path for backward compat."""
    tag = _tag(symbol)
    return f"ai/model_{tag}.pkl", f"ai/scaler_{tag}.pkl"


def _funding_to_series(funding_records: list) -> pd.Series:
    """Convert list of ccxt funding rate records to hourly pandas Series."""
    if not funding_records:
        return pd.Series(dtype=float)
    rows = [(pd.Timestamp(r["timestamp"], unit="ms", tz="UTC"), r["fundingRate"]) for r in funding_records]
    s = pd.Series({ts: rate for ts, rate in rows}).sort_index()
    s.index = s.index.tz_localize(None)
    return s.resample("1h").last().ffill()


def _squeeze_indicators(df: pd.DataFrame, period: int = 20,
                         bb_mult: float = 2.0, kc_mult: float = 1.5) -> pd.DataFrame:
    """
    TTM-style Volatility Squeeze.
    squeeze_active=1: BB inside Keltner Channel → compression, explosive move incoming.
    squeeze_fired=1:  squeeze just released (prev bar ON, current bar OFF) → entry signal.
    squeeze_momentum: positive=bullish release, negative=bearish release.
    """
    close, high, low = df["close"], df["high"], df["low"]
    ma  = close.rolling(period).mean()
    std = close.rolling(period).std()

    # Bollinger Bands
    bb_upper = ma + bb_mult * std
    bb_lower = ma - bb_mult * std

    # Keltner Channel (ATR-based)
    tr  = pd.concat([high - low,
                     (high - close.shift(1)).abs(),
                     (low  - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    kc_upper = ma + kc_mult * atr
    kc_lower = ma - kc_mult * atr

    squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)
    fired   = squeeze.shift(1).fillna(False) & ~squeeze   # was ON last bar, now OFF

    # Momentum: close vs midpoint of recent range + MA
    highest = high.rolling(period).max()
    lowest  = low.rolling(period).min()
    momentum = close - ((highest + lowest) / 2 + ma) / 2
    momentum_norm = momentum / close   # normalize by price

    return pd.DataFrame({
        "squeeze_active":   squeeze.astype(int),
        "squeeze_fired":    fired.astype(int),
        "squeeze_momentum": momentum_norm,
    }, index=df.index)


def _resample_htf_indicators(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample 1h OHLC to a higher timeframe (derived from the same data, no
    extra fetch) and compute a few indicators on it. Shifted by one period
    before the caller reindexes it back onto the 1h index, so each 1h bar only
    ever sees the last fully-closed higher-timeframe bar — never the one it's
    currently inside of (that would leak the future into the training label).
    """
    o = df["open"].resample(rule).first()
    h = df["high"].resample(rule).max()
    l = df["low"].resample(rule).min()
    c = df["close"].resample(rule).last()
    htf = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}).dropna()

    # ADX needs >= 2x its window (28 for window=14) or the ta library indexes
    # past the end of the series instead of returning NaN — verified empirically,
    # not documented. Callers with a small df (e.g. Grid's fetch) would otherwise crash here.
    if len(htf) < 30:
        empty_idx = pd.DatetimeIndex([], name=df.index.name)
        return pd.DataFrame({"rsi": pd.Series(dtype=float, index=empty_idx),
                              "adx": pd.Series(dtype=float, index=empty_idx),
                              "ema_cross_norm": pd.Series(dtype=float, index=empty_idx),
                              "macd_diff": pd.Series(dtype=float, index=empty_idx)})

    rsi  = ta.momentum.RSIIndicator(htf["close"]).rsi()
    adx  = ta.trend.ADXIndicator(htf["high"], htf["low"], htf["close"], 14).adx()
    ema9, ema21 = ta.trend.ema_indicator(htf["close"], 9), ta.trend.ema_indicator(htf["close"], 21)
    macd_diff = ta.trend.MACD(htf["close"]).macd_diff()

    out = pd.DataFrame({
        "rsi":            rsi,
        "adx":            adx,
        "ema_cross_norm": (ema9 - ema21) / htf["close"],
        "macd_diff":      macd_diff,
    })
    return out.shift(1)


def _pattern_signal(df: pd.DataFrame) -> pd.Series:
    """Rolling candlestick-pattern signal: bullish minus bearish pattern count
    over the trailing 5-candle window ending at each bar. Reuses the same
    pattern detector the live confluence scorer uses, so the ML model can
    learn nonlinear combinations of the same signal instead of only getting
    it as fixed confluence points."""
    signal = pd.Series(0.0, index=df.index)
    for i in range(4, len(df)):
        pats = detect_patterns(df.iloc[i - 4: i + 1])
        bulls = sum(1 for v in pats.values() if v == "bullish")
        bears = sum(1 for v in pats.values() if v == "bearish")
        signal.iloc[i] = bulls - bears
    return signal


def build_features(df: pd.DataFrame, funding_series: pd.Series = None) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)

    # Momentum (3 core features, non-redundant)
    f["returns_1"]     = df["close"].pct_change(1)
    f["returns_3"]     = df["close"].pct_change(3)
    f["rsi"]           = ta.momentum.RSIIndicator(df["close"]).rsi()

    # Trend (3 features: direction + strength + MACD histogram)
    ema_9              = ta.trend.ema_indicator(df["close"], 9)
    ema_21             = ta.trend.ema_indicator(df["close"], 21)
    f["ema_cross_norm"]= (ema_9 - ema_21) / df["close"]
    f["macd_diff"]     = ta.trend.MACD(df["close"]).macd_diff()
    adx_ind            = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], 14)
    f["adx"]           = adx_ind.adx()

    # Volatility (2 features: position in BB + regime)
    bb                 = ta.volatility.BollingerBands(df["close"])
    f["bb_pct"]        = bb.bollinger_pband()
    f["atr_norm"]      = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range() / df["close"]

    # Volume (2 features: relative volume + OBV flow)
    vol_sma            = df["volume"].rolling(14).mean()
    f["volume_ratio"]  = df["volume"] / vol_sma
    obv                = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
    f["obv_norm"]      = obv.pct_change(5)

    # VWAP distance (1 feature: price vs fair value)
    typical            = (df["high"] + df["low"] + df["close"]) / 3
    vwap               = (typical * df["volume"]).rolling(24).sum() / df["volume"].rolling(24).sum()
    f["vwap_dist"]     = (df["close"] - vwap) / df["close"]

    # Volatility Squeeze (2 features: compression state + momentum direction)
    sq = _squeeze_indicators(df)
    f["squeeze_active"]   = sq["squeeze_active"]
    f["squeeze_momentum"] = sq["squeeze_momentum"]

    # Funding rate (contrarian on-chain feature — positive = longs overextended = bearish).
    # Bitget's funding history caps at ~100 records (~33 days) regardless of what's
    # requested, far short of the price history we can get elsewhere — neutral-fill
    # rows before funding coverage starts instead of dropping them, same reasoning
    # as the 4H-indicator warmup above.
    if funding_series is not None and not funding_series.empty:
        aligned = funding_series.reindex(df.index, method="ffill")
        # Normalize: typical rate 0.01% = 1.0, extreme ±5x = ±5.0
        f["funding_norm"] = ((aligned / 0.0001).clip(-5, 5)).fillna(0.0)
        # Trend: 3-period change (are longs paying more or less?)
        f["funding_trend"] = (aligned.diff(3) / 0.0001).fillna(0.0)
    else:
        f["funding_norm"]  = 0.0
        f["funding_trend"] = 0.0

    # 4H context (4 features: resampled from this same 1h data, no extra fetch —
    # lets the model see whether the bigger-picture trend agrees with the 1h read).
    # Neutral-filled rather than left NaN during the 4H-indicator warmup window,
    # so early rows aren't dropped wholesale from an already-small dataset.
    htf_4h = _resample_htf_indicators(df, "4h")
    aligned_4h = htf_4h.reindex(df.index, method="ffill")
    f["rsi_4h"]            = aligned_4h["rsi"].fillna(50.0)
    f["adx_4h"]            = aligned_4h["adx"].fillna(0.0)
    f["ema_cross_norm_4h"] = aligned_4h["ema_cross_norm"].fillna(0.0)
    f["macd_diff_4h"]      = aligned_4h["macd_diff"].fillna(0.0)

    # Candlestick pattern signal (1 feature): lets the ensemble learn nonlinear
    # combinations of the same patterns the confluence scorer already checks
    f["pattern_signal"] = _pattern_signal(df)

    return f


def make_labels(df: pd.DataFrame, horizon: int = 3, threshold: float = None) -> pd.Series:
    """1=buy, -1=sell, 0=hold based on future returns.
    If threshold is None, uses ATR-adaptive threshold (0.8x ATR) so labels
    reflect meaningful moves relative to current volatility — not fixed noise.
    """
    future_return = df["close"].shift(-horizon) / df["close"] - 1
    if threshold is None:
        # ATR over 14 periods, normalised by close → volatility-relative threshold
        high_low  = df["high"] - df["low"]
        high_prev = (df["high"] - df["close"].shift(1)).abs()
        low_prev  = (df["low"]  - df["close"].shift(1)).abs()
        tr  = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        thr = (atr / df["close"]) * 0.8   # 80% of one ATR — filters out noise
        thr = thr.clip(lower=0.004, upper=0.03)   # min 0.4%, max 3%
    else:
        thr = threshold
    labels = pd.Series(0, index=df.index)
    labels[future_return > thr] = 1
    labels[future_return < -thr] = -1
    return labels


_train_progress: dict[str, int] = {}


def train(df: pd.DataFrame, symbol: str = "BTC/USDT",
          n_estimators: int = 200, progress_cb=None,
          funding_series: pd.Series = None) -> dict:
    """
    Train ensemble of 3 models (GBM warm_start + RF + ExtraTrees).
    Progress: GBM batches 20/40/60%, RF 80%, ET 100%.
    """
    model_paths, scaler_path = _paths(symbol)
    features = build_features(df, funding_series=funding_series)
    labels   = make_labels(df)

    valid = features.dropna().index.intersection(labels.dropna().index)
    valid = valid[:-3]
    X = features.loc[valid]
    y = labels.loc[valid]

    # ── Outlier removal (IQR-based, threshold scales with how much data we have —
    # plenty of samples → filter harder; scarce samples → be lenient, we can't
    # afford to throw away rows a thin-history symbol doesn't have to spare) ──
    iqr_mult = 4.5 if len(X) >= 300 else 7.0
    q1 = X.quantile(0.25); q3 = X.quantile(0.75); iqr = q3 - q1
    mask = ~((X < (q1 - iqr_mult * iqr)) | (X > (q3 + iqr_mult * iqr))).any(axis=1)
    X = X[mask]; y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── DI Score: FreqAI-style pairwise distance approach ────────────────────────
    # Save training matrix + avg pairwise dist so predict can compare.
    # DI = min_distance(x_pred, X_train) / avg_pairwise_train_dist
    # DI > threshold → regime shift outside training distribution → HOLD
    from sklearn.metrics import pairwise_distances as _pw
    pw_dists = _pw(X_train_s)           # (n, n) euclidean
    np.fill_diagonal(pw_dists, np.nan)
    avg_mean_dist = float(np.nanmean(pw_dists))
    di_stats = {
        "X_train": X_train_s,           # saved for min-distance lookup at predict time
        "avg_mean_dist": avg_mean_dist,
        "cols": list(X_train.columns),
    }
    os.makedirs("ai", exist_ok=True)
    with open(_di_path(symbol), "wb") as fh: pickle.dump(di_stats, fh)

    _train_progress[symbol] = 0

    # ── sample weights: class balancing × temporal recency (FreqAI wfactor) ──────
    # w_i = exp(-i / (wfactor * N))[::-1] → newest sample has highest weight
    # wfactor=0.9 → newest ~2.95x more weight than oldest
    from collections import Counter
    cls_counts = Counter(y_train)
    total = len(y_train)
    class_w = np.array([total / (len(cls_counts) * cls_counts[c]) for c in y_train])
    n = len(y_train)
    # Less temporal decay for small datasets — avoids overfitting to recent regime
    wfactor = 0.9 if n >= 400 else 0.4
    time_w  = np.exp(-np.arange(n) / (wfactor * n))[::-1]
    sw = class_w * time_w
    sw /= sw.mean()

    # ── GBM with warm_start (batched for progress reporting) ──────────────────
    batch = max(n_estimators // 3, 40)
    gbm = GradientBoostingClassifier(
        n_estimators=batch, max_depth=4,
        learning_rate=0.05, random_state=42, warm_start=True,
    )
    gbm.fit(X_train_s, y_train, sample_weight=sw)
    _train_progress[symbol] = 20
    if progress_cb: progress_cb(20)
    for step in range(1, 3):
        gbm.n_estimators += batch
        gbm.fit(X_train_s, y_train, sample_weight=sw)
        pct = 20 + step * 20
        _train_progress[symbol] = pct
        if progress_cb: progress_cb(pct)

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=10,
        min_samples_leaf=3, random_state=42, n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train_s, y_train)
    _train_progress[symbol] = 80
    if progress_cb: progress_cb(80)

    # ── Extra Trees ───────────────────────────────────────────────────────────
    et = ExtraTreesClassifier(
        n_estimators=n_estimators, max_depth=10,
        min_samples_leaf=3, random_state=42, n_jobs=-1,
        class_weight="balanced",
    )
    et.fit(X_train_s, y_train)
    _train_progress[symbol] = 100
    if progress_cb: progress_cb(100)

    # ── Ensemble evaluation ───────────────────────────────────────────────────
    votes = np.array([gbm.predict(X_test_s), rf.predict(X_test_s), et.predict(X_test_s)])
    ensemble_preds = np.array([Counter(votes[:, i]).most_common(1)[0][0] for i in range(votes.shape[1])])

    # balanced_accuracy weights each class equally → not fooled by hold-majority
    bal_acc = float(balanced_accuracy_score(y_test.values, ensemble_preds))
    raw_acc = float(np.mean(ensemble_preds == y_test.values))
    f1_mac  = float(f1_score(y_test.values, ensemble_preds, average="macro", zero_division=0))

    # directional precision: of all buy/sell predictions, how often correct?
    dir_mask = ensemble_preds != 0
    dir_precision = float(np.mean(ensemble_preds[dir_mask] == y_test.values[dir_mask])) if dir_mask.any() else 0.0

    os.makedirs("ai", exist_ok=True)
    for key, model in [("gbm", gbm), ("rf", rf), ("et", et)]:
        with open(model_paths[key], "wb") as f: pickle.dump(model, f)
    with open(scaler_path, "wb") as f: pickle.dump(scaler, f)

    return {
        "symbol":         symbol,
        "accuracy":       round(bal_acc, 3),     # balanced — the real metric
        "raw_accuracy":   round(raw_acc, 3),      # kept for reference
        "f1_macro":       round(f1_mac, 3),
        "dir_precision":  round(dir_precision, 3),
        "samples":        len(X_train),
        "features":       list(X.columns),
        "models":         ["GBM", "RF", "ExtraTrees"],
    }


def predict(df: pd.DataFrame, symbol: str = "BTC/USDT",
            funding_series: pd.Series = None) -> dict:
    """Ensemble vote from 3 models. Falls back to legacy single model if needed."""
    model_paths, scaler_path = _paths(symbol)

    # Try ensemble first
    if all(os.path.exists(p) for p in model_paths.values()) and os.path.exists(scaler_path):
        return _predict_ensemble(df, model_paths, scaler_path, funding_series=funding_series, symbol=symbol)

    # Fallback to old single-model
    legacy_model, legacy_scaler = _legacy_path(symbol)
    if os.path.exists(legacy_model) and os.path.exists(legacy_scaler):
        return _predict_single(df, legacy_model, legacy_scaler)

    return {"signal": 0, "confidence": 0.0, "label": "hold", "trained": False, "agreement": 0.0}


def _compute_di(X_scaled: np.ndarray, symbol: str) -> float:
    """
    FreqAI-style DI Score: min euclidean distance from current features to any training sample,
    normalised by average pairwise distance within training set.
    DI > 1.0 = current bar is farther from training data than training points are from each other.
    """
    path = _di_path(symbol)
    if not os.path.exists(path):
        return 0.0
    with open(path, "rb") as fh:
        stats = pickle.load(fh)
    X_train = stats.get("X_train")
    avg_dist = stats.get("avg_mean_dist", 1.0)
    if X_train is None or X_scaled.shape[1] != X_train.shape[1] or avg_dist == 0:
        return 0.0
    from sklearn.metrics import pairwise_distances as _pw
    dists = _pw(X_train, X_scaled)   # (n_train, 1)
    min_dist = float(dists.min())
    return min_dist / avg_dist


def _predict_ensemble(df: pd.DataFrame, model_paths: dict, scaler_path: str,
                      funding_series: pd.Series = None, symbol: str = "") -> dict:
    models = {}
    for key, path in model_paths.items():
        with open(path, "rb") as f: models[key] = pickle.load(f)
    with open(scaler_path, "rb") as f: scaler = pickle.load(f)

    features = build_features(df, funding_series=funding_series)
    last = features.iloc[[-1]].dropna(axis=1)
    expected_cols = scaler.feature_names_in_
    for col in expected_cols:
        if col not in last.columns:
            last[col] = 0.0
    last = last[expected_cols]
    X = scaler.transform(last)

    # ── DI Score: if market regime is too far from training → force HOLD ────────
    # Threshold scales with sample count: few samples → DI metric is noisy → be lenient
    _n_samples = 0
    _di_path_check = _di_path(symbol)
    if os.path.exists(_di_path_check):
        import pickle as _pkl
        with open(_di_path_check, "rb") as _fh:
            _xt = _pkl.load(_fh).get("X_train")
            if _xt is not None:
                _n_samples = _xt.shape[0]
    if _n_samples < 300:
        DI_THRESHOLD = 25.0   # too few samples to trust DI → very lenient
    elif _n_samples < 600:
        DI_THRESHOLD = 5.0    # moderate
    else:
        DI_THRESHOLD = 2.0    # well-trained model → strict
    di_score = _compute_di(X, symbol)
    if di_score > DI_THRESHOLD:
        return {
            "signal": 0, "confidence": 0.0, "label": "hold",
            "trained": True, "agreement": 0.0,
            "di_score": round(di_score, 3),
            "di_blocked": True,
            "reason": f"DI={di_score:.2f} > {DI_THRESHOLD} — Regime-Shift erkannt, kein Trade",
        }

    votes = []
    probas = []
    for m in models.values():
        v = int(m.predict(X)[0])
        p = m.predict_proba(X)[0]
        # Ensure 3-class probability order: [-1, 0, 1]
        class_order = list(m.classes_)
        p_aligned = [0.0, 0.0, 0.0]  # [sell, hold, buy]
        for cls_i, cls_val in enumerate(class_order):
            if cls_val == -1: p_aligned[0] = p[cls_i]
            elif cls_val == 0: p_aligned[1] = p[cls_i]
            elif cls_val == 1: p_aligned[2] = p[cls_i]
        votes.append(v)
        probas.append(p_aligned)

    cnt = Counter(votes)
    avg_proba = np.mean(probas, axis=0)

    # Use probability argmax as signal — more robust than majority vote when
    # models are hold-biased due to class imbalance in training data.
    signal = [-1, 0, 1][int(np.argmax(avg_proba))]
    agreement = cnt.get(signal, 0) / len(votes)

    # Confidence = max avg probability * agreement-factor
    confidence = float(max(avg_proba)) * (0.6 + 0.4 * agreement)

    label_map = {1: "buy", -1: "sell", 0: "hold"}
    return {
        "signal": signal,
        "confidence": round(confidence, 3),
        "label": label_map.get(signal, "hold"),
        "trained": True,
        "agreement": round(agreement, 2),
        "di_score": round(di_score, 3),
        "di_blocked": False,
        "votes": {"gbm": label_map.get(votes[0]), "rf": label_map.get(votes[1]), "et": label_map.get(votes[2])},
        "probabilities": {
            "sell": round(float(avg_proba[0]), 3),
            "hold": round(float(avg_proba[1]), 3),
            "buy":  round(float(avg_proba[2]), 3),
        },
    }


def _predict_single(df: pd.DataFrame, model_path: str, scaler_path: str) -> dict:
    with open(model_path,  "rb") as f: model  = pickle.load(f)
    with open(scaler_path, "rb") as f: scaler = pickle.load(f)
    features = build_features(df)
    last = features.iloc[[-1]].dropna(axis=1)
    expected_cols = scaler.feature_names_in_
    for col in expected_cols:
        if col not in last.columns: last[col] = 0.0
    last = last[expected_cols]
    X = scaler.transform(last)
    signal = int(model.predict(X)[0])
    proba  = model.predict_proba(X)[0]
    confidence = float(max(proba))
    label_map = {1: "buy", -1: "sell", 0: "hold"}
    return {
        "signal": signal,
        "confidence": round(confidence, 3),
        "label": label_map.get(signal, "hold"),
        "trained": True,
        "agreement": 1.0,
        "probabilities": {
            "sell": round(float(proba[0]), 3),
            "hold": round(float(proba[1]), 3),
            "buy":  round(float(proba[2]), 3),
        },
    }


def detect_market_structure(df: pd.DataFrame, n: int = 5, min_swing_atr: float = 0.5) -> dict:
    """
    Identify swing highs/lows (pivot points) to classify market structure:
    uptrend (HH+HL), downtrend (LL+LH), expanding, contracting, or sideways.
    n = candles on each side required to confirm a pivot.
    min_swing_atr = minimum pivot-to-pivot move, as a multiple of the recent
    average candle range, required to count as meaningfully higher/lower —
    without this, noise-level wiggles inside a flat range register as a full
    "trend" just as readily as a real move would.
    """
    if len(df) < n * 2 + 4:
        return {"trend": "unknown", "last_swing_high": 0.0, "last_swing_low": 0.0,
                "pivot_highs": [], "pivot_lows": []}

    highs = df["high"].values
    lows  = df["low"].values

    pivot_highs: list[float] = []
    pivot_lows:  list[float] = []

    for i in range(n, len(df) - n):
        window_h = highs[i - n: i + n + 1]
        window_l = lows[i - n: i + n + 1]
        if highs[i] == window_h.max():
            pivot_highs.append(float(highs[i]))
        if lows[i] == window_l.min():
            pivot_lows.append(float(lows[i]))

    min_swing = float((df["high"] - df["low"]).tail(20).mean()) * min_swing_atr

    trend = "sideways"
    if len(pivot_highs) >= 2 and len(pivot_lows) >= 2:
        hh = pivot_highs[-1] > pivot_highs[-2] + min_swing  # higher high
        hl = pivot_lows[-1]  > pivot_lows[-2]  + min_swing  # higher low
        lh = pivot_highs[-1] < pivot_highs[-2] - min_swing  # lower high
        ll = pivot_lows[-1]  < pivot_lows[-2]  - min_swing  # lower low

        if hh and hl:   trend = "uptrend"
        elif lh and ll: trend = "downtrend"
        elif hh and ll: trend = "expanding"
        elif lh and hl: trend = "contracting"

    return {
        "trend": trend,
        "last_swing_high": pivot_highs[-1] if pivot_highs else 0.0,
        "last_swing_low":  pivot_lows[-1]  if pivot_lows  else 0.0,
        "pivot_highs": pivot_highs[-3:],
        "pivot_lows":  pivot_lows[-3:],
    }


def calc_ichimoku(df: pd.DataFrame, tenkan_n: int = 9, kijun_n: int = 26,
                   senkou_b_n: int = 52, displacement: int = 26) -> dict:
    """
    Ichimoku Cloud. Senkou spans are shifted forward by `displacement` so that
    iloc[-1] reflects the cloud boundary aligned with the current candle
    (i.e. computed from data `displacement` bars ago, as per standard usage).
    """
    if len(df) < senkou_b_n + displacement:
        return {"available": False}

    high, low, close = df["high"], df["low"], df["close"]
    tenkan = (high.rolling(tenkan_n).max() + low.rolling(tenkan_n).min()) / 2
    kijun  = (high.rolling(kijun_n).max()  + low.rolling(kijun_n).min())  / 2
    senkou_a = ((tenkan + kijun) / 2).shift(displacement)
    senkou_b = ((high.rolling(senkou_b_n).max() + low.rolling(senkou_b_n).min()) / 2).shift(displacement)

    span_a, span_b = senkou_a.iloc[-1], senkou_b.iloc[-1]
    if pd.isna(span_a) or pd.isna(span_b):
        return {"available": False}

    price = float(close.iloc[-1])
    cloud_top, cloud_bottom = float(max(span_a, span_b)), float(min(span_a, span_b))

    if price > cloud_top:
        position = "above"
    elif price < cloud_bottom:
        position = "below"
    else:
        position = "inside"

    tk_cross = "none"
    if len(tenkan) >= 2 and not pd.isna(tenkan.iloc[-2]) and not pd.isna(kijun.iloc[-2]):
        prev_diff = tenkan.iloc[-2] - kijun.iloc[-2]
        cur_diff  = tenkan.iloc[-1] - kijun.iloc[-1]
        if prev_diff <= 0 and cur_diff > 0:
            tk_cross = "bullish"
        elif prev_diff >= 0 and cur_diff < 0:
            tk_cross = "bearish"

    return {
        "available": True,
        "tenkan": round(float(tenkan.iloc[-1]), 4),
        "kijun": round(float(kijun.iloc[-1]), 4),
        "cloud_top": round(cloud_top, 4),
        "cloud_bottom": round(cloud_bottom, 4),
        "cloud_bullish": bool(span_a > span_b),   # future cloud color
        "price_vs_cloud": position,
        "tk_cross": tk_cross,
    }


def get_indicators(df: pd.DataFrame) -> dict:
    """Return current indicator values + market structure for context."""
    f = build_features(df)
    last = f.iloc[-1]
    adx = float(last.get("adx", 0))

    # ATR in absolute terms (not normalised) for position sizing
    atr_norm = float(last.get("atr_norm", 0))
    close    = float(df["close"].iloc[-1])
    atr_abs  = atr_norm * close

    ms = detect_market_structure(df)
    ichimoku = calc_ichimoku(df)

    # Squeeze: compute on full df for fired detection (prev bar needed)
    sq     = _squeeze_indicators(df)
    sq_cur = sq.iloc[-1]

    return {
        "rsi":             round(float(last.get("rsi", 0)), 2),
        "macd_diff":       round(float(last.get("macd_diff", 0)), 4),
        "bb_pct":          round(float(last.get("bb_pct", 0.5)), 3),
        "ema_cross_norm":  round(float(last.get("ema_cross_norm", 0)), 5),
        "volume_ratio":    round(float(last.get("volume_ratio", 1)), 2),
        "atr_norm":        round(atr_norm, 5),
        "atr":             round(atr_abs, 4),
        "adx":             round(adx, 1),
        "adx_pos":         round(float(last.get("adx_pos", 0)), 1),
        "adx_neg":         round(float(last.get("adx_neg", 0)), 1),
        "regime":          detect_regime(adx),
        "bb_width":        round(float(last.get("bb_width", 0)), 4),
        "vwap_dist":       round(float(last.get("vwap_dist", 0)), 4),
        "bullish_div":     int(last.get("bullish_div", 0)),
        "bearish_div":     int(last.get("bearish_div", 0)),
        "market_structure":ms["trend"],
        "swing_high":      round(ms["last_swing_high"], 4),
        "swing_low":       round(ms["last_swing_low"], 4),
        "squeeze_active":  int(sq_cur["squeeze_active"]),
        "squeeze_fired":   int(sq_cur["squeeze_fired"]),
        "squeeze_momentum":round(float(sq_cur["squeeze_momentum"]), 5),
        "ichimoku":        ichimoku,
    }


def detect_regime(adx: float) -> str:
    if adx >= 40: return "strong_trend"
    if adx >= 25: return "trending"
    if adx >= 20: return "transitioning"
    return "ranging"
