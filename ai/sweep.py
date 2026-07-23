"""
Fast parameter sweep for the soft-gate confluence design (2026-07-22/23 experiment,
see project memory / ai/backtest.py module docstring).

Splits the expensive part (ML prediction + confluence scoring per bar — identical
cost no matter which soft-gate parameters we're testing) from the cheap part (trade
decision + PnL simulation, which DOES depend on the parameters):

  compute_signal_table() — trains once, walks forward once per (symbol, fold),
  caches everything the decision layer needs (scores, ML label/conf/agreement,
  ATR, vol-regime risk multiplier). This is the part that costs CPU (ML
  inference, feature engineering).

  simulate_from_table()  — pure-Python decision + PnL loop over a cached table.
  No ML inference, no pandas rolling ops — a few milliseconds per parameter
  combination instead of minutes, so sweeping hundreds of combinations finishes
  in seconds once the tables are built.

Kept separate from ai/backtest.py (which stays the source of truth for the
single-run walk-forward used by the web UI) since this is exploratory sweep
scaffolding, not production backtest code.
"""
import os
from typing import Optional

import numpy as np
import pandas as pd

from ai.backtest import _resample_ohlcv, _atr, _confluence_score_bt
from ai.ml_signal import (
    build_features, train as ml_train, get_indicators, detect_market_structure,
    predict, _paths, _pattern_signal, cvd_zscore_from_ohlcv, taker_ratio_zscore_from_ohlcv,
)
from ai.patterns import detect_patterns
from ai.vol_regime import classify_vol_regime, rolling_prob_storm
from trading.risk import RiskManager


def compute_signal_table(
    df_1h: pd.DataFrame, symbol: str, funding_series: pd.Series = None,
    train_pct: float = 0.70, min_conf: float = 0.40, lookback: int = 300,
    progress_cb=None,
) -> Optional[pd.DataFrame]:
    """Train once, walk forward once, cache everything a sweep combo needs."""
    def _progress(msg):
        if progress_cb:
            try: progress_cb(msg)
            except Exception: pass

    split = int(len(df_1h) * train_pct)
    df_train = df_1h.iloc[:split].copy()
    df_test = df_1h.iloc[split:].copy()

    fs_train = None
    if funding_series is not None and not funding_series.empty:
        fs_train = funding_series[funding_series.index <= df_train.index[-1]]
    ml_train(df_train, symbol, funding_series=fs_train)

    model_paths, scaler_path = _paths(symbol)
    if not all(os.path.exists(p) for p in model_paths.values()):
        return None

    atr_full = _atr(df_1h)
    pattern_signal_full = _pattern_signal(df_1h)
    prob_storm_full = rolling_prob_storm(df_1h)
    # Same precompute-once optimisation as pattern_signal/prob_storm above —
    # both are strictly causal with a fixed trailing lookback (rolling window
    # =100), so a value at a given timestamp is identical whether computed on
    # the full history or a window ending there (2026-07-23, order-flow
    # feature activation round, see project memory).
    cvd_full = cvd_zscore_from_ohlcv(df_1h)
    taker_ratio_full = taker_ratio_zscore_from_ohlcv(df_1h)

    rows = []
    for i in range(len(df_test)):
        abs_i = split + i
        ts = df_test.index[i]
        price = float(df_test.iloc[i]["close"])
        atr = float(atr_full.iloc[abs_i]) if abs_i < len(atr_full) else price * 0.01

        start = max(0, abs_i - lookback)
        window_1h = df_1h.iloc[start:abs_i + 1]
        if len(window_1h) < 50:
            continue

        fs_window = None
        if funding_series is not None and not funding_series.empty:
            fs_window = funding_series[funding_series.index <= ts]

        try:
            feats_1h = build_features(
                window_1h, funding_series=fs_window,
                precomputed_pattern_signal=pattern_signal_full,
                precomputed_prob_storm=prob_storm_full,
                precomputed_cvd_zscore=cvd_full,
                precomputed_taker_ratio_zscore=taker_ratio_full,
            )
            sig = predict(window_1h, symbol, funding_series=fs_window, features=feats_1h)
        except Exception:
            continue

        ml_label = sig.get("label", "hold")
        ml_conf = sig.get("confidence", 0.0)
        ml_agreement = sig.get("agreement", 0.0)

        ind_1h = get_indicators(window_1h, features=feats_1h)
        ind_4h, ind_1d = None, None
        try:
            df_4h_w = _resample_ohlcv(window_1h, "4h")
            if len(df_4h_w) >= 10:
                ms4 = detect_market_structure(df_4h_w)
                ind_4h = {"market_structure": ms4.get("trend", "unknown")}
        except Exception:
            pass
        try:
            df_1d_w = _resample_ohlcv(window_1h, "1D")
            if len(df_1d_w) >= 5:
                ind_1d = get_indicators(df_1d_w)
        except Exception:
            pass
        try:
            pats = detect_patterns(window_1h)
        except Exception:
            pats = {}

        score_long = _confluence_score_bt(True, ind_1h, ind_4h, ind_1d, pats)
        score_short = _confluence_score_bt(False, ind_1h, ind_4h, ind_1d, pats)
        vol_regime = classify_vol_regime(window_1h)

        rows.append({
            "ts": ts, "price": price, "atr": atr,
            "score_long": score_long, "score_short": score_short,
            "ml_label": ml_label, "ml_conf": ml_conf, "ml_agreement": ml_agreement,
            "vol_risk_mult": vol_regime["risk_multiplier"],
        })
        if progress_cb and i % 200 == 0:
            _progress(f"{symbol} signal-table {i}/{len(df_test)}")

    return pd.DataFrame(rows)


def _ml_points(label: str, conf: float, agreement: float, min_conf: float, ml_weight: float):
    if label == "hold" or conf < min_conf or agreement < 0.67:
        return 0, 0
    pts = ml_weight * (2 if agreement >= 0.99 else 1)
    return (pts, 0) if label == "buy" else (0, pts)


def simulate_from_table(
    table: pd.DataFrame, min_confluence: int = 4, hold_offset: int = 2,
    neutral_zone: int = 2, ml_weight: float = 1, skip_contra: bool = False,
    min_conf: float = 0.40, atr_sl_mult: float = 1.5, atr_tp_mult: float = 3.0,
    fee_pct: float = 0.0006, leverage: int = 5, max_position_pct: float = 0.20,
    default_risk_pct: float = 0.015,
    sl_tp_mode: str = "atr", fixed_sl_pct: float = 0.015, fixed_tp_pct: float = 0.03,
    reverse: bool = False,
) -> dict:
    """
    Cheap decision + PnL simulation over a precomputed signal table.
    sl_tp_mode: "atr" (default, SL/TP = atr_sl_mult/atr_tp_mult * current ATR — adapts
    per-symbol/per-regime) or "fixed_pct" (SL/TP = fixed_sl_pct/fixed_tp_pct off entry
    price, same distance regardless of volatility — see project memory, 2026-07-23
    fixed-vs-ATR-stop discussion for why this is a proof-of-concept comparison mode,
    not a recommended default).
    reverse: flip every long<->short decision (SL/TP flipped to match). See
    ai/backtest.py::run_backtest's reverse docstring for the diagnostic rationale.
    """
    trades = []
    open_trade: Optional[dict] = None
    cash = 1000.0
    closed_pnls: list[float] = []
    equities = []
    _risk = RiskManager()

    for row in table.itertuples():
        price, atr = row.price, row.atr

        if open_trade:
            hit = None
            if open_trade["side"] == "long":
                if price <= open_trade["sl"]: hit = "sl"
                elif price >= open_trade["tp"]: hit = "tp"
            else:
                if price >= open_trade["sl"]: hit = "sl"
                elif price <= open_trade["tp"]: hit = "tp"
            if hit:
                exit_price = open_trade["sl"] if hit == "sl" else open_trade["tp"]
                entry, side = open_trade["entry"], open_trade["side"]
                amount, margin = open_trade["amount"], open_trade["margin"]
                pnl_gross = (exit_price - entry) * amount * (1 if side == "long" else -1)
                fee = (entry + exit_price) * amount * fee_pct
                net_pnl = pnl_gross - fee
                cash += net_pnl
                closed_pnls.append(net_pnl)
                trades.append({
                    "entry_ts": open_trade["entry_ts"], "exit_ts": row.ts,
                    "pnl_usdt": net_pnl,
                    "pnl_pct": (net_pnl / margin * 100) if margin else 0,
                    "exit_type": hit, "side": side, "ml_was_hold": open_trade["ml_was_hold"],
                })
                open_trade = None

        equities.append(cash)
        if open_trade or cash <= 0:
            continue

        ml_was_hold = (row.ml_label == "hold")
        ml_long, ml_short = _ml_points(row.ml_label, row.ml_conf, row.ml_agreement, min_conf, ml_weight)
        s_long = row.score_long + ml_long
        s_short = row.score_short + ml_short
        eff_min = min_confluence + (hold_offset if ml_was_hold else 0)
        best = max(s_long, s_short)

        if best < eff_min or abs(s_long - s_short) < neutral_zone:
            continue

        side = "long" if s_long >= s_short else "short"
        if skip_contra and not ml_was_hold:
            label_side = "long" if row.ml_label == "buy" else "short"
            if side != label_side:
                continue

        if reverse:
            side = "short" if side == "long" else "long"

        if sl_tp_mode == "fixed_pct":
            sl = price * (1 - fixed_sl_pct) if side == "long" else price * (1 + fixed_sl_pct)
            tp = price * (1 + fixed_tp_pct) if side == "long" else price * (1 - fixed_tp_pct)
        else:
            sl = price - atr * atr_sl_mult if side == "long" else price + atr * atr_sl_mult
            tp = price + atr * atr_tp_mult if side == "long" else price - atr * atr_tp_mult
        equity = cash
        risk_pct = (_risk.kelly_risk_pct(closed_pnls) or default_risk_pct) * row.vol_risk_mult
        risk_amount = equity * risk_pct
        sl_dist = max(abs(price - sl), price * 0.005)
        raw_amount = risk_amount / sl_dist
        cap_amount = (equity * max_position_pct * leverage) / price
        amount = min(raw_amount, cap_amount)
        margin = (amount * price) / leverage

        open_trade = {
            "side": side, "entry": price, "sl": sl, "tp": tp, "entry_ts": row.ts,
            "amount": amount, "margin": margin, "ml_was_hold": ml_was_hold,
        }

    # Close any remaining trade at the last bar's price (mirrors ai/backtest.py)
    if open_trade:
        last_price = float(table["price"].iloc[-1])
        entry, side = open_trade["entry"], open_trade["side"]
        amount, margin = open_trade["amount"], open_trade["margin"]
        pnl_gross = (last_price - entry) * amount * (1 if side == "long" else -1)
        fee = (entry + last_price) * amount * fee_pct
        net_pnl = pnl_gross - fee
        cash += net_pnl
        closed_pnls.append(net_pnl)
        trades.append({
            "entry_ts": open_trade["entry_ts"], "exit_ts": table["ts"].iloc[-1],
            "pnl_usdt": net_pnl,
            "pnl_pct": (net_pnl / margin * 100) if margin else 0,
            "exit_type": "end_of_test", "side": side, "ml_was_hold": open_trade["ml_was_hold"],
        })

    if not trades:
        return {"trades": 0, "error": "No trades"}

    pnls = [t["pnl_pct"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    peak, max_dd = equities[0], 0.0
    for eq in equities:
        if eq > peak: peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd: max_dd = dd
    pnl_arr = np.array(pnls)
    sharpe = 0.0
    if len(pnl_arr) > 1 and pnl_arr.std() > 0:
        sharpe = round(float(pnl_arr.mean() / pnl_arr.std() * np.sqrt(8760 / len(table) * len(trades))), 2)

    hold_trades = [t for t in trades if t["ml_was_hold"]]
    hold_wins = [t for t in hold_trades if t["pnl_pct"] > 0]
    hold_losses = [t for t in hold_trades if t["pnl_pct"] <= 0]
    hold_pf = (round(abs(sum(t["pnl_pct"] for t in hold_wins) / sum(t["pnl_pct"] for t in hold_losses)), 2)
               if hold_losses else (99.0 if hold_wins else None))

    return {
        "trades": len(trades),
        "win_rate": round(len(wins) / len(trades) * 100, 1),
        "profit_factor": round(abs(sum(wins) / sum(losses)), 2) if losses else 99.0,
        "total_return_pct": round((cash - 1000) / 1000 * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "sharpe": sharpe,
        "hold_trades": len(hold_trades),
        "hold_pf": hold_pf,
    }
