"""
MTF Backtest Engine
Walk-forward simulation: train on first 70% of 1H data, test on remaining 30%.
For each test bar: ML signal + MTF confluence → simulate trade with ATR-based SL/TP
and Kelly-scaled ATR position sizing (mirrors trading/autotrader.py).
Claude is not included (too expensive) — uses rule: conf >= threshold AND confluence >= min_conf.
"""

import numpy as np
import pandas as pd
from typing import Optional

from ai.ml_signal import (
    build_features, make_labels, train as ml_train,
    get_indicators, detect_market_structure, _funding_to_series,
)
from ai.patterns import detect_patterns
from trading.risk import RiskManager


# ── helpers ──────────────────────────────────────────────────────────────────

def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate 1H OHLCV to 4H or 1D without lookahead."""
    r = df.resample(rule, closed="left", label="left")
    out = pd.DataFrame({
        "open":   r["open"].first(),
        "high":   r["high"].max(),
        "low":    r["low"].min(),
        "close":  r["close"].last(),
        "volume": r["volume"].sum(),
    }).dropna()
    return out


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl  = df["high"] - df["low"]
    hpc = (df["high"] - df["close"].shift(1)).abs()
    lpc = (df["low"]  - df["close"].shift(1)).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _confluence_score_bt(ml_label: str, ml_conf: float, ind_1h: dict,
                          ind_4h: dict = None, ind_1d: dict = None,
                          patterns: dict = None) -> int:
    """Lightweight confluence for backtesting (no live sentiment/OI/CVD — max ≈20 of 24)."""
    if ml_label == "hold":
        return 0
    is_long = ml_label == "buy"
    score = 0

    if ml_conf >= 0.72: score += 2
    elif ml_conf >= 0.65: score += 1

    rsi = ind_1h.get("rsi", 50)
    if is_long and rsi < 42: score += 1
    elif not is_long and rsi > 58: score += 1

    if is_long and ind_1h.get("macd_diff", 0) > 0: score += 1
    elif not is_long and ind_1h.get("macd_diff", 0) < 0: score += 1

    if is_long and ind_1h.get("ema_cross_norm", 0) > 0: score += 1
    elif not is_long and ind_1h.get("ema_cross_norm", 0) < 0: score += 1

    if is_long and ind_1h.get("vwap_dist", 0) > 0.001: score += 1
    elif not is_long and ind_1h.get("vwap_dist", 0) < -0.001: score += 1

    if ind_4h:
        ms = ind_4h.get("market_structure", "unknown")
        if is_long and ms == "uptrend": score += 2
        elif not is_long and ms == "downtrend": score += 2
        elif ms == "sideways": score -= 1

    if ind_1d:
        d_ema = ind_1d.get("ema_cross_norm", 0)
        d_rsi = ind_1d.get("rsi", 50)
        d_bull = d_ema > 0 and d_rsi > 50
        d_bear = d_ema < 0 and d_rsi < 50
        if is_long:
            if d_bull: score += 2
            elif d_bear: score -= 1
        else:
            if d_bear: score += 2
            elif d_bull: score -= 1

    if patterns:
        for _, pinfo in patterns.items():
            pt = pinfo if isinstance(pinfo, str) else pinfo.get("type", "")
            if is_long and pt == "bullish": score += 1; break
            elif not is_long and pt == "bearish": score += 1; break

    ichi = ind_1h.get("ichimoku") or {}
    if ichi.get("available"):
        pos_ = ichi["price_vs_cloud"]
        tk   = ichi["tk_cross"]
        if is_long:
            if pos_ == "above" and tk == "bullish": score += 2
            elif pos_ == "above": score += 1
            elif pos_ == "below": score -= 1
        else:
            if pos_ == "below" and tk == "bearish": score += 2
            elif pos_ == "below": score += 1
            elif pos_ == "above": score -= 1

    return max(score, 0)


# ── main backtest ─────────────────────────────────────────────────────────────

def run_backtest(
    df_1h: pd.DataFrame,
    symbol: str = "BTC/USDT",
    funding_series: pd.Series = None,
    train_pct: float = 0.70,
    min_confluence: int = 4,
    min_conf: float = 0.50,
    atr_sl_mult: float = 1.5,     # SL = entry ± ATR * mult
    atr_tp_mult: float = 3.0,     # TP = entry ± ATR * mult  (R:R = 2)
    fee_pct: float = 0.0006,      # 0.06% taker fee per side
    leverage: int = 5,
    max_position_pct: float = 0.20,   # cap: max margin per trade as % of equity (mirrors RiskConfig)
    default_risk_pct: float = 0.015,  # fallback risk-per-trade until Kelly has enough history
    progress_cb=None,             # optional callable(str) for live progress log
) -> dict:
    """
    Walk-forward MTF backtest.
    Returns metrics dict + trade_log list.
    """
    def _progress(msg: str):
        if progress_cb:
            try:
                progress_cb(msg)
            except Exception:
                pass

    if len(df_1h) < 200:
        return {"error": "Need at least 200 candles"}

    split = int(len(df_1h) * train_pct)
    df_train = df_1h.iloc[:split].copy()
    df_test  = df_1h.iloc[split:].copy()

    # ── Train model on first 70% ──────────────────────────────────────────────
    _progress(f"Training ML-Modell auf {split} Kerzen ({symbol})…")
    fs_train = None
    if funding_series is not None and not funding_series.empty:
        fs_train = funding_series[funding_series.index <= df_train.index[-1]]
    ml_train(df_train, symbol, funding_series=fs_train)
    _progress(f"Modell trainiert — starte Walk-Forward über {len(df_test)} Test-Kerzen…")

    from ai.ml_signal import predict, _paths
    import pickle, os
    model_paths, scaler_path = _paths(symbol)
    if not all(os.path.exists(p) for p in model_paths.values()):
        return {"error": "Model training failed"}

    # Pre-compute ATR for test set (using full df so warm-up is correct)
    atr_full = _atr(df_1h)

    # ── Walk-forward through test bars ───────────────────────────────────────
    trades = []
    open_trade: Optional[dict] = None
    equity_curve = []
    cash = 1000.0   # start with $1000 paper capital
    closed_pnls: list[float] = []   # realised USDT PnL history, feeds Kelly sizing
    _risk = RiskManager()

    # Need enough lookback for indicators: use rolling window
    lookback = 300   # bars of history for indicator computation
    progress_step = max(1, len(df_test) // 20)   # ~20 progress updates total

    for i in range(len(df_test)):
        if cash <= 0:
            _progress(f"Bankrupt — Equity ${cash:,.2f} <= 0, breche Backtest ab")
            break
        if i > 0 and i % progress_step == 0:
            pct = int(i / len(df_test) * 100)
            _progress(f"Bar {i}/{len(df_test)} ({pct}%) — {len(trades)} Trades, Equity ${cash:,.2f}")
        abs_i = split + i
        bar   = df_test.iloc[i]
        ts    = df_test.index[i]
        price = float(bar["close"])
        atr   = float(atr_full.iloc[abs_i]) if abs_i < len(atr_full) else price * 0.01

        # ── Check open trade SL/TP ────────────────────────────────────────────
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
                entry  = open_trade["entry"]
                side   = open_trade["side"]
                amount = open_trade["amount"]
                margin = open_trade["margin"]
                pnl_gross = (exit_price - entry) * amount * (1 if side == "long" else -1)
                fee       = (entry + exit_price) * amount * fee_pct   # entry + exit fee
                net_pnl   = pnl_gross - fee
                cash += net_pnl
                closed_pnls.append(net_pnl)
                roe_pct = (net_pnl / margin * 100) if margin else 0
                trades.append({
                    "entry_ts":  open_trade["entry_ts"],
                    "exit_ts":   ts,
                    "symbol":    symbol,
                    "side":      side,
                    "entry":     round(entry, 4),
                    "exit":      round(exit_price, 4),
                    "exit_type": hit,
                    "confluence":open_trade["confluence"],
                    "ml_conf":   open_trade["ml_conf"],
                    "risk_pct":  open_trade["risk_pct"],
                    "pnl_usdt":  round(net_pnl, 2),
                    "pnl_pct":   round(roe_pct, 3),
                    "cash_after":round(cash, 2),
                })
                _progress(f"{ts} {hit.upper()} {side.upper()} @ {exit_price:.4f} | PnL ${net_pnl:+.2f} ({roe_pct:+.1f}% ROE) | Equity ${cash:,.2f}")
                open_trade = None

        equity_curve.append({"ts": str(ts), "equity": round(cash, 2)})

        if open_trade:
            continue   # only one position at a time

        # ── Compute indicators on rolling window ──────────────────────────────
        start = max(0, abs_i - lookback)
        window_1h = df_1h.iloc[start:abs_i + 1]
        if len(window_1h) < 50:
            continue

        fs_window = None
        if funding_series is not None and not funding_series.empty:
            fs_window = funding_series[funding_series.index <= ts]

        # ML signal on current window
        try:
            sig = predict(window_1h, symbol, funding_series=fs_window)
        except Exception:
            continue

        ml_label = sig.get("label", "hold")
        ml_conf  = sig.get("confidence", 0.0)

        if ml_label == "hold" or ml_conf < min_conf:
            continue

        # MTF: aggregate 1H → 4H and 1D for confluence
        ind_1h = get_indicators(window_1h)
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

        confluence = _confluence_score_bt(ml_label, ml_conf, ind_1h, ind_4h, ind_1d, pats)

        if confluence < min_confluence:
            continue

        # ── Open trade (Kelly-scaled ATR position sizing, mirrors AutoTrader) ────
        side  = "long" if ml_label == "buy" else "short"
        sl    = price - atr * atr_sl_mult if side == "long" else price + atr * atr_sl_mult
        tp    = price + atr * atr_tp_mult if side == "long" else price - atr * atr_tp_mult

        equity      = cash
        risk_pct    = _risk.kelly_risk_pct(closed_pnls) or default_risk_pct
        risk_amount = equity * risk_pct
        sl_dist     = max(abs(price - sl), price * 0.005)
        raw_amount  = risk_amount / sl_dist
        max_margin  = equity * max_position_pct
        cap_amount  = (max_margin * leverage) / price
        amount      = min(raw_amount, cap_amount)
        margin      = (amount * price) / leverage

        open_trade = {
            "side": side, "entry": price, "sl": sl, "tp": tp,
            "entry_ts": ts, "confluence": confluence, "ml_conf": round(ml_conf, 3),
            "amount": amount, "margin": margin, "risk_pct": round(risk_pct, 4),
        }
        _progress(f"{ts} OPEN {side.upper()} @ {price:.4f} | C={confluence} conf={ml_conf:.2f} | "
                  f"Risk {risk_pct:.2%} (${risk_amount:.0f}) | SL={sl:.4f} TP={tp:.4f}")

    # Close any remaining trade at last bar price
    if open_trade:
        price  = float(df_test.iloc[-1]["close"])
        entry  = open_trade["entry"]
        side   = open_trade["side"]
        amount = open_trade["amount"]
        margin = open_trade["margin"]
        pnl_gross = (price - entry) * amount * (1 if side == "long" else -1)
        fee       = (entry + price) * amount * fee_pct
        net_pnl   = pnl_gross - fee
        cash += net_pnl
        closed_pnls.append(net_pnl)
        roe_pct = (net_pnl / margin * 100) if margin else 0
        trades.append({
            "entry_ts":   open_trade["entry_ts"],
            "exit_ts":    df_test.index[-1],
            "symbol":     symbol,
            "side":       side,
            "entry":      round(entry, 4),
            "exit":       round(price, 4),
            "exit_type":  "end_of_test",
            "confluence": open_trade["confluence"],
            "ml_conf":    open_trade["ml_conf"],
            "risk_pct":   open_trade["risk_pct"],
            "pnl_usdt":   round(net_pnl, 2),
            "pnl_pct":    round(roe_pct, 3),
            "cash_after": round(cash, 2),
        })

    # ── Metrics ───────────────────────────────────────────────────────────────
    if not trades:
        _progress("Backtest fertig — keine Trades ausgelöst")
        return {
            "symbol": symbol, "trades": 0, "error": "No trades triggered",
            "train_bars": split, "test_bars": len(df_test),
            "min_confluence": min_confluence, "min_conf": min_conf,
        }

    pnls = [t["pnl_pct"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    # Max drawdown from equity curve
    equities = [e["equity"] for e in equity_curve]
    peak = equities[0]
    max_dd = 0.0
    for eq in equities:
        if eq > peak: peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd: max_dd = dd

    # Sharpe (annualised, assuming 1H bars, 8760 bars/year)
    pnl_arr = np.array(pnls)
    sharpe = 0.0
    if len(pnl_arr) > 1 and pnl_arr.std() > 0:
        sharpe = round(float(pnl_arr.mean() / pnl_arr.std() * np.sqrt(8760 / len(df_test) * len(trades))), 2)

    sl_exits  = sum(1 for t in trades if t["exit_type"] == "sl")
    tp_exits  = sum(1 for t in trades if t["exit_type"] == "tp")
    longs  = sum(1 for t in trades if t["side"] == "long")
    shorts = sum(1 for t in trades if t["side"] == "short")

    total_return_pct = round((cash - 1000) / 1000 * 100, 2)
    win_rate = round(len(wins) / len(trades) * 100, 1)
    _progress(f"Backtest fertig — {len(trades)} Trades | WR {win_rate}% | Return {total_return_pct:+.2f}% | Sharpe {sharpe}")

    return {
        "symbol":          symbol,
        "train_bars":      split,
        "test_bars":       len(df_test),
        "test_period":     f"{df_test.index[0]} → {df_test.index[-1]}",
        "params": {
            "min_confluence": min_confluence,
            "min_conf":       min_conf,
            "atr_sl_mult":    atr_sl_mult,
            "atr_tp_mult":    atr_tp_mult,
            "leverage":       leverage,
        },
        "trades":          len(trades),
        "longs":           longs,
        "shorts":          shorts,
        "sl_exits":        sl_exits,
        "tp_exits":        tp_exits,
        "win_rate":        win_rate,
        "avg_win_pct":     round(float(np.mean(wins)), 3) if wins else 0,
        "avg_loss_pct":    round(float(np.mean(losses)), 3) if losses else 0,
        "profit_factor":   round(abs(sum(wins) / sum(losses)), 2) if losses else 99.0,
        "total_return_pct":total_return_pct,
        "max_drawdown_pct":round(max_dd * 100, 2),
        "sharpe":          sharpe,
        "trade_log":       trades,
        "equity_curve":    equity_curve[-200:],  # last 200 points for chart
    }
