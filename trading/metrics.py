"""Sharpe ratio / max drawdown from an engine's persisted equity_history
([{ts, equity}, ...], see FuturesPaperEngine.record_equity). Pure read-only
math over already-recorded data — no new sampling loop needed."""
import math
from datetime import datetime


def compute_metrics(equity_history: list[dict]) -> dict:
    """annualization assumes the ~5min poll cadence these engines record at
    (matches web/app.py's price-poll loop, see record_equity() call sites) —
    periods_per_year is an approximation, not exact, since the interval
    between snapshots isn't perfectly constant (skipped cycles, restarts)."""
    if len(equity_history) < 2:
        return {"sharpe": None, "max_drawdown_pct": None, "return_pct": None, "sample_size": len(equity_history)}

    equities = [p["equity"] for p in equity_history]
    returns = []
    for i in range(1, len(equities)):
        prev = equities[i - 1]
        if prev > 0:
            returns.append((equities[i] - prev) / prev)

    sharpe = None
    if len(returns) >= 10:
        mean_r = sum(returns) / len(returns)
        variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
        std_r = math.sqrt(variance)
        if std_r > 0:
            periods_per_year = _estimate_periods_per_year(equity_history)
            sharpe = (mean_r / std_r) * math.sqrt(periods_per_year)

    peak = equities[0]
    max_dd = 0.0
    for e in equities:
        peak = max(peak, e)
        if peak > 0:
            dd = (peak - e) / peak
            max_dd = max(max_dd, dd)

    return_pct = (equities[-1] - equities[0]) / equities[0] * 100 if equities[0] > 0 else None

    return {
        "sharpe": round(sharpe, 3) if sharpe is not None else None,
        "max_drawdown_pct": round(max_dd * 100, 2),
        "return_pct": round(return_pct, 2) if return_pct is not None else None,
        "sample_size": len(equity_history),
    }


def _estimate_periods_per_year(equity_history: list[dict]) -> float:
    try:
        t0 = datetime.fromisoformat(equity_history[0]["ts"])
        t1 = datetime.fromisoformat(equity_history[-1]["ts"])
        span_seconds = (t1 - t0).total_seconds()
        n = len(equity_history) - 1
        if span_seconds <= 0 or n <= 0:
            return 105120  # fallback: 5-min cadence assumption
        avg_interval_s = span_seconds / n
        return (365 * 24 * 3600) / avg_interval_s
    except Exception:
        return 105120
