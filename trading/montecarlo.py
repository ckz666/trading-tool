"""
Monte-Carlo trade-sequence resampling.

Takes the trade_log from a completed backtest and reshuffles the order of the
*same* realised trades many times, replaying each shuffled sequence against
the live RiskManager's drawdown-breaker threshold. A backtest only shows you
one path through the trades it happened to generate in chronological order —
this shows the distribution of outcomes across all the orders those same
trades could plausibly have landed in, which is what actually determines
whether the drawdown-breaker would have paused entries along the way.
"""

import random
from trading.risk import RiskConfig


def resample_drawdown(
    trade_log: list[dict],
    config: RiskConfig = None,
    n_sims: int = 2000,
    start_equity: float = 1000.0,
    seed: int = None,
) -> dict:
    """
    Shuffle-resample a completed backtest's trade sequence to estimate how
    often the drawdown-breaker would have paused new entries, independent of
    the specific chronological order the trades happened to occur in.

    Each simulated path replays all trades from trade_log in a random order,
    starting from start_equity, applying pnl_usdt sequentially. Per path we
    track: max drawdown reached, and whether it ever breached
    config.max_drawdown_pct (i.e. would have triggered RiskManager.check_drawdown
    to block new entries — existing positions still manage their own SL/TP,
    so this models a pause, not a wipeout).

    Returns None if there are too few trades to make resampling meaningful.
    """
    config = config or RiskConfig()
    pnls = [t["pnl_usdt"] for t in trade_log if "pnl_usdt" in t]
    n_trades = len(pnls)
    if n_trades < 10:
        return None

    rng = random.Random(seed)
    breaker_pct = config.max_drawdown_pct

    max_dds: list[float] = []
    final_equities: list[float] = []
    breached_count = 0

    for _ in range(n_sims):
        order = pnls[:]
        rng.shuffle(order)

        equity = start_equity
        peak = start_equity
        path_max_dd = 0.0
        breached = False

        for pnl in order:
            equity += pnl
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0.0
            if dd > path_max_dd:
                path_max_dd = dd
            if dd >= breaker_pct:
                breached = True

        max_dds.append(path_max_dd)
        final_equities.append(equity)
        if breached:
            breached_count += 1

    # Original chronological order, for comparison against the shuffled distribution
    orig_equity = start_equity
    orig_peak = start_equity
    orig_max_dd = 0.0
    orig_breached = False
    for pnl in pnls:
        orig_equity += pnl
        if orig_equity > orig_peak:
            orig_peak = orig_equity
        dd = (orig_peak - orig_equity) / orig_peak if orig_peak > 0 else 0.0
        if dd > orig_max_dd:
            orig_max_dd = dd
        if dd >= breaker_pct:
            orig_breached = True

    def _pct(arr: list[float], p: float) -> float:
        s = sorted(arr)
        idx = min(len(s) - 1, max(0, int(round(p / 100 * (len(s) - 1)))))
        return s[idx]

    return {
        "n_sims": n_sims,
        "n_trades": n_trades,
        "breaker_threshold_pct": round(breaker_pct * 100, 1),
        "breach_rate_pct": round(breached_count / n_sims * 100, 1),
        "max_drawdown_pct": {
            "p5":  round(_pct(max_dds, 5) * 100, 2),
            "p50": round(_pct(max_dds, 50) * 100, 2),
            "p95": round(_pct(max_dds, 95) * 100, 2),
            "worst": round(max(max_dds) * 100, 2),
        },
        "final_equity": {
            "p5":  round(_pct(final_equities, 5), 2),
            "p50": round(_pct(final_equities, 50), 2),
            "p95": round(_pct(final_equities, 95), 2),
        },
        "chronological_order": {
            "max_drawdown_pct": round(orig_max_dd * 100, 2),
            "breached_breaker": orig_breached,
            "final_equity": round(orig_equity, 2),
        },
    }
