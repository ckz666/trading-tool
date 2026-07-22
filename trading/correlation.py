"""
Cross-strategy correlation report for AutoTrader / Grid / Funding Harvest.

They already run in parallel on the shared wallet, which only pays off as
real diversification (σ_p = σ/√N) if their PnL streams are actually
uncorrelated. This measures it directly from trade history instead of
assuming it, and reports the realised portfolio-level Sharpe using the
standard equal-weight formula for N assets at average pairwise correlation
ρ̄ (reduces to σ/√N only in the idealised ρ̄ = 0 case):

    σ_p = σ̄ · sqrt( 1/N + (N-1)/N · ρ̄ )

That average-correlation formula is exact only when every pair shares the
same ρ̄ (an "equicorrelated" matrix). Real correlation matrices usually
aren't — e.g. two strategies could be tightly coupled while a third floats
free, which a single averaged number can't distinguish from three strategies
each moderately correlated with each other. Two additions cover that:
1. The exact equal-weight portfolio variance ratio wᵀCw (no equicorrelation
   assumption at all — just the real quadratic form).
2. The correlation matrix's top eigenvalue / N, i.e. how much of the total
   variance one dominant common factor explains. High eigenvalue
   concentration with a merely middling average correlation is the signature
   of exactly that asymmetric case — worth flagging even at N=3, and it's
   what actually generalises if a 4th strategy gets added later.
"""

from datetime import datetime
import numpy as np
import pandas as pd

# (engine name, trade_history action-prefix that marks a realised close, pnl field name)
_ENGINE_SPEC = {
    "autotrader":     ("close_", "pnl"),
    "grid":           ("close_grid", "realized_pnl"),
    "funding_harvest":("close", "net_pnl"),
}


def _daily_pnl(trade_history: list[dict], action_prefix: str, pnl_field: str) -> pd.Series:
    rows = []
    for r in trade_history:
        action = r.get("action", "")
        if not action.startswith(action_prefix):
            continue
        pnl = r.get(pnl_field)
        ts = r.get("ts")
        if pnl is None or not ts:
            continue
        rows.append((pd.to_datetime(ts).date(), pnl))
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows, columns=["date", "pnl"])
    return df.groupby("date")["pnl"].sum()


def _sharpe(daily_pnl: pd.Series, capital: float) -> float:
    if len(daily_pnl) < 5 or capital <= 0:
        return 0.0
    returns = daily_pnl / capital
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(365))


def strategy_correlation_report(
    autotrader_history: list[dict],
    grid_history: list[dict],
    funding_history: list[dict],
    capital_per_strategy: float = 3333.0,   # ≈ equal thirds of the shared 10k wallet
) -> dict:
    """
    Returns pairwise correlation between the three strategies' daily realised
    PnL, each strategy's own annualised Sharpe, and the portfolio-level
    Sharpe implied by their measured (not assumed) correlation.

    Returns None if there isn't enough overlapping daily history yet across
    all three strategies to compute a meaningful correlation.
    """
    series = {
        "autotrader":      _daily_pnl(autotrader_history, *_ENGINE_SPEC["autotrader"]),
        "grid":             _daily_pnl(grid_history, *_ENGINE_SPEC["grid"]),
        "funding_harvest":  _daily_pnl(funding_history, *_ENGINE_SPEC["funding_harvest"]),
    }

    sharpes = {name: round(_sharpe(s, capital_per_strategy), 2) for name, s in series.items()}

    df = pd.DataFrame(series).fillna(0.0)
    if len(df) < 10 or (df != 0).sum().min() < 5:
        return {
            "status": "insufficient_history",
            "days_available": len(df),
            "per_strategy_sharpe": sharpes,
        }

    corr = df.corr().round(3)
    pairs = [("autotrader", "grid"), ("autotrader", "funding_harvest"), ("grid", "funding_harvest")]
    pairwise = {f"{a}_vs_{b}": float(corr.loc[a, b]) for a, b in pairs}
    avg_corr = float(np.mean(list(pairwise.values())))

    active = [name for name, s in series.items() if (s != 0).sum() >= 5]
    n = len(active)
    avg_sharpe = round(float(np.mean([sharpes[name] for name in active])), 2) if active else 0.0

    # Exact equal-weight portfolio variance ratio wᵀCw (w = 1/n each) — no
    # equicorrelation assumption, just the real quadratic form on the
    # measured correlation matrix restricted to the active strategies.
    C = corr.loc[active, active].to_numpy()
    w = np.full(n, 1.0 / n) if n > 0 else np.array([])
    variance_ratio = float(w @ C @ w) if n > 0 else 1.0
    portfolio_scale = variance_ratio ** 0.5

    # Eigenvalue concentration: how much of the total variance one dominant
    # common factor explains, independent of the equal-weighting assumption above.
    eigenvalues = np.linalg.eigvalsh(C) if n > 0 else np.array([])
    top_eigenvalue = float(eigenvalues.max()) if len(eigenvalues) else 0.0
    top_factor_variance_pct = round(top_eigenvalue / n * 100, 1) if n > 0 else 0.0

    if n >= 2:
        portfolio_sharpe_est = round(avg_sharpe / portfolio_scale, 2) if portfolio_scale > 0 else avg_sharpe
        zero_corr_sharpe_est = round(avg_sharpe * (n ** 0.5), 2)
    else:
        portfolio_sharpe_est = avg_sharpe
        zero_corr_sharpe_est = avg_sharpe

    return {
        "status": "ok",
        "days_available": len(df),
        "per_strategy_sharpe": sharpes,
        "pairwise_correlation": pairwise,
        "avg_correlation": round(avg_corr, 3),
        "n_active_strategies": n,
        "portfolio_variance_ratio": round(variance_ratio, 3),
        "top_factor_variance_pct": top_factor_variance_pct,
        "portfolio_sharpe_estimate": portfolio_sharpe_est,
        "portfolio_sharpe_if_uncorrelated": zero_corr_sharpe_est,
        "diversification_note": (
            f"At the measured avg correlation of {avg_corr:.2f}, running {n} strategies "
            f"together gets you ~{portfolio_sharpe_est} portfolio Sharpe vs {zero_corr_sharpe_est} "
            f"in the idealised zero-correlation case, and {avg_sharpe} for a single strategy alone. "
            f"Equal-weight portfolio volatility is {portfolio_scale*100:.1f}% of a single strategy's; "
            f"one dominant common factor explains {top_factor_variance_pct}% of the variance."
        ),
    }
