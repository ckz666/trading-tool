"""Portfolio-level risk allocator across all engines sharing the wallet
(2026-07-23, DeepSeek design, see project memory).

Before this: only AutoTrader used Kelly sizing, and only from its OWN trade
history — blind to what the other three engines were doing. The other three
(Funding Harvest, Mean Reversion, Pairs Trading) used fixed risk_pct/
max_position_pct constants. The cross-engine risk-CHECK fix earlier this
session (AutoTrader.portfolio_value_fn) made the drawdown/daily-loss
breakers see combined equity — this is the SIZING counterpart: an engine's
position size should also reflect how much of the shared portfolio's total
risk budget it currently deserves, not just its own isolated history.

Design:
1. Each engine's daily P&L series (resampled from its own equity_history)
   yields per-engine mean/variance (mu_i, sigma_i).
2. Cold start (< MIN_DAYS_FOR_CORRELATION days of daily returns): diagonal-
   only Kelly, f_i = 0.5 * mu_i / sigma_i^2 — no cross-engine correlation
   term, since there isn't remotely enough history yet to estimate one
   reliably (all four engines are only days old as of this fix). This is
   the mode that will actually be active for weeks — not a rarely-hit edge
   case.
3. Once enough history exists: full covariance via sklearn's LedoitWolf
   shrinkage estimator (shrinks the sample covariance toward a diagonal
   prior — standard, avoids hand-rolling the shrinkage formula), half-Kelly
   vector f = 0.5 * Sigma^-1 * mu.
4. Total portfolio risk budget F_TOTAL (fraction of combined equity) is
   split across engines proportional to positive f_i (a negative f_i means
   "Kelly says don't size this up," not "short it" — excluded, not
   negated), then clamped per-engine to [MIN_ENGINE_PCT, MAX_ENGINE_PCT].

Each engine keeps sizing its own positions exactly as before (SL-distance-
based risk_pct*equity, or Funding Harvest's notional-cap approach) — only
the risk_pct/max_position_pct INPUT now comes from here instead of being a
fixed constructor constant. AutoTrader is the one exception: it keeps its
own per-symbol Kelly (real, useful nuance the other engines don't have —
different symbols carry different realised edge) and uses this allocator's
output as a CEILING on top of it rather than a full replacement, so a
strong per-symbol Kelly result can't ignore the portfolio-level budget.
"""
from datetime import datetime

import numpy as np
import pandas as pd

F_TOTAL = 0.20          # total portfolio risk budget as a fraction of combined equity
MIN_ENGINE_PCT = 0.002  # 0.2% floor — never fully zero out a running strategy
MAX_ENGINE_PCT = 0.05   # 5% ceiling — no single engine can dominate the budget
MIN_DAYS_FOR_CORRELATION = 30
DEFAULT_PCT = 0.01      # fallback when an engine has no usable history at all yet
RECOMPUTE_INTERVAL_SECONDS = 3600   # hourly is plenty for daily-resampled inputs


def _daily_returns(equity_history: list[dict]) -> pd.Series:
    """Resample an engine's [{ts, equity}] equity_history into daily % returns."""
    if len(equity_history) < 2:
        return pd.Series(dtype=float)
    df = pd.DataFrame(equity_history)
    df["ts"] = pd.to_datetime(df["ts"])
    s = df.set_index("ts")["equity"].resample("1D").last().ffill()
    return s.pct_change().dropna()


class PortfolioRiskAllocator:
    def __init__(self):
        self._cache: dict[str, float] = {}
        self._last_computed: float = 0.0
        self._mode: str = "cold_start"
        self._n_days: int = 0

    def maybe_recompute(self, engine_equity_histories: dict[str, list[dict]]) -> dict[str, float]:
        """Only actually recomputes every RECOMPUTE_INTERVAL_SECONDS — cheap
        to call every cycle, serves the cache the rest of the time."""
        now = datetime.now().timestamp()
        if now - self._last_computed < RECOMPUTE_INTERVAL_SECONDS and self._cache:
            return self._cache
        return self.compute(engine_equity_histories)

    def compute(self, engine_equity_histories: dict[str, list[dict]]) -> dict[str, float]:
        returns = {name: _daily_returns(hist) for name, hist in engine_equity_histories.items()}
        names = list(returns.keys())

        combined = pd.DataFrame({n: returns[n] for n in names}).fillna(0.0)
        n_days = len(combined)
        self._n_days = n_days
        self._mode = "correlated" if n_days >= MIN_DAYS_FOR_CORRELATION else "cold_start"

        if n_days < MIN_DAYS_FOR_CORRELATION:
            f = {}
            for n in names:
                r = returns[n]
                if len(r) < 3:
                    f[n] = 0.0
                    continue
                mu, var = float(r.mean()), float(r.var())
                # Epsilon floor, not a hard var>0 gate: a real (if short) run
                # of near-zero-variance positive returns is a GOOD sign, not
                # a "no signal" case — dividing by a tiny epsilon instead of
                # bailing to 0.0 lets it register as a strong positive f_i,
                # which then gets reined in by the MAX_ENGINE_PCT clamp below
                # like any other large f_i, rather than being silently
                # dropped to the floor for the wrong reason.
                f[n] = 0.5 * mu / max(var, 1e-6)
        else:
            from sklearn.covariance import LedoitWolf
            mu = combined.mean().values
            sigma = LedoitWolf().fit(combined.values).covariance_
            try:
                f_vec = 0.5 * np.linalg.solve(sigma, mu)
            except np.linalg.LinAlgError:
                f_vec = 0.5 * mu / np.diag(sigma)
            f = dict(zip(names, f_vec))

        positive = {n: v for n, v in f.items() if v > 0}
        total_f = sum(positive.values())

        result = {}
        for n in names:
            if n not in positive or total_f <= 0:
                # Too little history (< 3 daily return points) OR a non-
                # positive Kelly read (no edge, or negative) both land here —
                # DeepSeek's design explicitly wants the CONSERVATIVE floor
                # in both cases, not a more generous "benefit of the doubt"
                # default. An engine only earns a bigger budget by showing
                # positive risk-adjusted expectancy, never by having no data.
                result[n] = MIN_ENGINE_PCT
                continue
            b = F_TOTAL * positive[n] / total_f
            result[n] = max(MIN_ENGINE_PCT, min(b, MAX_ENGINE_PCT))

        self._cache = result
        self._last_computed = datetime.now().timestamp()
        return result

    def get_risk_pct(self, engine_name: str, default: float = DEFAULT_PCT) -> float:
        return self._cache.get(engine_name, default)

    def status(self) -> dict:
        return {
            "budget_pct": dict(self._cache),
            "f_total": F_TOTAL,
            "last_computed": datetime.fromtimestamp(self._last_computed).isoformat() if self._last_computed else None,
            "mode": self._mode,
            "days_of_history": self._n_days,
            "days_needed_for_correlation": MIN_DAYS_FOR_CORRELATION,
        }


_allocator: PortfolioRiskAllocator = None


def get_allocator() -> PortfolioRiskAllocator:
    global _allocator
    if _allocator is None:
        _allocator = PortfolioRiskAllocator()
    return _allocator
