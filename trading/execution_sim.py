"""Shared execution-realism helper (2026-07-23, see project memory) — the
paper engines used to fill every order instantly and completely at the last
ticker price, which quietly makes backtests/live paper results look better
than real execution would. Wraps FuturesClient/BitgetClient.estimate_execution()
(live orderbook walk) into one call each engine's open/close call sites can
use without duplicating the fetch-and-fallback logic.
"""
from typing import Optional


def walk_orderbook(levels: list, notional_usdt: float) -> Optional[dict]:
    """Shared math behind FuturesClient/BitgetClient.estimate_execution() —
    both wrap this after fetching their own orderbook, so the walk-the-book
    logic isn't duplicated between the perp and spot clients. levels: ccxt's
    standard [[price, qty], ...] format (already the correct side — asks for
    a buy, bids for a sell). Returns None if levels is empty/invalid."""
    if not levels:
        return None
    best_price = levels[0][0]
    if not best_price or best_price <= 0:
        return None

    remaining_notional = notional_usdt
    total_qty = 0.0
    total_cost = 0.0
    for price, qty in levels:
        if not price or not qty:
            continue
        level_notional = price * qty
        take_notional = min(remaining_notional, level_notional)
        take_qty = take_notional / price
        total_qty += take_qty
        total_cost += take_qty * price
        remaining_notional -= take_notional
        if remaining_notional <= 1e-9:
            break

    filled_pct = max(0.0, min(1.0, 1.0 - (remaining_notional / notional_usdt))) if notional_usdt > 0 else 0.0
    avg_price = (total_cost / total_qty) if total_qty > 0 else best_price
    slippage_pct = abs(avg_price - best_price) / best_price
    return {
        "avg_price": avg_price,
        "slippage_pct": slippage_pct,
        "filled_pct": filled_pct,
        "best_price": best_price,
    }


async def simulate_fill(client, symbol: str, side: str, price: float, amount: float) -> tuple[float, float, dict]:
    """side: 'buy' (long entry / short exit) or 'sell' (short entry / long exit).
    price/amount: the naive intended fill (last ticker price, full size).

    Returns (fill_price, fill_amount, info) — fill_price/fill_amount are what
    the caller should actually use for sizing/PnL. info always has
    'slippage_pct'/'filled_pct'/'simulated' keys so callers can log/expose it
    regardless of whether the live estimate succeeded.

    Falls back to the naive (price, amount) — filled_pct=1.0, no slippage —
    if the orderbook fetch fails (network hiccup, thin/delisted symbol with
    an empty book) or the notional is effectively zero. This is a
    best-effort realism layer, not a hard dependency: a fetch failure here
    must never block a trade the way a stale price feed would."""
    notional = price * amount
    if notional <= 0:
        return price, amount, {"slippage_pct": 0.0, "filled_pct": 1.0, "simulated": False}
    try:
        est = await client.estimate_execution(symbol, side, notional)
    except Exception:
        est = None
    if not est or not est.get("avg_price"):
        return price, amount, {"slippage_pct": 0.0, "filled_pct": 1.0, "simulated": False}
    fill_price = est["avg_price"]
    fill_amount = amount * est["filled_pct"]
    return fill_price, fill_amount, {
        "slippage_pct": est["slippage_pct"],
        "filled_pct": est["filled_pct"],
        "simulated": True,
    }
