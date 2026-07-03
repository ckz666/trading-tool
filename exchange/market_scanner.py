"""
Fetches top trending USDT-perpetual futures directly from Bitget public API.
Only returns pairs that are actually tradeable on Bitget.
"""
import aiohttp
import math
from typing import Optional

BITGET_TICKERS_URL = "https://api.bitget.com/api/v2/mix/market/tickers?productType=USDT-FUTURES"

_EXCLUDE = {
    "USDC", "BUSD", "TUSD", "USDP", "DAI", "FDUSD",         # stablecoins
    "PEPE", "SHIB", "FLOKI", "LUNC", "LUNA",                  # meme / dead
    # Tokenised stocks / ETFs / commodities — not standard crypto
    "SOXL", "SOXS", "NVDA", "TSLA", "AAPL", "AMZN", "GOOGL",
    "MSTR", "MRVL", "MU", "SNDK", "SPCX", "SKHYNIX", "DRAM",
    "PLTR", "COIN", "HOOD", "MARA", "RIOT", "HUT",
    "RKLB", "IONQ", "QUBT", "RGTI", "ACHR", "JOBY",           # more tokenised stocks
    "XAU", "XAG", "OIL", "BRN", "WTI",                        # commodities
}

MIN_VOLUME_USDT = 20_000_000   # $20M 24h — lower catches more small/new-coin momentum,
                                # still filters out the thinnest (sub-$10M) pairs


def _parse_tickers(raw: list) -> list[dict]:
    """Parse Bitget ticker list into normalised dicts."""
    out = []
    for t in raw:
        sym_raw = t.get("symbol", "")
        if not sym_raw.endswith("USDT"):
            continue
        base = sym_raw[:-4]
        if base in _EXCLUDE:
            continue
        try:
            vol   = float(t["usdtVolume"])
            chg   = float(t["change24h"]) * 100   # decimal → percent
            price = float(t["lastPr"])
        except (KeyError, ValueError, TypeError):
            continue
        if vol < MIN_VOLUME_USDT or price <= 0:
            continue
        out.append({
            "symbol":      base + "/USDT",
            "change_pct":  round(chg, 2),
            "volume_usdt": round(vol / 1e6, 1),   # millions
            "price":       price,
        })
    return out


async def _fetch_bitget_tickers() -> list[dict]:
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        ) as s:
            async with s.get(BITGET_TICKERS_URL) as r:
                data = await r.json()
        return _parse_tickers(data.get("data", []))
    except Exception as e:
        print(f"[MarketScanner] Bitget fetch error: {e}")
        return []


async def get_trending_symbols(
    top_n: int = 5,
    min_volume: float = MIN_VOLUME_USDT,
) -> list[dict]:
    """
    Top top_n Bitget USDT-perp pairs ranked by momentum score.
    score = |change_pct| * log10(volume_usdt_raw)
    Each entry: {symbol, change_pct, volume_usdt, price, score}
    """
    tickers = await _fetch_bitget_tickers()
    # re-filter with caller's min_volume (in raw USDT)
    tickers = [t for t in tickers if t["volume_usdt"] * 1e6 >= min_volume]

    for t in tickers:
        vol_raw = t["volume_usdt"] * 1e6
        t["score"] = round(abs(t["change_pct"]) * math.log10(max(vol_raw, 1)), 2)

    tickers.sort(key=lambda x: x["score"], reverse=True)
    return tickers[:top_n]


async def get_all_market_overview(
    top_n: int = 20,
    min_volume: float = MIN_VOLUME_USDT,
) -> list[dict]:
    """Top gainers + losers for UI display."""
    tickers = await _fetch_bitget_tickers()
    tickers = [t for t in tickers if t["volume_usdt"] * 1e6 >= min_volume]
    tickers.sort(key=lambda x: x["change_pct"], reverse=True)
    half = top_n // 2
    return tickers[:half] + tickers[-half:]


async def get_bitget_symbols() -> set[str]:
    """Return set of all tradeable BTC/USDT-style symbols on Bitget USDT-perps."""
    tickers = await _fetch_bitget_tickers()
    return {t["symbol"] for t in tickers}
