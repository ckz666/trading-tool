"""
Binance Futures public OHLCV client — no API key required.
Used only for backtesting (more history than Bitget).
Live trading stays on Bitget.
"""
import asyncio
import aiohttp

_BASE = "https://fapi.binance.com/fapi/v1/klines"
_FUNDING_BASE = "https://fapi.binance.com/fapi/v1/fundingRate"
_INTERVAL_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w",
}


def _to_binance_symbol(symbol: str) -> str:
    """'BTC/USDT' or 'BTC/USDT:USDT' → 'BTCUSDT'"""
    return symbol.split(":")[0].replace("/", "")


async def fetch_ohlcv_binance(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 1500,
) -> list:
    """
    Fetch historical OHLCV from Binance Futures.
    Returns list of [ts_ms, open, high, low, close, volume] — same format as ccxt.
    Paginates automatically for limit > 1000.
    """
    sym      = _to_binance_symbol(symbol)
    interval = _INTERVAL_MAP.get(timeframe, "1h")
    timeout  = aiohttp.ClientTimeout(total=15)
    all_candles: list = []

    async with aiohttp.ClientSession(timeout=timeout) as session:
        end_time = None
        remaining = limit

        while remaining > 0:
            batch = min(remaining, 1000)
            params: dict = {"symbol": sym, "interval": interval, "limit": batch}
            if end_time is not None:
                params["endTime"] = end_time

            async with session.get(_BASE, params=params) as r:
                if r.status != 200:
                    break
                raw = await r.json()

            if not raw:
                break

            # Binance returns: [open_time, open, high, low, close, volume, ...]
            parsed = [[int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])]
                      for c in raw]

            all_candles = parsed + all_candles
            remaining  -= len(parsed)

            if len(parsed) < batch:
                break  # no more history

            end_time = parsed[0][0] - 1   # fetch batch ending just before this one

    return all_candles[-limit:]


async def fetch_funding_rate_history_binance(symbol: str, limit: int = 1000) -> list:
    """
    Fetch historical funding rates from Binance Futures (public, no API key).
    Funding settles every 8h, so limit=1000 covers ~333 days — Bitget's own
    history caps at ~100 records (~33 days), which left funding_norm/
    funding_trend constant (0.0 feature importance) over most of any longer
    training window. Used for training data only, same reasoning as
    fetch_ohlcv_binance; live rate/execution stays on Bitget.

    Returns ccxt-compatible records ({"timestamp": ms, "fundingRate": float}),
    same shape ai.ml_signal._funding_to_series already expects.
    """
    sym = _to_binance_symbol(symbol)
    timeout = aiohttp.ClientTimeout(total=15)
    all_records: list = []

    async with aiohttp.ClientSession(timeout=timeout) as session:
        end_time = None
        remaining = limit

        while remaining > 0:
            batch = min(remaining, 1000)
            params: dict = {"symbol": sym, "limit": batch}
            if end_time is not None:
                params["endTime"] = end_time

            async with session.get(_FUNDING_BASE, params=params) as r:
                if r.status != 200:
                    break
                raw = await r.json()

            if not raw:
                break

            parsed = [{"timestamp": int(rec["fundingTime"]), "fundingRate": float(rec["fundingRate"])}
                      for rec in raw]

            all_records = parsed + all_records
            remaining -= len(parsed)

            if len(parsed) < batch:
                break  # no more history

            end_time = parsed[0]["timestamp"] - 1

    return all_records[-limit:]
