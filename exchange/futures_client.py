import aiohttp
import ccxt.async_support as ccxt
import config

_TF_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000,
           "4h": 14_400_000, "1d": 86_400_000}

def _tf_ms(timeframe: str) -> int:
    return _TF_MS.get(timeframe, 3_600_000)

def to_futures_symbol(symbol: str) -> str:
    """Convert 'BTC/USDT' → 'BTC/USDT:USDT' for Bitget USDT-M perpetuals."""
    if symbol.endswith("/USDT") and ":USDT" not in symbol:
        return symbol + ":USDT"
    return symbol


class FuturesClient:
    def __init__(self):
        self._exchange = ccxt.bitget({
            "apiKey": config.BITGET_API_KEY,
            "secret": config.BITGET_SECRET,
            "password": config.BITGET_PASSPHRASE,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })

    async def fetch_ticker(self, symbol: str) -> dict:
        return await self._exchange.fetch_ticker(to_futures_symbol(symbol))

    async def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 200) -> list:
        sym = to_futures_symbol(symbol)
        if limit <= 1000:
            return await self._exchange.fetch_ohlcv(sym, timeframe, limit=limit)
        # Bitget caps at 1000 candles per request — paginate backwards
        all_candles: list = []
        since = None
        remaining = limit
        while remaining > 0:
            batch_size = min(remaining, 1000)
            batch = await self._exchange.fetch_ohlcv(sym, timeframe, limit=batch_size, since=since)
            if not batch:
                break
            all_candles = batch + all_candles
            remaining -= len(batch)
            if len(batch) < batch_size:
                break  # exchange returned fewer than requested — no more history
            since = batch[0][0] - _tf_ms(timeframe) * batch_size
        return all_candles[-limit:]

    async def fetch_order_book(self, symbol: str, limit: int = 20) -> dict:
        return await self._exchange.fetch_order_book(to_futures_symbol(symbol), limit)

    async def fetch_funding_rate(self, symbol: str) -> dict | None:
        """Returns None on failure — NOT a fake neutral rate. A prior version
        returned {"rate": 0.0001} on error, which is indistinguishable from a
        real (if small) reading and let strategies trade on it silently (audit
        finding H-05, 2026-07-23, see project memory). Callers must check for
        None and skip the symbol/cycle rather than treat a failed fetch as data."""
        try:
            fr = await self._exchange.fetch_funding_rate(to_futures_symbol(symbol))
            return {
                "rate": fr.get("fundingRate", 0),
                "next_ts": fr.get("nextFundingDatetime", ""),
            }
        except Exception:
            return None

    async def fetch_open_interest(self, symbol: str) -> dict | None:
        """Returns None on failure — see fetch_funding_rate docstring, same fix."""
        try:
            oi = await self._exchange.fetch_open_interest(to_futures_symbol(symbol))
            return {"open_interest": oi.get("openInterest", 0)}
        except Exception:
            return None

    async def fetch_funding_rate_history(self, symbol: str, limit: int = 100) -> list:
        try:
            return await self._exchange.fetch_funding_rate_history(to_futures_symbol(symbol), limit=limit)
        except Exception:
            return []

    async def fetch_current_oi(self, symbol: str) -> float | None:
        """Fetch current Open Interest (base currency). Returns None on failure
        or missing data — was 0.0 (a real-looking reading, not a "no data"
        signal), see fetch_funding_rate docstring for why that's unsafe."""
        sym = symbol.replace("/USDT", "").replace(":USDT", "") + "USDT"
        url = "https://api.bitget.com/api/v2/mix/market/open-interest"
        try:
            timeout = aiohttp.ClientTimeout(total=6)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params={"symbol": sym, "productType": "USDT-FUTURES"}) as r:
                    d = await r.json()
                    if d.get("code") == "00000" and d.get("data"):
                        items = d["data"].get("openInterestList", [])
                        if items:
                            return float(items[0]["size"])
        except Exception:
            pass
        return None

    async def fetch_cvd(self, symbol: str, limit: int = 500) -> dict | None:
        """Cumulative Volume Delta from taker fills — buy pressure vs sell pressure.
        Returns None on failure/no data — was a fake {"cvd_ratio": 0.5, ...} (an
        exactly-neutral reading indistinguishable from a real balanced market),
        see fetch_funding_rate docstring for why that's unsafe."""
        sym = symbol.replace("/USDT", "").replace(":USDT", "") + "USDT"
        url = "https://api.bitget.com/api/v2/mix/market/fills-history"
        try:
            timeout = aiohttp.ClientTimeout(total=8)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params={
                    "symbol": sym, "productType": "USDT-FUTURES", "limit": limit,
                }) as r:
                    d = await r.json()
                    if d.get("code") == "00000" and d.get("data"):
                        buy_vol = sell_vol = 0.0
                        for trade in d["data"]:
                            size = float(trade.get("size", 0))
                            if trade.get("side", "").lower() == "buy":
                                buy_vol += size
                            else:
                                sell_vol += size
                        total = buy_vol + sell_vol
                        if total > 0:
                            return {
                                "buy_vol": buy_vol, "sell_vol": sell_vol,
                                "cvd_net": buy_vol - sell_vol, "cvd_ratio": buy_vol / total,
                            }
        except Exception:
            pass
        return None

    async def fetch_market_sentiment(self, symbol: str) -> dict:
        """Fetch live L/S ratio and OI from Bitget public REST API. Missing keys
        (rather than fake values) signal "no data" for whichever sub-fetch
        failed — callers already use dict.get() with no-op fallbacks for these,
        so an absent key correctly contributes nothing instead of a fabricated
        neutral reading. Each sub-fetch has its own try/except (2026-07-23 fix,
        audit finding H-05 area) — previously one shared try/except around all
        three meant a failure in the 2nd or 3rd call silently discarded an
        already-successful earlier result too."""
        sym = symbol.replace("/USDT", "").replace(":USDT", "") + "USDT"
        base = "https://api.bitget.com/api/v2/mix/market"
        result: dict = {}
        timeout = aiohttp.ClientTimeout(total=8)   # 8s hard timeout per symbol
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                # Account long/short ratio (last 5 × 1H)
                async with session.get(f"{base}/long-short?symbol={sym}&productType=USDT-FUTURES&period=1H&limit=5") as r:
                    d = await r.json()
                    if d.get("code") == "00000" and d.get("data"):
                        latest = d["data"][-1]
                        result["long_ratio"]  = float(latest["longRatio"])
                        result["short_ratio"] = float(latest["shortRatio"])
            except Exception:
                pass

            try:
                # Position long/short ratio (size-weighted, more reliable)
                async with session.get(f"{base}/position-long-short?symbol={sym}&productType=USDT-FUTURES&period=1H&limit=5") as r:
                    d = await r.json()
                    if d.get("code") == "00000" and d.get("data"):
                        latest = d["data"][-1]
                        result["pos_long_ratio"]  = float(latest["longPositionRatio"])
                        result["pos_short_ratio"] = float(latest["shortPositionRatio"])
            except Exception:
                pass

            try:
                # Current open interest (in base currency)
                async with session.get(f"{base}/open-interest?symbol={sym}&productType=USDT-FUTURES") as r:
                    d = await r.json()
                    if d.get("code") == "00000" and d.get("data"):
                        oi_list = d["data"].get("openInterestList", [])
                        if oi_list:
                            result["open_interest"] = float(oi_list[0]["size"])
            except Exception:
                pass
        return result

    async def set_leverage(self, symbol: str, leverage: int):
        await self._exchange.set_leverage(leverage, to_futures_symbol(symbol))

    async def open_long(self, symbol: str, amount: float, leverage: int) -> dict:
        await self.set_leverage(symbol, leverage)
        return await self._exchange.create_order(
            to_futures_symbol(symbol), "market", "buy", amount,
            params={"tdMode": "cross"}
        )

    async def open_short(self, symbol: str, amount: float, leverage: int) -> dict:
        await self.set_leverage(symbol, leverage)
        return await self._exchange.create_order(
            to_futures_symbol(symbol), "market", "sell", amount,
            params={"tdMode": "cross"}
        )

    async def close_position(self, symbol: str, side: str, amount: float) -> dict:
        close_side = "sell" if side == "long" else "buy"
        return await self._exchange.create_order(
            to_futures_symbol(symbol), "market", close_side, amount,
            params={"reduceOnly": True}
        )

    async def fetch_positions(self) -> list:
        return await self._exchange.fetch_positions()

    async def close(self):
        await self._exchange.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
