"""Free, no-API-key liquidation flow feed via Binance Futures' public
forceOrder WebSocket stream (`wss://fstream.binance.com/ws/!forceOrder@arr`,
broadcasts every liquidation across all USDT-M perp symbols on Binance, no
auth required).

Not a price-level heatmap (that needs Coinglass's Professional/Enterprise
tier, $699+/mo — checked 2026-07-23, no free/Hobbyist tier covers it despite
an earlier assumption otherwise, see project memory) — this is real-time
liquidation FLOW: how much long vs. short notional got force-closed recently.
Interpreted here as a momentum-confirming signal (heavy long liquidations =
longs getting flushed = bearish momentum; heavy short liquidations = a short
squeeze = bullish momentum), not a contrarian "magnet level" signal, since
flow alone (no price-level clustering) doesn't support the magnet
interpretation with any rigor.

One process-wide background collector (mirrors the SharedWallet/TradeJournal
singleton pattern) — a single WebSocket serves every symbol.
"""
import asyncio
import json
import time
from collections import defaultdict, deque

import aiohttp

STREAM_URL = "wss://fstream.binance.com/ws/!forceOrder@arr"
WINDOW_SECONDS = 4 * 3600   # keep 4h of events; callers can window down to less


class LiquidationStreamCollector:
    def __init__(self):
        self.events: dict[str, deque] = defaultdict(deque)   # symbol (Binance format, e.g. BTCUSDT) -> deque[(ts, side, usd)]
        self.running = False
        self._task: asyncio.Task = None
        self.connected = False
        self.last_error: str = ""

    async def _prune(self, symbol: str, now: float):
        dq = self.events[symbol]
        while dq and now - dq[0][0] > WINDOW_SECONDS:
            dq.popleft()

    async def _run(self):
        backoff = 5
        while self.running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(STREAM_URL, heartbeat=30, timeout=20) as ws:
                        self.connected = True
                        self.last_error = ""
                        backoff = 5
                        async for msg in ws:
                            if not self.running:
                                break
                            if msg.type != aiohttp.WSMsgType.TEXT:
                                continue
                            try:
                                data = json.loads(msg.data)
                                order = data.get("o", {})
                                symbol = order.get("s")
                                side = order.get("S")     # "SELL" liquidates a long, "BUY" liquidates a short
                                qty = float(order.get("q", 0))
                                price = float(order.get("ap", 0) or order.get("p", 0))
                                if not symbol or not qty or not price:
                                    continue
                                usd = qty * price
                                liq_side = "long" if side == "SELL" else "short"
                                now = time.time()
                                self.events[symbol].append((now, liq_side, usd))
                                await self._prune(symbol, now)
                            except Exception:
                                continue
            except Exception as e:
                self.connected = False
                self.last_error = str(e)
                print(f"[LiquidationStream] connection error: {e} — retrying in {backoff}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    def start(self):
        if self.running:
            return
        self.running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def flow_score(self, symbol: str, window_hours: float = 1.0) -> dict:
        """symbol: standard 'BTC/USDT' format, converted to Binance's 'BTCUSDT'.
        Returns {'available': False} if the stream isn't connected yet or has
        no events for this symbol in the window (new/thin symbols, or just
        quiet markets — liquidations are bursty, not constant)."""
        bsymbol = symbol.replace("/", "")
        dq = self.events.get(bsymbol)
        if not self.connected or not dq:
            return {"available": False}
        cutoff = time.time() - window_hours * 3600
        long_usd = sum(usd for ts, side, usd in dq if ts >= cutoff and side == "long")
        short_usd = sum(usd for ts, side, usd in dq if ts >= cutoff and side == "short")
        total = long_usd + short_usd
        if total < 1000:   # dust floor — a couple of small liquidations isn't a signal
            return {"available": False}
        return {
            "available": True,
            "long_liq_usd": round(long_usd, 0),
            "short_liq_usd": round(short_usd, 0),
            "dominant_side": "long" if long_usd > short_usd else "short",
            "dominance_ratio": round(max(long_usd, short_usd) / max(min(long_usd, short_usd), 1), 2),
        }


_collector: LiquidationStreamCollector = None


def get_collector() -> LiquidationStreamCollector:
    global _collector
    if _collector is None:
        _collector = LiquidationStreamCollector()
    return _collector
