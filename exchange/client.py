import ccxt.async_support as ccxt
import asyncio
from typing import Optional
import config


class BitgetClient:
    def __init__(self, paper: bool = False):
        self.paper = paper
        params = {
            "apiKey": config.BITGET_API_KEY,
            "secret": config.BITGET_SECRET,
            "password": config.BITGET_PASSPHRASE,
            "enableRateLimit": True,
        }
        self._exchange = ccxt.bitget(params)

    async def fetch_ticker(self, symbol: str) -> dict:
        return await self._exchange.fetch_ticker(symbol)

    async def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 200) -> list:
        return await self._exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

    async def fetch_order_book(self, symbol: str, limit: int = 20) -> dict:
        return await self._exchange.fetch_order_book(symbol, limit)

    async def estimate_execution(self, symbol: str, side: str, notional_usdt: float,
                                  max_levels: int = 20) -> dict | None:
        """Spot-side counterpart to FuturesClient.estimate_execution() — see
        its docstring (2026-07-23, execution-realism round, project memory).
        Used by Funding Harvest's spot leg."""
        from trading.execution_sim import walk_orderbook
        try:
            ob = await self.fetch_order_book(symbol, limit=max_levels)
        except Exception:
            return None
        levels = ob.get("asks") if side == "buy" else ob.get("bids")
        return walk_orderbook(levels, notional_usdt)

    async def fetch_balance(self) -> dict:
        if self.paper:
            raise RuntimeError("Use PaperEngine for paper balance")
        return await self._exchange.fetch_balance()

    async def create_order(self, symbol: str, order_type: str, side: str, amount: float, price: Optional[float] = None) -> dict:
        if self.paper:
            raise RuntimeError("Use PaperEngine for paper orders")
        return await self._exchange.create_order(symbol, order_type, side, amount, price)

    async def fetch_markets(self) -> list:
        return await self._exchange.fetch_markets()

    async def close(self):
        await self._exchange.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
