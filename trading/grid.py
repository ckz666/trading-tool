"""
Grid trading: place a ladder of buy levels below and sell levels above the
current price within a fixed range. Profits from price oscillating within
the range, not from predicting direction — this is the structural edge
regime detection was already built for (ADX < 20 = ranging).

Uses maker-style fees (limit orders resting in the book), unlike the
taker-fee assumption used for the momentum/scalp engines.

Paper-trading only, fully separate from the other engines.
"""
import asyncio
import json
import os
import uuid
from datetime import datetime
from typing import Optional

import pandas as pd

from exchange.futures_client import FuturesClient
from ai.ml_signal import get_indicators
from trading.wallet import SharedWallet

MAKER_FEE = 0.0002   # matches futures_paper.py's MAKER_FEE
STATE_FILE = "data/grid_state.json"


def _to_df(ohlcv: list) -> pd.DataFrame:
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df


class GridEngine:
    """Paper P&L bookkeeping for one grid per symbol. No strategy/regime
    logic here — that lives in GridTrader below."""

    def __init__(self, wallet: SharedWallet = None, initial_balance: float = 10000.0):
        self.wallet = wallet or SharedWallet(initial_balance)
        self.grids: dict[str, dict] = {}
        self.trade_history: list[dict] = []
        self.total_realized_pnl: float = 0.0
        self.total_fees_paid: float = 0.0
        self._load()

    def _save(self):
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        state = {
            "total_realized_pnl": self.total_realized_pnl,
            "total_fees_paid": self.total_fees_paid,
            "trade_history": self.trade_history,
            "grids": self.grids,
            "saved_at": datetime.now().isoformat(),
        }
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, STATE_FILE)
        self.wallet._save()

    def _load(self):
        if not os.path.exists(STATE_FILE):
            return
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
            self.total_realized_pnl = state.get("total_realized_pnl", 0.0)
            self.total_fees_paid = state.get("total_fees_paid", 0.0)
            self.trade_history = state.get("trade_history", [])
            self.grids = state.get("grids", {})
            print(f"[Grid] Loaded state: wallet balance={self.wallet.balance:.2f} USDT, "
                  f"{len(self.grids)} active grids, {len(self.trade_history)} trades")
        except Exception as e:
            print(f"[Grid] Could not load state: {e} — starting fresh")

    def open_grid(self, symbol: str, lower: float, upper: float, n_levels: int,
                  capital: float) -> dict:
        if symbol in self.grids:
            raise ValueError(f"Grid already active for {symbol}")
        if capital > self.wallet.balance:
            raise ValueError(f"Insufficient balance: need {capital:.2f}, have {self.wallet.balance:.2f}")
        if lower >= upper or n_levels < 2:
            raise ValueError("Invalid grid range/levels")

        step = (upper / lower) ** (1 / n_levels)
        lines = [round(lower * step ** i, 8) for i in range(n_levels + 1)]
        capital_per_level = capital / n_levels

        self.wallet.balance -= capital
        grid = {
            "symbol": symbol, "lower": lower, "upper": upper, "n_levels": n_levels,
            "lines": lines,
            "holding": [False] * n_levels,   # holding[i] = bought at lines[i], waiting to sell at lines[i+1]
            "qty": [0.0] * n_levels,
            "capital_per_level": capital_per_level,
            "reserve": capital,               # cash set aside for this grid's own buys/sells
            "realized_pnl": 0.0,
            "fees_paid": 0.0,
            "trades": 0,
            "opened_at": datetime.now().isoformat(),
        }
        self.grids[symbol] = grid

        record = {
            "id": str(uuid.uuid4())[:8], "action": "open_grid", "symbol": symbol,
            "lower": lower, "upper": upper, "n_levels": n_levels, "capital": capital,
            "ts": datetime.now().isoformat(),
        }
        self.trade_history.append(record)
        self._save()
        return record

    def update(self, symbol: str, price: float) -> list[dict]:
        """Check for level crosses at the current price and execute virtual
        maker fills. Returns the list of fills executed this call."""
        grid = self.grids.get(symbol)
        if not grid:
            return []
        fills = []
        lines = grid["lines"]

        # Buys: price at/below a line that isn't holding yet
        for i in range(grid["n_levels"]):
            if not grid["holding"][i] and price <= lines[i]:
                cost = grid["capital_per_level"]
                fee = cost * MAKER_FEE
                if grid["reserve"] < cost + fee:
                    continue
                qty = cost / lines[i]
                grid["reserve"] -= (cost + fee)
                grid["holding"][i] = True
                grid["qty"][i] = qty
                grid["fees_paid"] += fee
                grid["trades"] += 1
                self.total_fees_paid += fee
                fills.append({"side": "buy", "level": i, "price": lines[i], "qty": qty})

        # Sells: price at/above the line one above a held level
        for i in range(grid["n_levels"]):
            if grid["holding"][i] and price >= lines[i + 1]:
                qty = grid["qty"][i]
                proceeds = qty * lines[i + 1]
                fee = proceeds * MAKER_FEE
                pnl = proceeds - fee - (qty * lines[i])
                grid["reserve"] += (proceeds - fee)
                grid["holding"][i] = False
                grid["qty"][i] = 0.0
                grid["realized_pnl"] += pnl
                grid["fees_paid"] += fee
                grid["trades"] += 1
                self.total_realized_pnl += pnl
                self.total_fees_paid += fee
                fills.append({"side": "sell", "level": i, "price": lines[i + 1], "qty": qty, "pnl": pnl})

        if fills:
            for fill in fills:
                self.trade_history.append({
                    "id": str(uuid.uuid4())[:8], "action": f"grid_{fill['side']}", "symbol": symbol,
                    "price": fill["price"], "qty": fill["qty"], "pnl": fill.get("pnl"),
                    "ts": datetime.now().isoformat(),
                })
            self._save()
        return fills

    def close_grid(self, symbol: str, price: float, reason: str = "manual") -> dict:
        """Liquidate all held inventory at the current price and return the
        grid's reserve cash to the main balance."""
        grid = self.grids.pop(symbol, None)
        if not grid:
            raise ValueError(f"No active grid for {symbol}")

        liquidation_pnl = 0.0
        for i in range(grid["n_levels"]):
            if grid["holding"][i]:
                qty = grid["qty"][i]
                proceeds = qty * price
                fee = proceeds * MAKER_FEE
                pnl = proceeds - fee - (qty * grid["lines"][i])
                grid["reserve"] += (proceeds - fee)
                grid["fees_paid"] += fee
                grid["realized_pnl"] += pnl
                liquidation_pnl += pnl
                self.total_fees_paid += fee
                self.total_realized_pnl += pnl

        self.wallet.balance += max(grid["reserve"], 0)

        record = {
            "id": str(uuid.uuid4())[:8], "action": "close_grid", "reason": reason, "symbol": symbol,
            "price": price, "realized_pnl": round(grid["realized_pnl"], 4),
            "liquidation_pnl": round(liquidation_pnl, 4), "fees_paid": round(grid["fees_paid"], 4),
            "trades": grid["trades"], "ts": datetime.now().isoformat(),
        }
        self.trade_history.append(record)
        self._save()
        return record

    def grid_value(self, symbol: str, price: float) -> float:
        grid = self.grids.get(symbol)
        if not grid:
            return 0.0
        inventory_value = sum(grid["qty"][i] * price for i in range(grid["n_levels"]) if grid["holding"][i])
        return grid["reserve"] + inventory_value

    def portfolio_value(self, prices: dict[str, float]) -> float:
        """Shared-wallet balance + this engine's own grids only — not the
        true account total when other engines also hold positions, see
        /api/portfolio/total for that."""
        total = self.wallet.balance
        for sym in self.grids:
            if sym in prices:
                total += self.grid_value(sym, prices[sym])
        return max(total, 0)

    def status(self, prices: dict[str, float] = None) -> dict:
        prices = prices or {}
        grids_out = {}
        for sym, g in self.grids.items():
            price = prices.get(sym)
            grids_out[sym] = {
                "symbol": sym, "lower": g["lower"], "upper": g["upper"], "n_levels": g["n_levels"],
                "realized_pnl": round(g["realized_pnl"], 4), "fees_paid": round(g["fees_paid"], 4),
                "trades": g["trades"], "opened_at": g["opened_at"],
                "levels_holding": sum(g["holding"]),
                "value": round(self.grid_value(sym, price), 2) if price else None,
                "in_range": (g["lower"] <= price <= g["upper"]) if price else None,
            }
        return {
            "balance": round(self.wallet.balance, 2),
            "portfolio_value": round(self.portfolio_value(prices), 2),
            "total_realized_pnl": round(self.total_realized_pnl, 2),
            "total_fees_paid": round(self.total_fees_paid, 2),
            "active_grids": len(self.grids),
            "grids": grids_out,
            "trade_count": len(self.trade_history),
        }


class GridTrader:
    """Deploys a grid only when the symbol is in a ranging regime (reuses the
    existing ADX-based regime detector), sized around recent volatility (ATR)
    rather than a fixed percent. The critical risk control is the stop-loss:
    if price breaks meaningfully below the grid's lower bound, the classic
    grid-bot failure mode is holding falling inventory indefinitely — this
    closes the whole grid instead of hoping it recovers."""

    def __init__(
        self,
        symbols: list[str] = None,
        engine: GridEngine = None,
        interval_seconds: int = 300,
        timeframe: str = "1h",
        n_levels: int = 10,
        atr_range_mult: float = 2.0,        # grid spans price +/- atr_range_mult*ATR
        capital_per_grid_pct: float = 0.20,  # fraction of free balance allocated per new grid
        stop_loss_pct: float = 0.03,         # close if price breaks this far below the range
        adx_ranging_max: float = 20.0,
    ):
        self.symbols = symbols or ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
        self.engine = engine or GridEngine()
        self.interval = interval_seconds
        self.timeframe = timeframe
        self.n_levels = n_levels
        self.atr_range_mult = atr_range_mult
        self.capital_per_grid_pct = capital_per_grid_pct
        self.stop_loss_pct = stop_loss_pct
        self.adx_ranging_max = adx_ranging_max

        self.running = False
        self._task: Optional[asyncio.Task] = None
        self.cycle_count = 0
        self.log: list[dict] = []
        self.last_regime: dict[str, str] = {}

    def _log(self, level: str, msg: str, symbol: str = None):
        entry = {"ts": datetime.now().isoformat(), "level": level, "symbol": symbol or "ALL", "msg": msg}
        self.log.append(entry)
        self.log = self.log[-300:]
        tag = f"[{symbol}] " if symbol else ""
        print(f"[{entry['ts'][11:19]}] [Grid] [{level}] {tag}{msg}")

    async def _cycle(self):
        self.cycle_count += 1
        async with FuturesClient() as client:
            for symbol in self.symbols:
                try:
                    ohlcv = await client.fetch_ohlcv(symbol, self.timeframe, 300)
                    df = _to_df(ohlcv)
                    price = float(df["close"].iloc[-1])
                    indicators = get_indicators(df)
                    regime = indicators.get("regime", "unknown")
                    atr = indicators.get("atr", price * 0.015)
                    self.last_regime[symbol] = regime

                    has_grid = symbol in self.engine.grids

                    if has_grid:
                        grid = self.engine.grids[symbol]
                        fills = self.engine.update(symbol, price)
                        for f in fills:
                            extra = f" pnl={f['pnl']:+.4f}" if "pnl" in f else ""
                            self._log("TRADE", f"{f['side'].upper()} @ {f['price']:.4f} qty={f['qty']:.6f}{extra}", symbol)

                        if price < grid["lower"] * (1 - self.stop_loss_pct):
                            record = self.engine.close_grid(symbol, price, "stop_loss")
                            self._log("WARN", f"Stop-loss — price broke below range, closed @ "
                                              f"realized_pnl={record['realized_pnl']:+.2f}", symbol)

                    elif regime == "ranging":
                        lower = price - atr * self.atr_range_mult
                        upper = price + atr * self.atr_range_mult
                        capital = self.engine.wallet.balance * self.capital_per_grid_pct
                        if capital > 50 and lower > 0:
                            self.engine.open_grid(symbol, lower, upper, self.n_levels, capital)
                            self._log("TRADE", f"OPEN GRID {lower:.4f}-{upper:.4f} "
                                              f"({self.n_levels} levels, ${capital:.0f})", symbol)
                        else:
                            self._log("INFO", f"Ranging but insufficient free balance for a new grid", symbol)
                    else:
                        self._log("INFO", f"regime={regime} — not ranging, no grid", symbol)
                except Exception as e:
                    self._log("ERROR", f"Cycle error: {e}", symbol)

    async def _loop(self):
        while self.running:
            try:
                await self._cycle()
            except Exception as e:
                self._log("ERROR", f"Loop error: {e}")
            await asyncio.sleep(self.interval)

    def start(self):
        if self.running:
            return
        self.running = True
        self._task = asyncio.create_task(self._loop())
        self._log("INFO", f"GridTrader started — {self.symbols} | every {self.interval}s | "
                          f"{self.n_levels} levels | stop_loss={self.stop_loss_pct:.0%} below range")

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._log("INFO", "GridTrader stopped")

    def status(self) -> dict:
        return {
            "running": self.running,
            "symbols": self.symbols,
            "interval_seconds": self.interval,
            "n_levels": self.n_levels,
            "atr_range_mult": self.atr_range_mult,
            "stop_loss_pct": self.stop_loss_pct,
            "cycle_count": self.cycle_count,
            "last_regime": self.last_regime,
        }
