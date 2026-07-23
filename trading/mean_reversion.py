"""
Mean-Reversion Engine — range trading, the direct counterpart to AutoTrader's
trend-following (2026-07-23, see project memory / DeepSeek design discussion).

Reuses AutoTrader's existing scalp-fallback entry thresholds (RSI/Bollinger %B,
ADX<20 regime gate, trading/autotrader.py::_scalp_signal) — already production-
tested there as a fallback, just never had its own capital budget or performance
tracking, and only fired when the ML trend path had already declined. This makes
it a first-class, independently-measured strategy on the shared wallet instead.

R:R is deliberately the OPPOSITE of AutoTrader's trend path: SL=2x ATR wider than
TP=1x ATR (not tighter). Mean-reversion payoffs are asymmetric — frequent small
reversions, occasional large losses when a range breaks into a trend — so the
stop needs room to not get clipped by normal range noise, while the target stays
close to where the reversion is expected to stall out.

Reuses FuturesPaperEngine (position mgmt, SL/TP, fee accounting) with its own
state file so it runs independently of AutoTrader on the same SharedWallet —
same pattern as FundingHarvestEngine/GridEngine.
"""
import asyncio
from datetime import datetime
from typing import Optional

import pandas as pd

from exchange.futures_client import FuturesClient
from trading.futures_paper import FuturesPaperEngine
from ai.ml_signal import get_indicators
from notifications.telegram import notify_fire_and_forget
from trading.journal import get_journal

MR_STATE_FILE = "data/mean_reversion_state.json"


def _to_df(ohlcv: list) -> pd.DataFrame:
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df


class MeanReversionHarvester:
    """Scans candidate symbols for RSI/Bollinger extremes in ranging markets
    (ADX < adx_max) and manages entries/exits. Pure rule-based, no ML — the
    opposite market regime of AutoTrader's trend model."""

    def __init__(
        self,
        symbols: list[str] = None,
        engine: FuturesPaperEngine = None,
        interval_seconds: int = 300,
        adx_max: float = 20.0,
        rsi_oversold: float = 32.0,
        rsi_overbought: float = 68.0,
        bb_pct_low: float = 0.15,
        bb_pct_high: float = 0.85,
        atr_sl_mult: float = 2.0,     # wider than TP on purpose, see module docstring
        atr_tp_mult: float = 1.0,
        risk_pct: float = 0.005,      # 0.5% of engine equity per trade
        leverage: int = 3,
        max_total_margin_pct: float = 0.20,   # cap: total MR margin as % of engine equity
    ):
        self.symbols = symbols or ["BTC/USDT", "ETH/USDT"]
        self.engine = engine or FuturesPaperEngine(state_file=MR_STATE_FILE)
        self.interval = interval_seconds
        self.adx_max = adx_max
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.bb_pct_low = bb_pct_low
        self.bb_pct_high = bb_pct_high
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.risk_pct = risk_pct
        self.leverage = leverage
        self.max_total_margin_pct = max_total_margin_pct

        self.running = False
        self._task: Optional[asyncio.Task] = None
        self.cycle_count = 0
        self.log: list[dict] = []
        self.live_prices: dict[str, float] = {}

    def _log(self, level: str, msg: str, symbol: str = None):
        entry = {"ts": datetime.now().isoformat(), "level": level, "symbol": symbol or "ALL", "msg": msg}
        self.log.append(entry)
        self.log = self.log[-300:]
        tag = f"[{symbol}] " if symbol else ""
        print(f"[{entry['ts'][11:19]}] [MeanReversion] [{level}] {tag}{msg}")
        if level == "TRADE":
            notify_fire_and_forget(f"↔️ <b>Mean Reversion</b> {tag}\n{msg}")

    def _signal(self, indicators: dict) -> tuple[str, str]:
        adx = indicators.get("adx", 100)
        if adx >= self.adx_max:
            return "hold", ""
        rsi = indicators.get("rsi", 50)
        bb_pct = indicators.get("bb_pct", 0.5)
        if rsi <= self.rsi_oversold and bb_pct <= self.bb_pct_low:
            return "buy", f"RSI überverkauft ({rsi:.0f}) nahe unterem BB-Band ({bb_pct:.2f}), ADX={adx:.0f}"
        if rsi >= self.rsi_overbought and bb_pct >= self.bb_pct_high:
            return "sell", f"RSI überkauft ({rsi:.0f}) nahe oberem BB-Band ({bb_pct:.2f}), ADX={adx:.0f}"
        return "hold", ""

    def _engine_equity(self) -> float:
        # This engine's own attributable equity (free balance + its own locked
        # margin) — NOT the full shared-wallet portfolio, which other engines
        # also draw from. Mirrors the sizing approach in FundingHarvestEngine.
        return self.engine.wallet.balance + sum(p.margin for p in self.engine.positions.values())

    async def _cycle(self):
        self.cycle_count += 1
        async with FuturesClient() as client:
            for symbol in self.symbols:
                try:
                    ohlcv = await client.fetch_ohlcv(symbol, "1h", 300)
                    df = _to_df(ohlcv)
                    price = float(df["close"].iloc[-1])
                    self.live_prices[symbol] = price

                    trigger = self.engine.check_sl_tp_liquidation(symbol, price)
                    if trigger:
                        record = self.engine.close_position(symbol, price, trigger)
                        self._log("TRADE", f"{trigger.upper()} closed @ {price:.4f} | "
                                            f"net_pnl={record['pnl']:+.2f}", symbol)
                        get_journal().record("mean_reversion", symbol, "close", trigger, pnl=record["pnl"])
                        continue

                    if symbol in self.engine.positions:
                        continue   # one position per symbol, managed via SL/TP only

                    indicators = get_indicators(df)
                    label, reason = self._signal(indicators)
                    if label == "hold":
                        continue

                    equity = self._engine_equity()
                    margin_used = sum(p.margin for p in self.engine.positions.values())
                    if margin_used >= equity * self.max_total_margin_pct:
                        continue   # capital budget for this strategy is exhausted

                    atr = indicators.get("atr", price * 0.015)
                    side = "long" if label == "buy" else "short"
                    sl = price - atr * self.atr_sl_mult if side == "long" else price + atr * self.atr_sl_mult
                    tp = price + atr * self.atr_tp_mult if side == "long" else price - atr * self.atr_tp_mult

                    risk_amount = equity * self.risk_pct
                    sl_dist = max(abs(price - sl), price * 0.003)
                    raw_amount = risk_amount / sl_dist
                    remaining_margin_budget = equity * self.max_total_margin_pct - margin_used
                    cap_amount = (remaining_margin_budget * self.leverage) / price
                    amount = min(raw_amount, cap_amount)
                    if amount * price / self.leverage < 10:   # dust guard
                        continue

                    record = self.engine.open_position(
                        symbol, side, amount, price, self.leverage,
                        stop_loss=sl, take_profit=tp, mode="mean_reversion",
                    )
                    self._log("TRADE", f"OPEN {side.upper()} @ {price:.4f} | {reason} | "
                                        f"SL={sl:.4f} TP={tp:.4f}", symbol)
                    get_journal().record("mean_reversion", symbol, f"open_{side}", reason)
                except Exception as e:
                    self._log("ERROR", f"Cycle error: {e}", symbol)

            self.engine.record_equity(self.live_prices)

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
        self._log("INFO", f"MeanReversionHarvester started — {self.symbols} | every {self.interval}s | "
                          f"RSI {self.rsi_oversold}/{self.rsi_overbought} | ADX<{self.adx_max} | "
                          f"SL={self.atr_sl_mult}xATR TP={self.atr_tp_mult}xATR | leverage={self.leverage}x")

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._log("INFO", "MeanReversionHarvester stopped")

    def status(self) -> dict:
        return {
            "running": self.running,
            "symbols": self.symbols,
            "interval_seconds": self.interval,
            "adx_max": self.adx_max,
            "rsi_oversold": self.rsi_oversold,
            "rsi_overbought": self.rsi_overbought,
            "bb_pct_low": self.bb_pct_low,
            "bb_pct_high": self.bb_pct_high,
            "atr_sl_mult": self.atr_sl_mult,
            "atr_tp_mult": self.atr_tp_mult,
            "risk_pct": self.risk_pct,
            "leverage": self.leverage,
            "max_total_margin_pct": self.max_total_margin_pct,
            "cycle_count": self.cycle_count,
            "engine": self.engine.status(self.live_prices),
        }

    def get_log(self, limit: int = 50) -> list:
        return list(reversed(self.log[-limit:]))
