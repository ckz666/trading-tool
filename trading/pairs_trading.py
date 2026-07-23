"""
Pairs Trading Engine — market-neutral BTC/ETH spread mean-reversion (2026-07-23,
DeepSeek design discussion, see project memory). Third of the diversification
strategies alongside Funding Harvest and Mean Reversion, and the most
uncorrelated one: long the underperformer + short the outperformer on the
log(ETH/BTC) spread's rolling z-score, profiting from spread convergence
rather than from either asset's own direction. If BTC and ETH both fall
together, the pair can still profit as long as the SPREAD reverts.

Signal: z = (s - rolling_mean(s, window)) / rolling_std(s, window), where
s = log(close_eth) - log(close_btc), window=200 hourly bars (~8.3 days).
Entry at |z| >= z_entry, exit at z crossing z_exit (mean reversion achieved),
hard stop at |z| >= z_stop (spread kept diverging — thesis failed), and a
time-stop after max_hold_bars (stale pair, cut it loose either way).

Reuses FuturesPaperEngine (own state file) the same way Mean Reversion does —
one leg per symbol, tracked as a pair via self.open_pair rather than through
the engine itself (which only knows about individual single-symbol positions).
"""
import asyncio
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from exchange.futures_client import FuturesClient
from trading.futures_paper import FuturesPaperEngine
from notifications.telegram import notify_fire_and_forget
from trading.journal import get_journal
from trading.portfolio_risk import get_allocator as get_risk_allocator
from trading.execution_sim import simulate_fill
from ai.vol_regime import classify_vol_regime

PAIRS_STATE_FILE = "data/pairs_trading_state.json"


def _to_df(ohlcv: list) -> pd.DataFrame:
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df


class PairsTradingHarvester:
    def __init__(
        self,
        symbol_a: str = "ETH/USDT",
        symbol_b: str = "BTC/USDT",
        engine: FuturesPaperEngine = None,
        interval_seconds: int = 300,
        window: int = 200,
        z_entry: float = 2.0,
        z_exit: float = 0.0,
        z_stop: float = 3.0,
        max_hold_bars: int = 48,
        risk_pct: float = 0.01,
        leverage: int = 3,
        max_total_margin_pct: float = 0.20,
        funding_diff_floor: float = -0.001,   # skip entry if the short leg's funding is much worse
    ):
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b
        self.engine = engine or FuturesPaperEngine(state_file=PAIRS_STATE_FILE)
        self.interval = interval_seconds
        self.window = window
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.z_stop = z_stop
        self.max_hold_bars = max_hold_bars
        self.risk_pct = risk_pct
        self.leverage = leverage
        self.max_total_margin_pct = max_total_margin_pct
        self.funding_diff_floor = funding_diff_floor

        self.running = False
        self._task: Optional[asyncio.Task] = None
        self.cycle_count = 0
        self.log: list[dict] = []
        self.live_prices: dict[str, float] = {}
        # {"direction": "long_a_short_b"|"short_a_long_b", "opened_bar": int, "entry_z": float}
        self.open_pair: Optional[dict] = None
        self._bar_count = 0
        self.last_z: Optional[float] = None

    def _log(self, level: str, msg: str, symbol: str = None):
        entry = {"ts": datetime.now().isoformat(), "level": level, "symbol": symbol or "ALL", "msg": msg}
        self.log.append(entry)
        self.log = self.log[-300:]
        tag = f"[{symbol}] " if symbol else ""
        print(f"[{entry['ts'][11:19]}] [PairsTrading] [{level}] {tag}{msg}")
        if level == "TRADE":
            notify_fire_and_forget(f"⚖️ <b>Pairs Trading</b> {tag}\n{msg}")

    def _compute_zscore(self, df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[Optional[float], Optional[float]]:
        n = min(len(df_a), len(df_b))
        if n < self.window + 1:
            return None, None
        close_a, close_b = df_a["close"].iloc[-n:].values, df_b["close"].iloc[-n:].values
        s = pd.Series(np.log(close_a) - np.log(close_b))
        mean = s.rolling(self.window).mean().iloc[-1]
        std = s.rolling(self.window).std(ddof=0).iloc[-1]
        if pd.isna(std) or std < 1e-8:
            return 0.0, float(std) if not pd.isna(std) else 0.0
        z = (s.iloc[-1] - mean) / std
        return float(z), float(std)

    async def _close_pair(self, client, price_a: float, price_b: float, reason: str):
        # Execution-realism (2026-07-23, see project memory): each leg exits
        # on its own side (closing a long = sell, closing a short = buy) —
        # the two legs of a pair are NOT symmetric here, they're opposite.
        async def _exit_fill(symbol, price):
            pos = self.engine.positions.get(symbol)
            if pos is None:
                return price, 1.0
            exit_side = "sell" if pos.side == "long" else "buy"
            fill_price, _fill_amount, fill_info = await simulate_fill(client, symbol, exit_side, price, pos.amount)
            filled_pct = fill_info["filled_pct"] if fill_info["simulated"] else 1.0
            return fill_price, filled_pct

        (exit_a, filled_a), (exit_b, filled_b) = await asyncio.gather(
            _exit_fill(self.symbol_a, price_a), _exit_fill(self.symbol_b, price_b))

        def _close_leg(symbol, exit_price, filled_pct, reason):
            # Bug fix (2026-07-23, found via DeepSeek code review of the
            # same pattern in AutoTrader): a thin book can't always absorb
            # the whole exit at the reported price — close only what
            # actually filled rather than booking PnL for size that never
            # traded there. Leaves any unfilled remainder open; the next
            # cycle's liquidation check still watches it even though
            # self.open_pair's z-score tracking is cleared below regardless.
            if symbol not in self.engine.positions:
                return None
            if filled_pct < 0.999:
                return self.engine.partial_close_position(symbol, exit_price, filled_pct, reason)
            return self.engine.close_position(symbol, exit_price, reason)

        rec_a = _close_leg(self.symbol_a, exit_a, filled_a, reason)
        rec_b = _close_leg(self.symbol_b, exit_b, filled_b, reason)
        pnl_a = rec_a["pnl"] if rec_a else 0.0
        pnl_b = rec_b["pnl"] if rec_b else 0.0
        partial_tag = " (Teilfüllung, Rest bleibt offen)" if (filled_a < 0.999 or filled_b < 0.999) else ""
        self._log("TRADE", f"CLOSE PAIR ({reason}) z={self.last_z:.2f}{partial_tag} | "
                            f"{self.symbol_a}={pnl_a:+.2f} {self.symbol_b}={pnl_b:+.2f} | total={pnl_a+pnl_b:+.2f}")
        get_journal().record("pairs_trading", f"{self.symbol_a}/{self.symbol_b}", "close_pair",
                              f"{reason}, z={self.last_z:.2f}", pnl=pnl_a + pnl_b)
        self.open_pair = None

    async def _cycle(self):
        self.cycle_count += 1
        self._bar_count += 1
        async with FuturesClient() as client:
            try:
                ohlcv_a, ohlcv_b = await asyncio.gather(
                    client.fetch_ohlcv(self.symbol_a, "1h", self.window + 20),
                    client.fetch_ohlcv(self.symbol_b, "1h", self.window + 20),
                )
                df_a, df_b = _to_df(ohlcv_a), _to_df(ohlcv_b)
                price_a, price_b = float(df_a["close"].iloc[-1]), float(df_b["close"].iloc[-1])
                self.live_prices[self.symbol_a] = price_a
                self.live_prices[self.symbol_b] = price_b

                z, std = self._compute_zscore(df_a, df_b)
                if z is None:
                    self._log("INFO", f"Warmup: brauche {self.window} Bars Spread-Historie")
                    return
                self.last_z = z

                # Liquidation is still a real per-leg risk even though the pair is
                # meant to be market-neutral — check it before anything else.
                for sym, price in ((self.symbol_a, price_a), (self.symbol_b, price_b)):
                    if sym in self.engine.positions and self.engine.check_sl_tp_liquidation(sym, price) == "liquidation":
                        pos = self.engine.positions.get(sym)
                        exit_side = "sell" if pos.side == "long" else "buy"
                        exit_price, _fill_amount, fill_info = await simulate_fill(client, sym, exit_side, price, pos.amount)
                        filled_pct = fill_info["filled_pct"] if fill_info["simulated"] else 1.0
                        if filled_pct < 0.999:
                            self.engine.partial_close_position(sym, exit_price, filled_pct, "liquidation")
                        else:
                            self.engine.close_position(sym, exit_price, "liquidation")
                        self._log("ERROR", f"{sym} LIQUIDATED — Pair-Hedge gebrochen", sym)
                        self.open_pair = None

                if self.open_pair:
                    direction = self.open_pair["direction"]
                    bars_held = self._bar_count - self.open_pair["opened_bar"]
                    reason = None
                    if direction == "long_a_short_b" and z <= self.z_exit:
                        reason = "z_reverted"
                    elif direction == "short_a_long_b" and z >= self.z_exit:
                        reason = "z_reverted"
                    elif direction == "long_a_short_b" and z <= -self.z_stop:
                        reason = "stop"
                    elif direction == "short_a_long_b" and z >= self.z_stop:
                        reason = "stop"
                    elif bars_held >= self.max_hold_bars:
                        reason = "time_stop"

                    if reason:
                        await self._close_pair(client, price_a, price_b, reason)
                    return   # one pair at a time — managed only while open

                if abs(z) < self.z_entry:
                    return

                fr_a, fr_b = await asyncio.gather(
                    client.fetch_funding_rate(self.symbol_a),
                    client.fetch_funding_rate(self.symbol_b),
                )
                if fr_a is None or fr_b is None:
                    self._log("WARN", "Funding-Rate-Fetch fehlgeschlagen — kein Entry diesen Zyklus")
                    return

                if z > self.z_entry:
                    direction = "short_a_long_b"
                    short_rate, long_rate = fr_a["rate"], fr_b["rate"]
                else:
                    direction = "long_a_short_b"
                    long_rate, short_rate = fr_a["rate"], fr_b["rate"]

                funding_diff = short_rate - long_rate
                if funding_diff < self.funding_diff_floor:
                    self._log("INFO", f"Funding-Differential zu negativ ({funding_diff*100:.4f}%/8h) — Entry unterdrückt")
                    return

                equity = self.engine.wallet.balance + sum(p.margin for p in self.engine.positions.values())
                margin_used = sum(p.margin for p in self.engine.positions.values())
                if margin_used >= equity * self.max_total_margin_pct:
                    return

                avg_price = (price_a + price_b) / 2
                sl_usdt = (self.z_stop - self.z_entry) * std * avg_price
                if sl_usdt <= 0:
                    return
                # Portfolio-level Kelly allocator (2026-07-23, see project
                # memory) — same override pattern as Mean Reversion.
                risk_pct = get_risk_allocator().get_risk_pct("pairs_trading", default=self.risk_pct)
                # Vol-size modulator (2026-07-23, execution-realism round item
                # #2, see project memory) — applied off symbol_b (the more
                # liquid/representative "market" leg, e.g. BTC) as a proxy for
                # general market stress: spreads and correlations both tend to
                # get less reliable when the wider market is turbulent, even
                # though the pair itself is directionally market-neutral.
                vol_regime = classify_vol_regime(df_b)
                risk_pct = risk_pct * vol_regime["continuous_risk_multiplier"]
                risk_amount = equity * risk_pct
                notional_per_leg = risk_amount / (sl_usdt / avg_price)
                remaining_budget = equity * self.max_total_margin_pct - margin_used
                notional_per_leg = min(notional_per_leg, (remaining_budget * self.leverage) / 2)
                if notional_per_leg < 20:
                    return

                amount_a, amount_b = notional_per_leg / price_a, notional_per_leg / price_b
                side_a = "long" if direction == "long_a_short_b" else "short"
                side_b = "short" if direction == "long_a_short_b" else "long"

                # Execution-realism (2026-07-23, see project memory): both legs
                # get slippage-adjusted fills before sizing the wide backstop
                # stops off them.
                fill_side_a = "buy" if side_a == "long" else "sell"
                fill_side_b = "buy" if side_b == "long" else "sell"
                (fill_price_a, fill_amount_a, fill_info_a), (fill_price_b, fill_amount_b, fill_info_b) = await asyncio.gather(
                    simulate_fill(client, self.symbol_a, fill_side_a, price_a, amount_a),
                    simulate_fill(client, self.symbol_b, fill_side_b, price_b, amount_b),
                )

                # Wide backstop SL/TP per leg (15%) — the real exit is the z-score
                # check above, this only guards against a leg going to zero while
                # the other blows up (hedge failure / exchange-specific event).
                def _wide_stops(price, side):
                    return ((price * 0.85, price * 1.15) if side == "long" else (price * 1.15, price * 0.85))

                sl_a, tp_a = _wide_stops(fill_price_a, side_a)
                sl_b, tp_b = _wide_stops(fill_price_b, side_b)

                self.engine.open_position(self.symbol_a, side_a, fill_amount_a, fill_price_a, self.leverage,
                                           stop_loss=sl_a, take_profit=tp_a, mode="pairs")
                try:
                    self.engine.open_position(self.symbol_b, side_b, fill_amount_b, fill_price_b, self.leverage,
                                               stop_loss=sl_b, take_profit=tp_b, mode="pairs")
                except Exception as e:
                    # Leg B failed (e.g. insufficient margin) — don't leave leg A
                    # as a naked, unhedged directional bet.
                    exit_side_a = "sell" if side_a == "long" else "buy"
                    exit_price_a, _fill_amount, fill_info_a2 = await simulate_fill(client, self.symbol_a, exit_side_a, price_a, fill_amount_a)
                    filled_pct_a = fill_info_a2["filled_pct"] if fill_info_a2["simulated"] else 1.0
                    if filled_pct_a < 0.999:
                        self.engine.partial_close_position(self.symbol_a, exit_price_a, filled_pct_a, "hedge_leg_failed")
                        self._log("ERROR", f"Leg B fehlgeschlagen ({e}), Leg A nur {filled_pct_a:.0%} geschlossen (dünnes Orderbuch) "
                                           f"— ungehedgter Rest bleibt offen, sofort prüfen!")
                    else:
                        self.engine.close_position(self.symbol_a, exit_price_a, "hedge_leg_failed")
                        self._log("ERROR", f"Leg B fehlgeschlagen ({e}), Leg A wieder geschlossen — kein ungehedgtes Bein")
                    return

                self.open_pair = {"direction": direction, "opened_bar": self._bar_count, "entry_z": z}
                slip_a = f" slip{fill_info_a['slippage_pct']:.2%}" if fill_info_a["simulated"] and fill_info_a["slippage_pct"] > 0 else ""
                slip_b = f" slip{fill_info_b['slippage_pct']:.2%}" if fill_info_b["simulated"] and fill_info_b["slippage_pct"] > 0 else ""
                self._log("TRADE", f"OPEN PAIR {direction} | z={z:.2f} std={std:.5f} | vol×{vol_regime['continuous_risk_multiplier']:.2f} | "
                                    f"{self.symbol_a}={side_a}@{fill_price_a:.2f}{slip_a} {self.symbol_b}={side_b}@{fill_price_b:.2f}{slip_b}")
                get_journal().record("pairs_trading", f"{self.symbol_a}/{self.symbol_b}", "open_pair",
                                      f"log-Spread z-Score {z:.2f} ≥ Entry-Schwelle {self.z_entry} — "
                                      f"{direction.replace('_', ' ')}")

            except Exception as e:
                self._log("ERROR", f"Cycle error: {e}")

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
        self._log("INFO", f"PairsTradingHarvester started — {self.symbol_a}/{self.symbol_b} | "
                          f"window={self.window} z_entry={self.z_entry} z_exit={self.z_exit} "
                          f"z_stop={self.z_stop} | leverage={self.leverage}x")

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._log("INFO", "PairsTradingHarvester stopped")

    def status(self) -> dict:
        return {
            "running": self.running,
            "symbol_a": self.symbol_a,
            "symbol_b": self.symbol_b,
            "interval_seconds": self.interval,
            "window": self.window,
            "z_entry": self.z_entry,
            "z_exit": self.z_exit,
            "z_stop": self.z_stop,
            "max_hold_bars": self.max_hold_bars,
            "risk_pct": self.risk_pct,
            "leverage": self.leverage,
            "max_total_margin_pct": self.max_total_margin_pct,
            "cycle_count": self.cycle_count,
            "last_z": round(self.last_z, 3) if self.last_z is not None else None,
            "open_pair": self.open_pair,
            "engine": self.engine.status(self.live_prices),
        }

    def get_log(self, limit: int = 50) -> list:
        return list(reversed(self.log[-limit:]))
