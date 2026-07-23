"""
Delta-neutral funding-rate harvesting: long spot + short perp on the same
symbol, same notional. Price moves roughly cancel between the two legs;
the P&L source is the funding payments a short perp receives when funding
is positive (longs paying shorts) — not a directional price call.

Paper-trading only, fully separate from the trend/scalp AutoTrader.
"""
import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional

from exchange.client import BitgetClient
from exchange.futures_client import FuturesClient
from trading.wallet import SharedWallet
from notifications.telegram import notify_fire_and_forget
from trading.journal import get_journal

SPOT_FEE = 0.001        # 0.1% taker, matches trading/paper.py
PERP_TAKER_FEE = 0.0006  # matches trading/futures_paper.py
MAINTENANCE_MARGIN = 0.005

STATE_FILE = "data/funding_harvest_state.json"


@dataclass
class HarvestPosition:
    symbol: str
    spot_qty: float
    perp_qty: float
    entry_spot_price: float
    entry_perp_price: float
    leverage: int
    margin: float
    liquidation_price: float
    funding_accrued: float = 0.0
    fees_paid: float = 0.0
    opened_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_funding_check: str = field(default_factory=lambda: datetime.now().isoformat())

    def basis_pct(self, spot_price: float, perp_price: float) -> float:
        return (perp_price - spot_price) / spot_price * 100

    def unrealized_price_pnl(self, spot_price: float, perp_price: float) -> float:
        """Delta-neutral by design: spot gain/loss should roughly offset the
        short perp's loss/gain. What's left over is basis drift, not direction."""
        spot_pnl = (spot_price - self.entry_spot_price) * self.spot_qty
        perp_pnl = (self.entry_perp_price - perp_price) * self.perp_qty
        return spot_pnl + perp_pnl

    def to_dict(self, spot_price: float = None, perp_price: float = None) -> dict:
        upnl = self.unrealized_price_pnl(spot_price, perp_price) if spot_price and perp_price else 0.0
        return {
            "symbol": self.symbol,
            "spot_qty": self.spot_qty,
            "perp_qty": self.perp_qty,
            "entry_spot_price": self.entry_spot_price,
            "entry_perp_price": self.entry_perp_price,
            "leverage": self.leverage,
            "margin": round(self.margin, 4),
            "liquidation_price": round(self.liquidation_price, 4),
            "funding_accrued": round(self.funding_accrued, 4),
            "fees_paid": round(self.fees_paid, 4),
            "unrealized_price_pnl": round(upnl, 4),
            "net_pnl": round(self.funding_accrued - self.fees_paid + upnl, 4),
            "basis_pct": round(self.basis_pct(spot_price, perp_price), 4) if spot_price and perp_price else 0.0,
            "opened_at": self.opened_at,
        }


class FundingHarvestEngine:
    """Paper P&L bookkeeping for paired spot+perp positions. No strategy logic
    here — that lives in FundingHarvester below."""

    def __init__(self, wallet: SharedWallet = None, initial_balance: float = 10000.0):
        self.wallet = wallet or SharedWallet(initial_balance)
        self.positions: dict[str, HarvestPosition] = {}
        self.trade_history: list[dict] = []
        self.total_funding_earned = 0.0
        self.total_fees_paid = 0.0
        self.equity_history: list[dict] = []   # [{ts, equity}], see trading/metrics.py
        self._load()

    def _save(self):
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        state = {
            "total_funding_earned": self.total_funding_earned,
            "total_fees_paid": self.total_fees_paid,
            "trade_history": self.trade_history,
            "equity_history": self.equity_history[-2000:],
            "positions": {sym: pos.to_dict() for sym, pos in self.positions.items()},
            "saved_at": datetime.now().isoformat(),
        }
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, STATE_FILE)
        self.wallet._save()

    def record_equity(self, prices: dict[str, dict]):
        """Mirrors FuturesPaperEngine.record_equity — see trading/metrics.py."""
        equity = self.portfolio_value(prices)
        self.equity_history.append({"ts": datetime.now().isoformat(), "equity": round(equity, 2)})
        if len(self.equity_history) > 2000:
            self.equity_history = self.equity_history[:1] + self.equity_history[1:-500:2] + self.equity_history[-500:]

    def _load(self):
        if not os.path.exists(STATE_FILE):
            return
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
            self.total_funding_earned = state.get("total_funding_earned", 0.0)
            self.total_fees_paid = state.get("total_fees_paid", 0.0)
            self.trade_history = state.get("trade_history", [])
            self.equity_history = state.get("equity_history", [])
            for sym, d in state.get("positions", {}).items():
                self.positions[sym] = HarvestPosition(
                    symbol=d["symbol"], spot_qty=d["spot_qty"], perp_qty=d["perp_qty"],
                    entry_spot_price=d["entry_spot_price"], entry_perp_price=d["entry_perp_price"],
                    leverage=d["leverage"], margin=d["margin"], liquidation_price=d["liquidation_price"],
                    funding_accrued=d.get("funding_accrued", 0.0), fees_paid=d.get("fees_paid", 0.0),
                    opened_at=d.get("opened_at", ""),
                )
            print(f"[FundingHarvest] Loaded state: wallet balance={self.wallet.balance:.2f} USDT, "
                  f"{len(self.positions)} open pos, {len(self.trade_history)} trades")
        except Exception as e:
            print(f"[FundingHarvest] Could not load state: {e} — starting fresh")

    def open_position(self, symbol: str, notional_usdt: float, spot_price: float,
                       perp_price: float, leverage: int = 2) -> dict:
        if symbol in self.positions:
            raise ValueError(f"Harvest position already open for {symbol}")

        spot_qty = notional_usdt / spot_price
        spot_fee = notional_usdt * SPOT_FEE
        perp_qty = notional_usdt / perp_price
        margin = notional_usdt / leverage
        perp_fee = notional_usdt * PERP_TAKER_FEE

        total_cost = notional_usdt + spot_fee + margin + perp_fee
        if self.wallet.balance < total_cost:
            raise ValueError(f"Insufficient balance: need {total_cost:.2f}, have {self.wallet.balance:.2f}")

        # short perp liquidation: price rises past this and the perp leg gets force-closed
        liq_price = perp_price * (1 + 1 / leverage - MAINTENANCE_MARGIN)

        self.wallet.balance -= total_cost
        fees = spot_fee + perp_fee
        self.total_fees_paid += fees

        pos = HarvestPosition(
            symbol=symbol, spot_qty=spot_qty, perp_qty=perp_qty,
            entry_spot_price=spot_price, entry_perp_price=perp_price,
            leverage=leverage, margin=margin, liquidation_price=liq_price,
            fees_paid=fees,
        )
        self.positions[symbol] = pos

        record = {
            "id": str(uuid.uuid4())[:8], "action": "open", "symbol": symbol,
            "notional_usdt": notional_usdt, "spot_price": spot_price, "perp_price": perp_price,
            "leverage": leverage, "fees": fees, "liq_price": liq_price,
            "ts": datetime.now().isoformat(),
        }
        self.trade_history.append(record)
        self._save()
        return record

    def accrue_funding(self, symbol: str, funding_rate: float, perp_price: float) -> float:
        """Called each time funding settles (~every 8h on Bitget). Short perp
        receives when funding_rate is positive. Realized immediately — funding
        is an actual cash settlement, not unrealized P&L."""
        pos = self.positions.get(symbol)
        if not pos:
            return 0.0
        notional = pos.perp_qty * perp_price
        payment = notional * funding_rate
        pos.funding_accrued += payment
        self.total_funding_earned += payment
        self.wallet.balance += payment
        pos.last_funding_check = datetime.now().isoformat()
        self._save()
        return payment

    def close_position(self, symbol: str, spot_price: float, perp_price: float, reason: str = "manual") -> dict:
        pos = self.positions.pop(symbol, None)
        if not pos:
            raise ValueError(f"No harvest position for {symbol}")

        spot_proceeds = pos.spot_qty * spot_price
        spot_fee = spot_proceeds * SPOT_FEE
        spot_pnl = spot_proceeds - spot_fee - (pos.spot_qty * pos.entry_spot_price)

        perp_pnl_gross = (pos.entry_perp_price - perp_price) * pos.perp_qty
        perp_fee = pos.perp_qty * perp_price * PERP_TAKER_FEE
        perp_pnl = perp_pnl_gross - perp_fee

        fees = spot_fee + perp_fee
        self.total_fees_paid += fees

        returned = pos.margin + perp_pnl + spot_proceeds - spot_fee
        self.wallet.balance += max(returned, 0)

        net_pnl = pos.funding_accrued + spot_pnl + perp_pnl

        record = {
            "id": str(uuid.uuid4())[:8], "action": "close", "reason": reason, "symbol": symbol,
            "spot_price": spot_price, "perp_price": perp_price,
            "funding_earned": round(pos.funding_accrued, 4),
            "spot_pnl": round(spot_pnl, 4), "perp_pnl": round(perp_pnl, 4),
            "fees": round(fees, 4), "net_pnl": round(net_pnl, 4),
            "ts": datetime.now().isoformat(),
        }
        self.trade_history.append(record)
        self._save()
        return record

    def is_liquidated(self, symbol: str, perp_price: float) -> bool:
        pos = self.positions.get(symbol)
        if not pos:
            return False
        return perp_price >= pos.liquidation_price

    def portfolio_value(self, prices: dict[str, dict]) -> float:
        """prices: {symbol: {"spot": px, "perp": px}}

        The spot leg's full current market value must be added back — opening
        a position deducts the entire notional from balance to buy real spot
        holdings, not just margin. unrealized_price_pnl() only returns the
        *change* since entry, so using margin + that delta alone silently
        drops the spot leg's principal from the total (looked like ~34%
        of the paper portfolio had vanished; it was just uncounted, not lost).
        Shared-wallet balance + this engine's own positions only — not the
        true account total when other engines also hold positions, see
        /api/portfolio/total for that."""
        total = self.wallet.balance
        for sym, pos in self.positions.items():
            p = prices.get(sym)
            if p:
                spot_value = pos.spot_qty * p["spot"]
                perp_pnl = (pos.entry_perp_price - p["perp"]) * pos.perp_qty
                total += spot_value + pos.margin + perp_pnl
        return max(total, 0)

    def status(self, prices: dict[str, dict] = None) -> dict:
        prices = prices or {}
        return {
            "balance": round(self.wallet.balance, 2),
            "portfolio_value": round(self.portfolio_value(prices), 2),
            "total_funding_earned": round(self.total_funding_earned, 2),
            "total_fees_paid": round(self.total_fees_paid, 2),
            "open_positions": len(self.positions),
            "positions": {
                sym: pos.to_dict(prices.get(sym, {}).get("spot"), prices.get(sym, {}).get("perp"))
                for sym, pos in self.positions.items()
            },
            "trade_count": len(self.trade_history),
        }


class FundingHarvester:
    """Scans candidate symbols' funding rates and manages entries/exits.
    No prediction, no ML — pure threshold logic on the observed rate."""

    def __init__(
        self,
        symbols: list[str] = None,
        engine: FundingHarvestEngine = None,
        interval_seconds: int = 900,       # scan every 15 min
        # Round-trip cost is 2x(SPOT_FEE + PERP_TAKER_FEE) = 0.32% of notional (open + close,
        # each leg both sides). At the old entry_rate_threshold (0.008%/8h) breakeven required
        # 0.32/0.008 = 40 settlements (~13 days) — but the old exit_rate_threshold (0.002%/8h)
        # let real rate noise trigger an exit after just 1 settlement almost every time, so
        # positions realized ~$0.20 in funding against $3-6 in fees on nearly every trade
        # (confirmed against 18 closed round-trips in prod: 31.19 USDT fees vs 0.59 USDT funding
        # earned in total). Fixed three ways together: (1) raise the entry bar so only rates
        # that can plausibly cover the round trip get taken, (2) widen the exit hysteresis so a
        # brief dip back toward zero doesn't immediately close a position, (3) enforce a minimum
        # number of settlements before the rate-exit is even evaluated, so fees get a chance to
        # amortize regardless of short-term rate noise. min_hold_settlements=13 at
        # entry_rate_threshold=0.025%/8h means the guaranteed-minimum hold alone (13 x 0.025%
        # = 0.325%) already clears the 0.32% round-trip cost even in the worst case where the
        # rate does nothing but sit right at the entry floor the whole time.
        entry_rate_threshold: float = 0.00025,   # ~27% APR — must plausibly clear the 0.32% round trip
        exit_rate_threshold: float = 0.0,        # only exit once the edge is genuinely gone, not just dipping
        min_hold_settlements: int = 13,          # ~4.3 days — floor lets fees amortize before any rate-exit
        max_basis_pct: float = 0.5,        # close if spot/perp diverge beyond this — hedge is failing
        max_position_pct: float = 0.25,    # cap per-symbol notional as a fraction of equity
        leverage: int = 2,
    ):
        self.symbols = symbols or ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
        self.engine = engine or FundingHarvestEngine()
        self.interval = interval_seconds
        self.entry_rate_threshold = entry_rate_threshold
        self.exit_rate_threshold = exit_rate_threshold
        self.min_hold_settlements = min_hold_settlements
        self.max_basis_pct = max_basis_pct
        self.max_position_pct = max_position_pct
        self.leverage = leverage

        self.running = False
        self._task: Optional[asyncio.Task] = None
        self.cycle_count = 0
        self.last_rates: dict[str, float] = {}
        self.log: list[dict] = []
        self._funding_settle_hours = {0, 8, 16}   # Bitget settles funding at 00:00/08:00/16:00 UTC
        self._last_settled_hour: dict[str, int] = {}
        self._settlement_count: dict[str, int] = {}

    def _log(self, level: str, msg: str, symbol: str = None):
        entry = {"ts": datetime.now().isoformat(), "level": level, "symbol": symbol or "ALL", "msg": msg}
        self.log.append(entry)
        self.log = self.log[-300:]
        tag = f"[{symbol}] " if symbol else ""
        print(f"[{entry['ts'][11:19]}] [FundingHarvest] [{level}] {tag}{msg}")
        if level == "TRADE":
            notify_fire_and_forget(f"💰 <b>Funding Harvest</b> {tag}\n{msg}")

    async def _cycle(self):
        self.cycle_count += 1
        prices_this_cycle: dict[str, dict] = {}
        async with BitgetClient() as spot, FuturesClient() as perp:
            for symbol in self.symbols:
                try:
                    spot_t, perp_t, fr = await asyncio.gather(
                        spot.fetch_ticker(symbol),
                        perp.fetch_ticker(symbol),
                        perp.fetch_funding_rate(symbol),
                    )
                    if fr is None:
                        # fetch_funding_rate() signals a failed fetch with None
                        # (2026-07-23 fix, audit finding H-05) — it used to return
                        # a fake {"rate": 0.0001}, indistinguishable from a real
                        # reading, which this engine would have silently traded
                        # decisions on. Skip the symbol this cycle instead of
                        # guessing; an existing position's SL/basis checks below
                        # still need a real rate too, so skip those as well.
                        self._log("WARN", "Funding-Rate-Fetch fehlgeschlagen — Zyklus übersprungen", symbol)
                        continue
                    spot_price, perp_price = spot_t["last"], perp_t["last"]
                    prices_this_cycle[symbol] = {"spot": spot_price, "perp": perp_price}
                    rate = fr["rate"]
                    self.last_rates[symbol] = rate
                    basis = abs(perp_price - spot_price) / spot_price * 100
                    has_position = symbol in self.engine.positions

                    if not has_position and rate >= self.entry_rate_threshold:
                        # size off free balance, not total equity — keeps sizing simple and
                        # avoids needing live prices for every other open position just to
                        # size this one entry
                        notional = min(self.engine.wallet.balance * self.max_position_pct,
                                        self.engine.wallet.balance * 0.9)
                        if notional > 50:  # dust guard
                            record = self.engine.open_position(symbol, notional, spot_price, perp_price, self.leverage)
                            self._settlement_count[symbol] = 0
                            self._log("TRADE", f"OPEN harvest ${notional:.0f} @ rate={rate*100:.4f}%/8h "
                                               f"(~{rate*3*365*100:.1f}% APR) | basis={basis:.3f}%", symbol)
                            get_journal().record("funding_harvest", symbol, "open",
                                f"Funding-Rate {rate*100:.4f}%/8h (~{rate*3*365*100:.1f}% APR) ≥ Schwelle "
                                f"{self.entry_rate_threshold*100:.4f}%, Basis {basis:.3f}%")

                    elif has_position:
                        # Liquidation and basis-blowout are genuine ongoing risks to the
                        # hedge itself — checked every poll, no reason to wait.
                        if self.engine.is_liquidated(symbol, perp_price):
                            record = self.engine.close_position(symbol, spot_price, perp_price, "liquidation")
                            self._settlement_count.pop(symbol, None)
                            self._log("ERROR", f"LIQUIDATED — closed @ net_pnl={record['net_pnl']:+.2f}", symbol)
                            get_journal().record("funding_harvest", symbol, "close", "Liquidation", pnl=record["net_pnl"])
                        elif basis >= self.max_basis_pct:
                            record = self.engine.close_position(symbol, spot_price, perp_price, "basis_risk")
                            self._settlement_count.pop(symbol, None)
                            self._log("WARN", f"Basis {basis:.3f}% >= cap, closed @ net_pnl={record['net_pnl']:+.2f}", symbol)
                            get_journal().record("funding_harvest", symbol, "close",
                                f"Basis {basis:.3f}% ≥ Cap {self.max_basis_pct:.2f}% — Hedge droht auseinanderzulaufen",
                                pnl=record["net_pnl"])
                        else:
                            # The exit-rate check used to run every poll (every 15 min),
                            # closing positions on short-lived rate noise before a single
                            # real funding settlement (only 3x/day) ever happened — a
                            # round-trip in fees for nothing. Now only reconsidered right
                            # at settlement, using the rate that was actually just paid.
                            now = datetime.utcnow()
                            just_settled = (now.hour in self._funding_settle_hours
                                            and self._last_settled_hour.get(symbol) != now.hour)
                            if just_settled:
                                payment = self.engine.accrue_funding(symbol, rate, perp_price)
                                self._last_settled_hour[symbol] = now.hour
                                settled = self._settlement_count.get(symbol, 0) + 1
                                self._settlement_count[symbol] = settled
                                self._log("INFO",
                                    f"Funding settled: {payment:+.4f} USDT (rate={rate*100:.4f}%) | "
                                    f"settlement {settled}/{self.min_hold_settlements}", symbol)
                                # Below min_hold_settlements: skip the rate-exit check entirely,
                                # even if the rate has already gone negative — the round-trip fee
                                # is sunk either way, so bailing out early only locks in the
                                # fee loss for certain instead of giving funding a chance to offset it.
                                if settled >= self.min_hold_settlements and rate < self.exit_rate_threshold:
                                    record = self.engine.close_position(symbol, spot_price, perp_price, "rate_dropped")
                                    self._settlement_count.pop(symbol, None)
                                    self._log("TRADE", f"CLOSE rate dropped to {rate*100:.4f}%/8h after "
                                                       f"{settled} settlements @ net_pnl={record['net_pnl']:+.2f}", symbol)
                                    get_journal().record("funding_harvest", symbol, "close",
                                        f"Rate auf {rate*100:.4f}%/8h gefallen (< Exit-Schwelle) nach "
                                        f"{settled} Settlements", pnl=record["net_pnl"])
                except Exception as e:
                    self._log("ERROR", f"Cycle error: {e}", symbol)
            if prices_this_cycle:
                self.engine.record_equity(prices_this_cycle)

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
        self._log("INFO", f"FundingHarvester started — {self.symbols} | every {self.interval}s | "
                          f"entry>={self.entry_rate_threshold*100:.4f}%/8h | leverage={self.leverage}x")

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._log("INFO", "FundingHarvester stopped")

    def status(self) -> dict:
        return {
            "running": self.running,
            "symbols": self.symbols,
            "interval_seconds": self.interval,
            "entry_rate_threshold": self.entry_rate_threshold,
            "exit_rate_threshold": self.exit_rate_threshold,
            "min_hold_settlements": self.min_hold_settlements,
            "settlement_count": self._settlement_count,
            "max_basis_pct": self.max_basis_pct,
            "leverage": self.leverage,
            "cycle_count": self.cycle_count,
            "last_rates": self.last_rates,
        }
