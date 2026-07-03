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

    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions: dict[str, HarvestPosition] = {}
        self.trade_history: list[dict] = []
        self.total_funding_earned = 0.0
        self.total_fees_paid = 0.0
        self._load()

    def _save(self):
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        state = {
            "balance": self.balance,
            "initial_balance": self.initial_balance,
            "total_funding_earned": self.total_funding_earned,
            "total_fees_paid": self.total_fees_paid,
            "trade_history": self.trade_history,
            "positions": {sym: pos.to_dict() for sym, pos in self.positions.items()},
            "saved_at": datetime.now().isoformat(),
        }
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, STATE_FILE)

    def _load(self):
        if not os.path.exists(STATE_FILE):
            return
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
            self.balance = state.get("balance", self.initial_balance)
            self.initial_balance = state.get("initial_balance", self.initial_balance)
            self.total_funding_earned = state.get("total_funding_earned", 0.0)
            self.total_fees_paid = state.get("total_fees_paid", 0.0)
            self.trade_history = state.get("trade_history", [])
            for sym, d in state.get("positions", {}).items():
                self.positions[sym] = HarvestPosition(
                    symbol=d["symbol"], spot_qty=d["spot_qty"], perp_qty=d["perp_qty"],
                    entry_spot_price=d["entry_spot_price"], entry_perp_price=d["entry_perp_price"],
                    leverage=d["leverage"], margin=d["margin"], liquidation_price=d["liquidation_price"],
                    funding_accrued=d.get("funding_accrued", 0.0), fees_paid=d.get("fees_paid", 0.0),
                    opened_at=d.get("opened_at", ""),
                )
            print(f"[FundingHarvest] Loaded state: balance={self.balance:.2f} USDT, "
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
        if self.balance < total_cost:
            raise ValueError(f"Insufficient balance: need {total_cost:.2f}, have {self.balance:.2f}")

        # short perp liquidation: price rises past this and the perp leg gets force-closed
        liq_price = perp_price * (1 + 1 / leverage - MAINTENANCE_MARGIN)

        self.balance -= total_cost
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
        self.balance += payment
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
        self.balance += max(returned, 0)

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
        """prices: {symbol: {"spot": px, "perp": px}}"""
        total = self.balance
        for sym, pos in self.positions.items():
            p = prices.get(sym)
            if p:
                total += pos.margin + pos.unrealized_price_pnl(p["spot"], p["perp"])
        return max(total, 0)

    def status(self, prices: dict[str, dict] = None) -> dict:
        prices = prices or {}
        return {
            "balance": round(self.balance, 2),
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
        entry_rate_threshold: float = 0.00008,   # ~8.8% APR, covers round-trip fees with margin
        exit_rate_threshold: float = 0.00002,    # hysteresis: exit well below entry, avoid flip-flop
        max_basis_pct: float = 0.5,        # close if spot/perp diverge beyond this — hedge is failing
        max_position_pct: float = 0.25,    # cap per-symbol notional as a fraction of equity
        leverage: int = 2,
    ):
        self.symbols = symbols or ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
        self.engine = engine or FundingHarvestEngine()
        self.interval = interval_seconds
        self.entry_rate_threshold = entry_rate_threshold
        self.exit_rate_threshold = exit_rate_threshold
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

    def _log(self, level: str, msg: str, symbol: str = None):
        entry = {"ts": datetime.now().isoformat(), "level": level, "symbol": symbol or "ALL", "msg": msg}
        self.log.append(entry)
        self.log = self.log[-300:]
        tag = f"[{symbol}] " if symbol else ""
        print(f"[{entry['ts'][11:19]}] [FundingHarvest] [{level}] {tag}{msg}")

    async def _cycle(self):
        self.cycle_count += 1
        async with BitgetClient() as spot, FuturesClient() as perp:
            for symbol in self.symbols:
                try:
                    spot_t, perp_t, fr = await asyncio.gather(
                        spot.fetch_ticker(symbol),
                        perp.fetch_ticker(symbol),
                        perp.fetch_funding_rate(symbol),
                    )
                    spot_price, perp_price = spot_t["last"], perp_t["last"]
                    rate = fr["rate"]
                    self.last_rates[symbol] = rate
                    basis = abs(perp_price - spot_price) / spot_price * 100
                    has_position = symbol in self.engine.positions

                    if not has_position and rate >= self.entry_rate_threshold:
                        # size off free balance, not total equity — keeps sizing simple and
                        # avoids needing live prices for every other open position just to
                        # size this one entry
                        notional = min(self.engine.balance * self.max_position_pct,
                                        self.engine.balance * 0.9)
                        if notional > 50:  # dust guard
                            record = self.engine.open_position(symbol, notional, spot_price, perp_price, self.leverage)
                            self._log("TRADE", f"OPEN harvest ${notional:.0f} @ rate={rate*100:.4f}%/8h "
                                               f"(~{rate*3*365*100:.1f}% APR) | basis={basis:.3f}%", symbol)

                    elif has_position:
                        if self.engine.is_liquidated(symbol, perp_price):
                            record = self.engine.close_position(symbol, spot_price, perp_price, "liquidation")
                            self._log("ERROR", f"LIQUIDATED — closed @ net_pnl={record['net_pnl']:+.2f}", symbol)
                        elif basis >= self.max_basis_pct:
                            record = self.engine.close_position(symbol, spot_price, perp_price, "basis_risk")
                            self._log("WARN", f"Basis {basis:.3f}% >= cap, closed @ net_pnl={record['net_pnl']:+.2f}", symbol)
                        elif rate < self.exit_rate_threshold:
                            record = self.engine.close_position(symbol, spot_price, perp_price, "rate_dropped")
                            self._log("TRADE", f"CLOSE rate dropped to {rate*100:.4f}%/8h @ net_pnl={record['net_pnl']:+.2f}", symbol)
                        else:
                            # settle funding once per settlement window, not every 15-min poll
                            now = datetime.utcnow()
                            if now.hour in self._funding_settle_hours and self._last_settled_hour.get(symbol) != now.hour:
                                payment = self.engine.accrue_funding(symbol, rate, perp_price)
                                self._last_settled_hour[symbol] = now.hour
                                self._log("INFO", f"Funding settled: {payment:+.4f} USDT (rate={rate*100:.4f}%)", symbol)
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
            "max_basis_pct": self.max_basis_pct,
            "leverage": self.leverage,
            "cycle_count": self.cycle_count,
            "last_rates": self.last_rates,
        }
