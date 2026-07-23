import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from trading.wallet import SharedWallet

MAKER_FEE = 0.0002
TAKER_FEE = 0.0006
MAINTENANCE_MARGIN = 0.005  # 0.5%
FUNDING_RATE = 0.0001       # 0.01% per 8h (simplified)

STATE_FILE = "data/futures_paper_state.json"


@dataclass
class FuturesPosition:
    symbol: str
    side: str           # 'long' | 'short'
    amount: float
    entry_price: float
    leverage: int
    margin: float
    liquidation_price: float
    stop_loss: float
    take_profit: float
    opened_at: str = field(default_factory=lambda: datetime.now().isoformat())
    funding_paid: float = 0.0
    trailing_sl: bool = False
    trail_pct: float = 0.02    # trail distance as fraction of price
    sl_high_water: float = 0.0 # best price seen (for trailing)
    take_profit_1: float = 0.0 # partial TP at 50% of full TP distance
    partial_closed: bool = False  # True after TP1 was hit (50% already closed)
    mode: str = "trend"        # 'trend' (momentum/confluence) | 'scalp' (ranging mean-reversion)

    def update_trailing_sl(self, current_price: float) -> bool:
        """Move SL in profitable direction. Returns True if SL moved."""
        if not self.trailing_sl:
            return False
        if self.side == "long":
            new_sl = current_price * (1 - self.trail_pct)
            if new_sl > self.stop_loss:
                self.stop_loss = round(new_sl, 4)
                self.sl_high_water = max(self.sl_high_water, current_price)
                return True
        else:
            new_sl = current_price * (1 + self.trail_pct)
            if new_sl < self.stop_loss:
                self.stop_loss = round(new_sl, 4)
                self.sl_high_water = min(self.sl_high_water or current_price, current_price)
                return True
        return False

    def unrealized_pnl(self, current_price: float) -> float:
        if self.side == "long":
            return (current_price - self.entry_price) * self.amount
        else:
            return (self.entry_price - current_price) * self.amount

    def roe(self, current_price: float) -> float:
        pnl = self.unrealized_pnl(current_price)
        return pnl / self.margin * 100 if self.margin else 0

    def is_liquidated(self, current_price: float) -> bool:
        if self.side == "long":
            return current_price <= self.liquidation_price
        else:
            return current_price >= self.liquidation_price

    def to_dict(self, current_price: float = None) -> dict:
        upnl = self.unrealized_pnl(current_price) if current_price else 0
        # "unrealized_pnl"/"roe_pct" are pure price movement — a position shows
        # ~0 right after opening even though the entry fee (and, once accrued,
        # funding) is already a real, sunk cost. net_pnl/net_roe_pct answer "what
        # would I actually walk away with if I closed this right now": entry fee
        # (paid already) + an estimated exit fee at the current price (what
        # close_position() would actually charge) + funding paid so far.
        # See project memory, uPnL discrepancy report 2026-07-23.
        net_pnl = 0.0
        if current_price:
            entry_fee = self.amount * self.entry_price * TAKER_FEE
            est_exit_fee = self.amount * current_price * TAKER_FEE
            net_pnl = upnl - entry_fee - est_exit_fee - self.funding_paid
        return {
            "symbol": self.symbol,
            "side": self.side,
            "amount": self.amount,
            "entry_price": self.entry_price,
            "leverage": self.leverage,
            "margin": round(self.margin, 4),
            "liquidation_price": round(self.liquidation_price, 4),
            "stop_loss": round(self.stop_loss, 4),
            "take_profit": round(self.take_profit, 4),
            "take_profit_1": round(self.take_profit_1, 4),
            "partial_closed": self.partial_closed,
            "unrealized_pnl": round(upnl, 4),
            "roe_pct": round(self.roe(current_price) if current_price else 0, 2),
            "net_pnl": round(net_pnl, 4),
            "net_roe_pct": round(net_pnl / self.margin * 100, 2) if self.margin else 0,
            "opened_at": self.opened_at,
            "funding_paid": round(self.funding_paid, 4),
            "trailing_sl": self.trailing_sl,
            "trail_pct": self.trail_pct,
            "sl_high_water": round(self.sl_high_water, 4),
            "mode": self.mode,
        }


def calc_liquidation_price(entry: float, side: str, leverage: int,
                            maintenance_margin: float = MAINTENANCE_MARGIN) -> float:
    """Still a simplified isolated-margin approximation (ignores mark-price vs.
    last-price, fees, funding, cross-margin effects — audit finding H-01, 2026-07-23,
    see project memory), but maintenance_margin can now be the real per-symbol tier
    rate (exchange.futures_client.py::fetch_maintenance_margin_rate) instead of
    always the same guessed constant — that was the larger source of error for
    this bot's mixed BTC/altcoin symbol universe, tier-by-position-size barely
    matters at these position sizes (checked against real Bitget tier data: this
    bot's positions never leave tier 1 for any symbol)."""
    if side == "long":
        return entry * (1 - 1 / leverage + maintenance_margin)
    else:
        return entry * (1 + 1 / leverage - maintenance_margin)


class FuturesPaperEngine:
    def __init__(self, wallet: SharedWallet = None, initial_balance: float = 10000.0,
                 state_file: str = STATE_FILE):
        # state_file: lets a second, independent strategy (e.g. Mean Reversion) reuse
        # this same tested engine (position mgmt, SL/TP, fee accounting) on the same
        # shared wallet without colliding with AutoTrader's own state file — same
        # pattern FundingHarvestEngine/GridEngine already use for their own state.
        self.state_file = state_file
        self.wallet = wallet or SharedWallet(initial_balance)
        self.positions: dict[str, FuturesPosition] = {}
        self.trade_history: list[dict] = []
        self.total_pnl: float = 0.0
        self.equity_history: list[dict] = []   # [{ts, equity}]
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────
    # Balance lives in the shared wallet now, not here — this file only
    # persists this engine's own positions/history.

    def _save(self):
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        state = {
            "total_pnl": self.total_pnl,
            "trade_history": self.trade_history,
            "equity_history": self.equity_history[-2000:],
            "positions": {
                sym: pos.to_dict() for sym, pos in self.positions.items()
            },
            "saved_at": datetime.now().isoformat(),
        }
        tmp = self.state_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, self.state_file)   # atomic write
        self.wallet._save()

    def _load(self):
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file) as f:
                state = json.load(f)
            self.total_pnl       = state.get("total_pnl", 0.0)
            self.trade_history   = state.get("trade_history", [])
            self.equity_history  = state.get("equity_history", [])
            for sym, d in state.get("positions", {}).items():
                self.positions[sym] = FuturesPosition(
                    symbol           = d["symbol"],
                    side             = d["side"],
                    amount           = d["amount"],
                    entry_price      = d["entry_price"],
                    leverage         = d["leverage"],
                    margin           = d["margin"],
                    liquidation_price= d["liquidation_price"],
                    stop_loss        = d["stop_loss"],
                    take_profit      = d["take_profit"],
                    opened_at        = d.get("opened_at", ""),
                    funding_paid     = d.get("funding_paid", 0.0),
                    trailing_sl      = d.get("trailing_sl", False),
                    trail_pct        = d.get("trail_pct", 0.02),
                    sl_high_water    = d.get("sl_high_water", 0.0),
                    take_profit_1    = d.get("take_profit_1", 0.0),
                    partial_closed   = d.get("partial_closed", False),
                    mode             = d.get("mode", "trend"),
                )
            print(f"[FuturesPaper:{self.state_file}] Loaded state: wallet balance={self.wallet.balance:.2f} USDT, "
                  f"{len(self.positions)} open pos, {len(self.trade_history)} trades")
        except Exception as e:
            print(f"[FuturesPaper] Could not load state: {e} — starting fresh")

    def reset(self, initial_balance: float = None):
        """Wipe this engine's own state. Does NOT touch the shared wallet
        balance — call wallet.reset() separately if you want that too."""
        self.positions = {}
        self.trade_history = []
        self.total_pnl = 0.0
        self.equity_history = []
        if os.path.exists(self.state_file):
            os.remove(self.state_file)

    # ── Trading ───────────────────────────────────────────────────────────────

    def open_position(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        leverage: int,
        stop_loss: float,
        take_profit: float,
        trailing_sl: bool = False,
        trail_pct: float = 0.02,
        mode: str = "trend",
        maintenance_margin: float = MAINTENANCE_MARGIN,
    ) -> dict:
        """maintenance_margin: real per-symbol tier rate if the caller fetched one
        (exchange.futures_client.py::fetch_maintenance_margin_rate), else the old
        fixed default — see calc_liquidation_price docstring."""
        if symbol in self.positions:
            raise ValueError(f"Position already open for {symbol}. Close first.")

        notional = amount * price
        margin = notional / leverage
        fee = notional * TAKER_FEE

        if self.wallet.balance < margin + fee:
            raise ValueError(f"Insufficient margin: need {margin+fee:.2f} USDT, have {self.wallet.balance:.2f}")

        liq_price = calc_liquidation_price(price, side, leverage, maintenance_margin)
        self.wallet.balance -= (margin + fee)
        # Entry fee is a real, immediate cost (already deducted from wallet.balance
        # above) but close_position()/partial_close_position() only ever net the EXIT
        # fee into total_pnl — entry fees were silently never counted, making total_pnl
        # (and anything displaying it, e.g. the AutoTrader tab) systematically overstate
        # performance vs. the true wallet-balance-derived total ([[project-trading-roadmap]],
        # discrepancy report 2026-07-23). Recognise it here, symmetric with the exit fee.
        self.total_pnl -= fee

        # Partial TP1 at 50% of full TP distance from entry
        if side == "long":
            tp1 = price + (take_profit - price) * 0.5
        else:
            tp1 = price - (price - take_profit) * 0.5

        pos = FuturesPosition(
            symbol=symbol, side=side, amount=amount, entry_price=price,
            leverage=leverage, margin=margin, liquidation_price=liq_price,
            stop_loss=stop_loss, take_profit=take_profit,
            trailing_sl=trailing_sl, trail_pct=trail_pct,
            sl_high_water=price, take_profit_1=round(tp1, 4),
            mode=mode,
        )
        self.positions[symbol] = pos

        record = {
            "id": str(uuid.uuid4())[:8],
            "action": f"open_{side}",
            "symbol": symbol, "amount": amount, "price": price,
            "leverage": leverage, "margin": margin, "fee": fee,
            "liq_price": liq_price, "sl": stop_loss, "tp": take_profit,
            "mode": mode,
            "ts": datetime.now().isoformat(),
        }
        self.trade_history.append(record)
        self._save()
        return record

    def close_position(self, symbol: str, price: float, reason: str = "manual") -> dict:
        pos = self.positions.pop(symbol, None)
        if not pos:
            raise ValueError(f"No open position for {symbol}")

        pnl = pos.unrealized_pnl(price)
        fee = pos.amount * price * TAKER_FEE
        pnl_net = pnl - fee - pos.funding_paid
        returned_margin = pos.margin + pnl_net
        self.wallet.balance += max(returned_margin, 0)
        self.total_pnl += pnl_net

        record = {
            "id": str(uuid.uuid4())[:8],
            "action": f"close_{pos.side}",
            "reason": reason,
            "symbol": symbol, "amount": pos.amount,
            "entry_price": pos.entry_price, "exit_price": price,
            "leverage": pos.leverage, "pnl": round(pnl_net, 4),
            "roe_pct": round(pos.roe(price), 2),
            "mode": pos.mode,
            "ts": datetime.now().isoformat(),
        }
        self.trade_history.append(record)
        self._save()
        return record

    def apply_funding(self, symbol: str, current_price: float):
        pos = self.positions.get(symbol)
        if not pos:
            return
        funding = pos.amount * current_price * FUNDING_RATE
        if pos.side == "long":
            pos.funding_paid += funding
        else:
            pos.funding_paid -= funding

    def partial_close_position(self, symbol: str, price: float,
                               close_fraction: float = 0.5,
                               reason: str = "partial_tp") -> dict:
        """Close a fraction of a position without removing it from positions."""
        pos = self.positions.get(symbol)
        if not pos:
            raise ValueError(f"No open position for {symbol}")

        close_amount  = pos.amount * close_fraction
        remain_amount = pos.amount - close_amount

        pnl_per_unit  = (price - pos.entry_price) if pos.side == "long" else (pos.entry_price - price)
        pnl           = pnl_per_unit * close_amount
        fee           = close_amount * price * TAKER_FEE
        funding_share = pos.funding_paid * close_fraction   # proportional funding cost
        pnl_net       = pnl - fee - funding_share

        orig_margin = pos.margin
        returned = orig_margin * close_fraction + pnl_net
        self.wallet.balance += max(returned, 0)
        self.total_pnl += pnl_net

        pos.amount        = remain_amount
        pos.margin        = orig_margin * (1 - close_fraction)
        pos.funding_paid  -= funding_share   # deduct settled portion
        pos.partial_closed = True

        record = {
            "id":          str(uuid.uuid4())[:8],
            "action":      f"partial_close_{pos.side}",
            "reason":      reason,
            "symbol":      symbol,
            "amount":      round(close_amount, 8),
            "entry_price": pos.entry_price,
            "exit_price":  price,
            "leverage":    pos.leverage,
            "pnl":         round(pnl_net, 4),
            "roe_pct":     round(pnl_net / orig_margin * 100, 2) if orig_margin else 0,
            "mode":        pos.mode,
            "ts":          datetime.now().isoformat(),
        }
        self.trade_history.append(record)
        self._save()
        return record

    def check_sl_tp_liquidation(self, symbol: str, price: float) -> Optional[str]:
        pos = self.positions.get(symbol)
        if not pos:
            return None
        if pos.is_liquidated(price):
            return "liquidation"
        # update trailing SL before checking (SL may have moved)
        pos.update_trailing_sl(price)
        if pos.side == "long":
            if price <= pos.stop_loss:   return "stop_loss"
            # TP1: partial close if not yet done
            if not pos.partial_closed and pos.take_profit_1 > 0 and price >= pos.take_profit_1:
                return "take_profit_1"
            if price >= pos.take_profit: return "take_profit"
        else:
            if price >= pos.stop_loss:   return "stop_loss"
            if not pos.partial_closed and pos.take_profit_1 > 0 and price <= pos.take_profit_1:
                return "take_profit_1"
            if price <= pos.take_profit: return "take_profit"
        return None

    def record_equity(self, prices: dict):
        """Called periodically to snapshot the equity curve. Max 2000 points."""
        equity = self.portfolio_value(prices)
        self.equity_history.append({
            "ts": datetime.now().isoformat(),
            "equity": round(equity, 2),
        })
        if len(self.equity_history) > 2000:
            # downsample: keep first + every other old point + last 500
            self.equity_history = self.equity_history[:1] + \
                                   self.equity_history[1:-500:2] + \
                                   self.equity_history[-500:]

    def portfolio_value(self, prices: dict) -> float:
        """Shared-wallet balance + this engine's own open positions. Not the
        true account total when other engines also hold positions — see
        /api/portfolio/total for that."""
        total = self.wallet.balance
        for sym, pos in self.positions.items():
            if sym in prices:
                total += pos.margin + pos.unrealized_pnl(prices[sym])
        return max(total, 0)

    def get_position(self, symbol: str, current_price: float = None) -> Optional[dict]:
        pos = self.positions.get(symbol)
        if not pos:
            return None
        return pos.to_dict(current_price)

    def status(self, prices: dict = None) -> dict:
        prices = prices or {}
        unrealized = sum(pos.unrealized_pnl(prices[sym])
                          for sym, pos in self.positions.items() if sym in prices)
        return {
            "balance": round(self.wallet.balance, 2),
            "portfolio_value": round(self.portfolio_value(prices), 2),
            "total_pnl": round(self.total_pnl, 2),   # realised only (closed trades, all fees)
            # Realised + currently-open positions' unrealised PnL — matches what
            # /api/portfolio/total shows for this engine's own scope (that endpoint
            # additionally spans Grid/FundingHarvest via the shared wallet balance,
            # this one doesn't, so the two can still differ when those engines have
            # their own separate PnL — see project memory, discrepancy report 2026-07-23).
            "total_pnl_live": round(self.total_pnl + unrealized, 2),
            "open_positions": len(self.positions),
            "positions": {sym: pos.to_dict(prices.get(sym)) for sym, pos in self.positions.items()},
            "trade_count": len(self.trade_history),
        }
