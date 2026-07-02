from dataclasses import dataclass, field
from datetime import datetime, date


@dataclass
class RiskConfig:
    max_position_pct: float = 0.20      # max 20% of portfolio per trade
    max_daily_loss_pct: float = 0.05    # stop trading if daily loss > 5%
    max_drawdown_pct: float = 0.15      # pause if equity drops 15% from peak
    default_stop_loss_pct: float = 0.02
    default_take_profit_pct: float = 0.04
    min_confidence: float = 0.55        # ML confidence threshold


@dataclass
class OpenPosition:
    symbol: str
    amount: float
    entry_price: float
    stop_loss: float
    take_profit: float
    opened_at: str = field(default_factory=lambda: datetime.now().isoformat())
    unrealized_pnl: float = 0.0


class RiskManager:
    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        self.positions: dict[str, OpenPosition] = {}
        self.daily_pnl: float = 0.0
        self._day: date = date.today()
        self.blocked: bool = False
        self.block_reason: str = ""
        self.peak_equity: float = 0.0   # highest portfolio value ever seen

    def _reset_daily_if_needed(self):
        today = date.today()
        if today != self._day:
            self.daily_pnl = 0.0
            self._day = today
            self.blocked = False
            self.block_reason = ""

    def check_drawdown(self, portfolio_value: float) -> bool:
        """Returns False (block new entries) if equity dropped >max_drawdown_pct from peak.
        Existing positions are still managed (SL/TP still fires), only new entries blocked.
        """
        if portfolio_value > self.peak_equity:
            self.peak_equity = portfolio_value
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - portfolio_value) / self.peak_equity
            if drawdown >= self.config.max_drawdown_pct:
                self.block_reason = (
                    f"Drawdown-Breaker: {drawdown:.1%} unter Peak "
                    f"(Peak ${self.peak_equity:,.0f} → aktuell ${portfolio_value:,.0f})"
                )
                return False
        return True

    def check_daily_loss(self, portfolio_value: float) -> bool:
        self._reset_daily_if_needed()
        if self.daily_pnl < 0:
            loss_pct = abs(self.daily_pnl) / portfolio_value
            if loss_pct >= self.config.max_daily_loss_pct:
                self.blocked = True
                self.block_reason = f"Daily loss limit reached: {loss_pct*100:.1f}%"
                return False
        return True

    def max_order_usdt(self, portfolio_value: float) -> float:
        return portfolio_value * self.config.max_position_pct

    def kelly_risk_pct(
        self,
        trade_pnls: list[float],
        fraction: float = 0.5,
        min_trades: int = 20,
        floor_pct: float = 0.005,
        cap_pct: float = 0.03,
    ) -> float | None:
        """Half-Kelly (default) fraction of equity to risk per trade, from realised trade PnLs.

        Returns None if there isn't enough history or no edge — caller should fall back
        to a fixed risk_pct in that case. Result is clamped to [floor_pct, cap_pct] since
        Kelly on a small/noisy sample can overshoot even at half-fraction.
        """
        if len(trade_pnls) < min_trades:
            return None
        wins = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p < 0]
        if not wins or not losses:
            return None
        win_rate = len(wins) / len(trade_pnls)
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))
        if avg_loss == 0:
            return None
        r = avg_win / avg_loss
        kelly = win_rate - (1 - win_rate) / r
        if kelly <= 0:
            return None
        return max(floor_pct, min(kelly * fraction, cap_pct))

    def calc_stop_loss(self, price: float, side: str, stop_pct: float) -> float:
        if side == "buy":
            return price * (1 - stop_pct)
        return price * (1 + stop_pct)

    def calc_take_profit(self, price: float, side: str, tp_pct: float) -> float:
        if side == "buy":
            return price * (1 + tp_pct)
        return price * (1 - tp_pct)

    def open_position(self, symbol: str, amount: float, price: float,
                       stop_loss: float, take_profit: float) -> OpenPosition:
        pos = OpenPosition(
            symbol=symbol,
            amount=amount,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        self.positions[symbol] = pos
        return pos

    def update_position_pnl(self, symbol: str, current_price: float):
        if symbol in self.positions:
            pos = self.positions[symbol]
            pos.unrealized_pnl = (current_price - pos.entry_price) * pos.amount

    def check_sl_tp(self, symbol: str, current_price: float) -> str | None:
        """Returns 'stop_loss' or 'take_profit' if triggered, else None."""
        pos = self.positions.get(symbol)
        if not pos:
            return None
        if current_price <= pos.stop_loss:
            return "stop_loss"
        if current_price >= pos.take_profit:
            return "take_profit"
        return None

    def close_position(self, symbol: str, exit_price: float) -> float:
        pos = self.positions.pop(symbol, None)
        if not pos:
            return 0.0
        pnl = (exit_price - pos.entry_price) * pos.amount
        self.daily_pnl += pnl
        return pnl

    def get_position(self, symbol: str) -> dict:
        pos = self.positions.get(symbol)
        if not pos:
            return {"amount": 0, "entry_price": 0, "unrealized_pnl": 0}
        return {
            "amount": pos.amount,
            "entry_price": pos.entry_price,
            "stop_loss": pos.stop_loss,
            "take_profit": pos.take_profit,
            "unrealized_pnl": pos.unrealized_pnl,
            "opened_at": pos.opened_at,
        }

    def status(self) -> dict:
        drawdown = (self.peak_equity - 0) / self.peak_equity if self.peak_equity > 0 else 0
        return {
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "daily_pnl": round(self.daily_pnl, 2),
            "open_positions": len(self.positions),
            "peak_equity": round(self.peak_equity, 2),
            "max_drawdown_pct": self.config.max_drawdown_pct,
            "max_position_pct": self.config.max_position_pct,
            "max_daily_loss_pct": self.config.max_daily_loss_pct,
        }
