import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Callable

from trading.risk import RiskConfig


@dataclass
class BacktestResult:
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)
    initial_balance: float = 10000.0
    final_balance: float = 0.0

    @property
    def total_trades(self) -> int:
        return len([t for t in self.trades if t["side"] == "sell"])

    @property
    def win_rate(self) -> float:
        sells = [t for t in self.trades if t["side"] == "sell"]
        if not sells:
            return 0.0
        wins = [t for t in sells if t.get("pnl", 0) > 0]
        return len(wins) / len(sells)

    @property
    def total_return_pct(self) -> float:
        return (self.final_balance - self.initial_balance) / self.initial_balance * 100

    @property
    def max_drawdown_pct(self) -> float:
        if not self.equity_curve:
            return 0.0
        eq = pd.Series([e["equity"] for e in self.equity_curve])
        peak = eq.cummax()
        drawdown = (eq - peak) / peak
        return float(drawdown.min() * 100)

    @property
    def sharpe_ratio(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        eq = pd.Series([e["equity"] for e in self.equity_curve])
        returns = eq.pct_change().dropna()
        if returns.std() == 0:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(252))

    def summary(self) -> dict:
        return {
            "initial_balance": self.initial_balance,
            "final_balance": round(self.final_balance, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "total_trades": self.total_trades,
            "win_rate_pct": round(self.win_rate * 100, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
        }


def run_backtest(
    ohlcv: list,
    strategy_fn: Callable[[pd.DataFrame], pd.Series],
    initial_balance: float = 10000.0,
    fee_pct: float = 0.001,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
) -> BacktestResult:
    """Stop-loss/take-profit default to the same RiskConfig used by live/paper trading,
    so the backtest reflects the risk management the rest of the app actually applies."""
    if stop_loss_pct is None:
        stop_loss_pct = RiskConfig().default_stop_loss_pct
    if take_profit_pct is None:
        take_profit_pct = RiskConfig().default_take_profit_pct

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    signals = strategy_fn(df)  # 1=buy, -1=sell, 0=hold

    result = BacktestResult(initial_balance=initial_balance)
    balance = initial_balance
    position = 0.0
    entry_price = 0.0
    stop_loss_price = 0.0
    take_profit_price = 0.0

    for i, (ts, row) in enumerate(df.iterrows()):
        signal = signals.iloc[i] if i < len(signals) else 0
        price = row["close"]

        if position > 0 and row["low"] <= stop_loss_price:
            exit_price = stop_loss_price
            proceeds = position * exit_price * (1 - fee_pct)
            pnl = proceeds - (position * entry_price)
            result.trades.append({"ts": str(ts), "side": "sell", "price": exit_price, "amount": position, "pnl": pnl, "reason": "stop_loss"})
            balance = proceeds
            position = 0
            entry_price = 0

        elif position > 0 and row["high"] >= take_profit_price:
            exit_price = take_profit_price
            proceeds = position * exit_price * (1 - fee_pct)
            pnl = proceeds - (position * entry_price)
            result.trades.append({"ts": str(ts), "side": "sell", "price": exit_price, "amount": position, "pnl": pnl, "reason": "take_profit"})
            balance = proceeds
            position = 0
            entry_price = 0

        if signal == 1 and balance > 0 and position == 0:
            amount = (balance * (1 - fee_pct)) / price
            position = amount
            entry_price = price
            stop_loss_price = price * (1 - stop_loss_pct)
            take_profit_price = price * (1 + take_profit_pct)
            balance = 0
            result.trades.append({"ts": str(ts), "side": "buy", "price": price, "amount": amount})

        elif signal == -1 and position > 0:
            proceeds = position * price * (1 - fee_pct)
            pnl = proceeds - (position * entry_price)
            result.trades.append({"ts": str(ts), "side": "sell", "price": price, "amount": position, "pnl": pnl, "reason": "signal"})
            balance = proceeds
            position = 0
            entry_price = 0

        equity = balance + position * price
        result.equity_curve.append({"ts": str(ts), "equity": equity})

    # close open position at last price
    if position > 0:
        last_price = df["close"].iloc[-1]
        balance = position * last_price * (1 - fee_pct)
        position = 0

    result.final_balance = balance
    return result
