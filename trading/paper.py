import uuid
from datetime import datetime
from typing import Optional
import config


class PaperEngine:
    def __init__(self, initial_balance: float = None):
        self.balance = {"USDT": initial_balance or config.PAPER_BALANCE}
        self.positions: dict[str, float] = {}
        self.orders: list[dict] = []
        self.trade_history: list[dict] = []

    def get_balance(self) -> dict:
        return dict(self.balance)

    def get_positions(self) -> dict:
        return dict(self.positions)

    def place_order(self, symbol: str, side: str, amount: float, price: float, order_type: str = "market") -> dict:
        base, quote = symbol.split("/")
        cost = amount * price
        fee = cost * 0.001  # 0.1% fee

        if side == "buy":
            if self.balance.get(quote, 0) < cost + fee:
                raise ValueError(f"Insufficient {quote} balance: need {cost + fee:.2f}, have {self.balance.get(quote, 0):.2f}")
            self.balance[quote] = self.balance.get(quote, 0) - cost - fee
            self.positions[base] = self.positions.get(base, 0) + amount

        elif side == "sell":
            if self.positions.get(base, 0) < amount:
                raise ValueError(f"Insufficient {base} position: need {amount}, have {self.positions.get(base, 0):.8f}")
            self.positions[base] = self.positions.get(base, 0) - amount
            self.balance[quote] = self.balance.get(quote, 0) + cost - fee
            if self.positions[base] < 1e-8:
                del self.positions[base]

        order = {
            "id": str(uuid.uuid4())[:8],
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "price": price,
            "cost": cost,
            "fee": fee,
            "type": order_type,
            "status": "closed",
            "timestamp": datetime.now().isoformat(),
        }
        self.orders.append(order)
        self.trade_history.append(order)
        return order

    def portfolio_value(self, prices: dict[str, float]) -> float:
        total = self.balance.get("USDT", 0)
        for asset, qty in self.positions.items():
            symbol = f"{asset}/USDT"
            if symbol in prices:
                total += qty * prices[symbol]
        return total

    def pnl(self, prices: dict[str, float]) -> float:
        return self.portfolio_value(prices) - (config.PAPER_BALANCE)
