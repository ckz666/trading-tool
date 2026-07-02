import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable


@dataclass
class Alert:
    symbol: str
    condition: str  # "above" | "below" | "change_pct"
    value: float
    triggered: bool = False
    triggered_at: Optional[str] = None
    callback: Optional[Callable] = field(default=None, repr=False)


class AlertManager:
    def __init__(self):
        self._alerts: list[Alert] = []
        self._last_prices: dict[str, float] = {}
        self._log: list[dict] = []

    def add_alert(self, symbol: str, condition: str, value: float, callback: Optional[Callable] = None) -> Alert:
        alert = Alert(symbol=symbol, condition=condition, value=value, callback=callback)
        self._alerts.append(alert)
        return alert

    def remove_alert(self, index: int):
        if 0 <= index < len(self._alerts):
            self._alerts.pop(index)

    def check(self, symbol: str, price: float) -> list[Alert]:
        prev = self._last_prices.get(symbol)
        self._last_prices[symbol] = price
        triggered = []

        for alert in self._alerts:
            if alert.symbol != symbol or alert.triggered:
                continue

            hit = False
            if alert.condition == "above" and price >= alert.value:
                hit = True
            elif alert.condition == "below" and price <= alert.value:
                hit = True
            elif alert.condition == "change_pct" and prev:
                change = abs(price - prev) / prev * 100
                if change >= alert.value:
                    hit = True

            if hit:
                alert.triggered = True
                alert.triggered_at = datetime.now().isoformat()
                entry = {
                    "symbol": symbol,
                    "condition": alert.condition,
                    "value": alert.value,
                    "price": price,
                    "ts": alert.triggered_at,
                }
                self._log.append(entry)
                triggered.append(alert)
                if alert.callback:
                    asyncio.create_task(asyncio.coroutine(alert.callback)(entry))

        return triggered

    def get_alerts(self) -> list[dict]:
        return [
            {
                "symbol": a.symbol,
                "condition": a.condition,
                "value": a.value,
                "triggered": a.triggered,
                "triggered_at": a.triggered_at,
            }
            for a in self._alerts
        ]

    def get_log(self) -> list[dict]:
        return list(self._log)
