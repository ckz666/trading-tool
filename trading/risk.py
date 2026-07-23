from dataclasses import dataclass
from datetime import date
import json
import os

from notifications.telegram import notify_fire_and_forget


@dataclass
class RiskConfig:
    max_position_pct: float = 0.20      # max 20% of portfolio per trade
    max_daily_loss_pct: float = 0.05    # stop trading if daily loss > 5%
    max_drawdown_pct: float = 0.15      # pause if equity drops 15% from peak
    default_stop_loss_pct: float = 0.02
    default_take_profit_pct: float = 0.04
    min_confidence: float = 0.55        # ML confidence threshold


class RiskManager:
    def __init__(self, config: RiskConfig = None, state_file: str = None):
        # state_file: persists peak_equity/daily_pnl across restarts (audit finding
        # 2026-07-23, see project memory — the drawdown breaker was silently losing
        # its real peak on every service restart, measuring drawdown only from
        # whatever equity happened to be at boot instead of the true all-time high).
        # None (default) keeps the old ephemeral behaviour — used by ai/backtest.py
        # and ai/sweep.py, which deliberately want a fresh RiskManager per run.
        #
        # This class used to also track its own OpenPosition objects and expose
        # open_position()/check_sl_tp()/close_position()/update_position_pnl(),
        # but those had a long-only PnL bug (no `side` field — always computed
        # (exit-entry)*amount, wrong sign for shorts) AND were never actually
        # called anywhere (grep-verified, audit finding H-02, 2026-07-23, see
        # project memory) — real position tracking has lived in
        # trading/futures_paper.py::FuturesPaperEngine (correctly long/short-aware)
        # since before this file's methods were last touched. Removed rather than
        # fixed, since keeping a parallel, unused, wrong implementation around
        # only risks someone reaching for it later.
        self.config = config or RiskConfig()
        self.daily_pnl: float = 0.0   # realised-only, kept for display — NOT used for the loss check anymore, see check_daily_loss
        self.day_start_equity: float = 0.0   # true portfolio value at the start of today, includes unrealised
        self._day: date = date.today()
        self.blocked: bool = False
        self.block_reason: str = ""
        self.peak_equity: float = 0.0   # highest portfolio value ever seen
        self.state_file = state_file
        # Separate per-check transition trackers (Telegram-notification-only,
        # not persisted) — needed because check_daily_loss() and check_drawdown()
        # share self.blocked/block_reason but are called back-to-back every
        # cycle (check_daily_loss first, then check_drawdown — see autotrader.py
        # _run_symbol). If only one of the two is actually breached, the OTHER
        # check's "not breached" branch would otherwise clear self.blocked to
        # False for a moment before the breached one sets it back True a line
        # later — same-cycle, so the final state each cycle was always correct,
        # but naively notifying on every self.blocked transition inside each
        # function would spam an alert+recovery pair every single cycle. These
        # two flags track each check's OWN breach state independently so a
        # notification only fires on a genuine transition of that specific check.
        self._dd_blocked: bool = False
        self._daily_blocked: bool = False
        self._load_state()

    def _load_state(self):
        if not self.state_file or not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file) as f:
                state = json.load(f)
            self.peak_equity = state.get("peak_equity", 0.0)
            self.daily_pnl = state.get("daily_pnl", 0.0)
            self.day_start_equity = state.get("day_start_equity", 0.0)
            saved_day = state.get("day")
            if saved_day:
                self._day = date.fromisoformat(saved_day)
            print(f"[RiskManager:{self.state_file}] Loaded state: peak_equity=${self.peak_equity:,.2f} "
                  f"day_start_equity=${self.day_start_equity:,.2f}")
        except Exception as e:
            print(f"[RiskManager:{self.state_file}] Could not load state: {e} — starting fresh")

    def _save_state(self):
        if not self.state_file:
            return
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            tmp = self.state_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump({
                    "peak_equity": self.peak_equity,
                    "daily_pnl": self.daily_pnl,
                    "day_start_equity": self.day_start_equity,
                    "day": self._day.isoformat(),
                }, f, indent=2)
            os.replace(tmp, self.state_file)
        except Exception as e:
            print(f"[RiskManager:{self.state_file}] Could not save state: {e}")

    def _reset_daily_if_needed(self, portfolio_value: float = None):
        today = date.today()
        if today != self._day:
            self.daily_pnl = 0.0
            self._day = today
            self._daily_blocked = False
            if not self._dd_blocked:
                self.blocked = False
                self.block_reason = ""
            if portfolio_value is not None:
                self.day_start_equity = portfolio_value

    def check_drawdown(self, portfolio_value: float) -> bool:
        """Returns False (block new entries) if equity dropped >max_drawdown_pct from peak.
        Existing positions are still managed (SL/TP still fires), only new entries blocked.

        self.blocked/block_reason are re-evaluated fresh on every call (cleared
        here when the drawdown is no longer breached) rather than only ever being
        set — they used to latch on once triggered and never clear, so the status
        display could keep showing a stale "blocked" reason long after the
        portfolio had recovered, even though check_drawdown's return value itself
        (which actually gates new entries) was already correctly re-computed each
        time — a real display bug, not a trading-logic one, but confusing enough
        that the user noticed it looked wrong (2026-07-23, see project memory).
        """
        if portfolio_value > self.peak_equity:
            self.peak_equity = portfolio_value
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - portfolio_value) / self.peak_equity
            if drawdown >= self.config.max_drawdown_pct:
                self.blocked = True
                self.block_reason = (
                    f"Drawdown-Breaker: {drawdown:.1%} unter Peak "
                    f"(Peak ${self.peak_equity:,.0f} → aktuell ${portfolio_value:,.0f})"
                )
                if not self._dd_blocked:
                    notify_fire_and_forget(f"🚨 <b>Risk-Block ausgelöst</b>\n{self.block_reason}")
                self._dd_blocked = True
                self._save_state()
                return False
        if self._dd_blocked:
            notify_fire_and_forget(f"✅ <b>Drawdown-Block aufgehoben</b>\nWieder unter {self.config.max_drawdown_pct:.0%} Drawdown")
        self._dd_blocked = False
        # Only clear the combined display flag if the OTHER check isn't also
        # currently blocking — otherwise this would stomp on a daily-loss block
        # that's still active (see class docstring on _dd_blocked/_daily_blocked).
        if not self._daily_blocked:
            self.blocked = False
            self.block_reason = ""
        self._save_state()
        return True

    def check_daily_loss(self, portfolio_value: float) -> bool:
        """Computed from a true equity time series (day-start portfolio value vs.
        now — includes unrealised PnL) rather than only summing realised trade
        closes (audit finding H-03, 2026-07-23, see project memory). The old
        daily_pnl-only check couldn't see a large loss on a position that was
        still open — it only reacted once that position actually closed, by
        which point the loss had already happened. day_start_equity is seeded
        on the first-ever call and re-seeded at each day rollover.

        Same self.blocked/block_reason clear-on-recovery fix as check_drawdown
        above — see that docstring."""
        if self.day_start_equity <= 0:
            self.day_start_equity = portfolio_value
        self._reset_daily_if_needed(portfolio_value)
        if self.day_start_equity > 0:
            loss_pct = (self.day_start_equity - portfolio_value) / self.day_start_equity
            if loss_pct >= self.config.max_daily_loss_pct:
                self.blocked = True
                self.block_reason = (
                    f"Daily loss limit reached: {loss_pct*100:.1f}% "
                    f"(Tagesstart ${self.day_start_equity:,.0f} → aktuell ${portfolio_value:,.0f})"
                )
                if not self._daily_blocked:
                    notify_fire_and_forget(f"🚨 <b>Risk-Block ausgelöst</b>\n{self.block_reason}")
                self._daily_blocked = True
                self._save_state()
                return False
        if self._daily_blocked:
            notify_fire_and_forget(f"✅ <b>Daily-Loss-Block aufgehoben</b>\nWieder unter {self.config.max_daily_loss_pct:.0%} Tagesverlust")
        self._daily_blocked = False
        if not self._dd_blocked:
            self.blocked = False
            self.block_reason = ""
        self._save_state()
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

    def status(self) -> dict:
        return {
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "daily_pnl": round(self.daily_pnl, 2),
            "day_start_equity": round(self.day_start_equity, 2),
            "peak_equity": round(self.peak_equity, 2),
            "max_drawdown_pct": self.config.max_drawdown_pct,
            "max_position_pct": self.config.max_position_pct,
            "max_daily_loss_pct": self.config.max_daily_loss_pct,
        }
