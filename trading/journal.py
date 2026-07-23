"""Shared trade journal across all engines (AutoTrader, Funding Harvest, Mean
Reversion, Pairs Trading) — one append-only, persisted log of every open/close
with a human-readable reason, so "why did it do that" is answerable without
digging through each engine's own rolling in-memory log (which caps at a few
hundred entries and resets on restart).

One process-wide singleton (get_journal()), one state file — mirrors the
SharedWallet pattern used across the engines.
"""
import json
import os
import uuid
from datetime import datetime

STATE_FILE = "data/trade_journal.json"
MAX_ENTRIES = 5000


class TradeJournal:
    def __init__(self, state_file: str = STATE_FILE):
        self.state_file = state_file
        self.entries: list[dict] = []
        self._load()

    def _load(self):
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file) as f:
                self.entries = json.load(f)
        except Exception as e:
            print(f"[TradeJournal] Could not load state: {e} — starting fresh")

    def _save(self):
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        tmp = self.state_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.entries[-MAX_ENTRIES:], f, indent=2)
        os.replace(tmp, self.state_file)

    def record(self, engine: str, symbol: str, action: str, reason: str,
               pnl: float = None, extra: dict = None) -> dict:
        """action: 'open_long'|'open_short'|'close_long'|'close_short'|'open_pair'|'close_pair'|...
        reason: free-text explanation (confluence criteria, RSI/ADX levels, z-score, funding rate, ...)"""
        entry = {
            "id": str(uuid.uuid4())[:8],
            "ts": datetime.now().isoformat(),
            "engine": engine,
            "symbol": symbol,
            "action": action,
            "reason": reason,
            "pnl": round(pnl, 4) if pnl is not None else None,
            **(extra or {}),
        }
        self.entries.append(entry)
        self.entries = self.entries[-MAX_ENTRIES:]
        self._save()
        return entry

    def recent(self, limit: int = 100, engine: str = None, symbol: str = None) -> list:
        rows = self.entries
        if engine:
            rows = [e for e in rows if e["engine"] == engine]
        if symbol:
            rows = [e for e in rows if e["symbol"] == symbol]
        return list(reversed(rows[-limit:]))


_journal: TradeJournal = None


def get_journal() -> TradeJournal:
    global _journal
    if _journal is None:
        _journal = TradeJournal()
    return _journal
