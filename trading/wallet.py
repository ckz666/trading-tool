"""Single shared cash balance used by all three paper-trading engines
(FuturesPaperEngine/AutoTrader, FundingHarvestEngine, GridEngine) so opening
a position in any one of them draws down the same real pool of capital,
instead of each running its own independent 10,000 USDT account.
"""
import json
import os
from datetime import datetime

STATE_FILE = "data/shared_wallet_state.json"


class SharedWallet:
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self._load()

    def _save(self):
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        state = {
            "balance": self.balance,
            "initial_balance": self.initial_balance,
            "saved_at": datetime.now().isoformat(),
        }
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, STATE_FILE)  # atomic write

    def _load(self):
        if not os.path.exists(STATE_FILE):
            return
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
            self.balance = state.get("balance", self.initial_balance)
            self.initial_balance = state.get("initial_balance", self.initial_balance)
            print(f"[Wallet] Loaded shared balance: {self.balance:.2f} USDT")
        except Exception as e:
            print(f"[Wallet] Could not load state: {e} — starting fresh")

    def reset(self, initial_balance: float = 10000.0):
        """Wipe to a fresh balance. Does not touch any engine's positions/
        history — callers are expected to reset those separately."""
        self.balance = initial_balance
        self.initial_balance = initial_balance
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
        self._save()
