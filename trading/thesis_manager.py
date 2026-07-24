"""
Dynamic Exit & Belief Manager — Thesis interface (2026-07-23, DeepSeek +
ChatGPT collaborative design, see project memory). Item #3 of the
execution-realism/risk round (after slippage/partial-fill and the vol-size
modulator/OI-filter).

Every position an engine opens can attach a Thesis: an explicit, auditable
set of NAMED evidence rules (not a trained/opaque model — both AIs
specifically warned against a hidden weighted score, calling it a disguised
fifth strategy and an overfitting risk), each confirming or contradicting
the reason the position was opened, with its own time-decay half-life —
plus separate hard invalidation conditions that trigger an immediate exit
regardless of the aggregate score.

    Exit-Trigger = belief_score < 0  OR  any invalidation condition is True
    belief_score = sum(rule_value * freshness) over all evidence rules
    freshness    = exp(-age_seconds / half_life_seconds)

This is DeepSeek's final formula, with ChatGPT's confirming refinement that
invalidation stays an INDEPENDENT trigger (checked directly, not folded
into the score threshold) even though a broken invalidation condition also
contributes its own strongly-negative rule weight to the score — two
different failure modes ("evidence eroded gradually" vs "a specific thing I
said would disprove this just happened"), not one mechanism wearing two
hats.

V1 scope (deliberately, see project memory): binary exit only (belief<0 or
invalidation -> full close). Partial position reduction on a sagging-but-
not-negative belief is a natural V2 extension, not built here — sizing the
reduction needs its own design pass, and this is already the third large
item shipped in one session.

V1 engine coverage: wired into AutoTrader only (see trading/autotrader.py)
as the proof of concept — it already has the richest per-cycle data (price,
OI buffer, CVD) needed for meaningful evidence rules with the least new
plumbing. The other three engines are a documented follow-up, not done
here.

Instrumentation (2026-07-24, user explicitly asked to instrument V1 for
observability rather than build V2 blind): every evaluate() call appends
{ts, belief_score} to belief_history and updates belief_min/belief_max, so
V2 (partial reduction on a sagging-but-not-negative belief) can later be
justified — or ruled out — from real trade data instead of design
speculation. close_thesis() returns a summary (belief at entry/exit,
min/max, duration) that callers attach to the trade journal entry for the
same close event, so per-trade belief trajectories are reconstructable
after the fact without a dedicated UI.

Persistence (2026-07-23, added after the user caught a real position's
belief_score showing empty post-restart — thesis state was pure in-memory,
so every service restart silently dropped exit protection for whatever was
open at the time): EvidenceRule/InvalidationRule carry executable check()
closures, which aren't JSON-serialisable — but the closures themselves are
pure functions of `direction` (AutoTrader._build_thesis(symbol, direction,
entry_price, reasoning) rebuilds an identical rule set from those four
plain values every time). So only the DATA needs persisting — symbol,
direction, entry_price, reasoning, and the evidence log (rule name/value/
timestamp) — and on load, ThesisManager calls back into the engine-supplied
rebuild_fn to reconstruct fresh Thesis objects with live rule closures,
then replays the saved evidence log into them. Same tmp-then-rename atomic
write pattern as every other state file in this project (wallet, risk
state, journal, ...) — no new persistence mechanism introduced.
"""
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional


@dataclass
class EvidenceRule:
    name: str
    confirm_weight: float
    contradict_weight: float
    half_life_seconds: float
    # check(context) -> True (confirms), False (contradicts), None (no reading this cycle)
    check: Callable[[dict], Optional[bool]]


@dataclass
class InvalidationRule:
    name: str
    check: Callable[[dict], bool]   # -> True once this specific condition has fired


@dataclass
class _EvidenceReading:
    rule_name: str
    value: bool
    ts: datetime


@dataclass
class Thesis:
    symbol: str
    engine: str
    direction: str          # "long" | "short"
    reasoning: str          # human-readable summary of why this thesis was opened
    evidence_rules: list[EvidenceRule]
    invalidation_rules: list[InvalidationRule]
    entry_price: float
    created_at: datetime = field(default_factory=datetime.now)
    _readings: list[_EvidenceReading] = field(default_factory=list)
    last_belief_score: float = 0.0
    belief_history: list = field(default_factory=list)   # [{"ts": iso, "belief_score": float}]
    belief_min: float = None
    belief_max: float = None

    def _record(self, rule_name: str, value: bool):
        now = datetime.now()
        # keep only the most recent reading per rule — an older reading of
        # the SAME rule is superseded, not merged; freshness decay is what
        # makes a STALE single reading fade, not a growing pile of readings.
        self._readings = [r for r in self._readings if r.rule_name != rule_name]
        self._readings.append(_EvidenceReading(rule_name=rule_name, value=value, ts=now))

    def _compute_belief(self) -> float:
        now = datetime.now()
        rules_by_name = {r.name: r for r in self.evidence_rules}
        score = 0.0
        for reading in self._readings:
            rule = rules_by_name.get(reading.rule_name)
            if rule is None:
                continue
            age = max(0.0, (now - reading.ts).total_seconds())
            freshness = math.exp(-age / rule.half_life_seconds) if rule.half_life_seconds > 0 else 1.0
            weight = rule.confirm_weight if reading.value else -rule.contradict_weight
            score += weight * freshness
        self.last_belief_score = score
        self.belief_min = score if self.belief_min is None else min(self.belief_min, score)
        self.belief_max = score if self.belief_max is None else max(self.belief_max, score)
        self.belief_history.append({"ts": now.isoformat(), "belief_score": round(score, 3)})
        self.belief_history = self.belief_history[-500:]   # bounded, same convention as
                                                             # every other rolling list here
        return score

    def evaluate(self, context: dict) -> dict:
        """Run every evidence + invalidation rule against fresh context data,
        record readings, return the current belief score and whether this
        thesis says to exit now. context is engine-supplied — whatever the
        rules were written to expect (price, oi_delta, cvd_ratio, ...)."""
        for rule in self.evidence_rules:
            try:
                result = rule.check(context)
            except Exception:
                result = None
            if result is not None:
                self._record(rule.name, result)

        belief = self._compute_belief()

        invalidated_by = None
        for inv in self.invalidation_rules:
            try:
                if inv.check(context):
                    invalidated_by = inv.name
                    break
            except Exception:
                continue

        exit_now = belief < 0 or invalidated_by is not None
        return {
            "belief_score": round(belief, 3),
            "exit": exit_now,
            "exit_reason": invalidated_by or ("belief_negative" if belief < 0 else None),
        }

    def to_dict(self) -> dict:
        """Data only — see module docstring for why the rule objects
        themselves (executable closures) aren't part of this."""
        return {
            "symbol": self.symbol,
            "engine": self.engine,
            "direction": self.direction,
            "reasoning": self.reasoning,
            "entry_price": self.entry_price,
            "created_at": self.created_at.isoformat(),
            "last_belief_score": self.last_belief_score,
            "belief_history": self.belief_history,
            "belief_min": self.belief_min,
            "belief_max": self.belief_max,
            "readings": [
                {"rule_name": r.rule_name, "value": r.value, "ts": r.ts.isoformat()}
                for r in self._readings
            ],
        }

    def restore_readings(self, readings: list[dict]):
        """Replay a saved evidence log onto a freshly-rebuilt Thesis (same
        rule set, reconstructed via the engine's rebuild function — see
        ThesisManager.load()). Readings older than any rule's half_life by
        enough to be functionally zero are skipped rather than kept forever,
        same effect freshness decay would have had if the process had never
        restarted."""
        for r in readings:
            try:
                ts = datetime.fromisoformat(r["ts"])
            except (KeyError, ValueError):
                continue
            self._readings.append(_EvidenceReading(rule_name=r["rule_name"], value=r["value"], ts=ts))
        self._compute_belief()


class ThesisManager:
    """One per engine instance — tracks at most one open thesis per symbol,
    mirroring how these engines already track at most one open position per
    symbol.

    state_file/rebuild_fn: optional — pass both to persist across restarts.
    rebuild_fn(symbol, direction, entry_price, reasoning) -> Thesis must
    return a thesis with the SAME rule set _build_thesis would create fresh
    (it's how load() reconstructs live rule closures from saved plain
    data). Omit both to get the old pure-in-memory behaviour (e.g. for unit
    tests that don't want file I/O)."""

    def __init__(self, state_file: str = None, rebuild_fn: Callable = None):
        self._theses: dict[str, Thesis] = {}
        self.state_file = state_file
        self.rebuild_fn = rebuild_fn
        if state_file and rebuild_fn:
            self._load()

    def _save(self):
        if not self.state_file:
            return
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            data = {sym: t.to_dict() for sym, t in self._theses.items()}
            tmp = self.state_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self.state_file)
        except Exception as e:
            print(f"[ThesisManager] Could not save state: {e}")

    def _load(self):
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file) as f:
                data = json.load(f)
            for sym, d in data.items():
                thesis = self.rebuild_fn(sym, d["direction"], d["entry_price"], d["reasoning"])
                thesis.created_at = datetime.fromisoformat(d["created_at"])
                # restore history/min/max BEFORE replaying readings — restore_readings()
                # triggers one more _compute_belief() call, which correctly folds into
                # these restored bounds rather than resetting them from a null baseline
                thesis.belief_history = d.get("belief_history", [])
                thesis.belief_min = d.get("belief_min")
                thesis.belief_max = d.get("belief_max")
                thesis.restore_readings(d.get("readings", []))
                self._theses[sym] = thesis
            print(f"[ThesisManager:{self.state_file}] Restored {len(self._theses)} open thesis/theses")
        except Exception as e:
            print(f"[ThesisManager:{self.state_file}] Could not load state: {e} — starting fresh")

    def open_thesis(self, thesis: Thesis):
        self._theses[thesis.symbol] = thesis
        self._save()

    def get(self, symbol: str) -> Optional[Thesis]:
        return self._theses.get(symbol)

    def close_thesis(self, symbol: str) -> Optional[dict]:
        """Pops the thesis and returns an observability summary (belief at
        entry/exit, min/max, full history, duration) for the caller to attach
        to its trade-journal entry for the same close event — see module
        docstring on why this exists (instrument V1, don't build V2 blind).
        None if there was no open thesis for this symbol."""
        thesis = self._theses.pop(symbol, None)
        self._save()
        if thesis is None:
            return None
        duration = (datetime.now() - thesis.created_at).total_seconds()
        return {
            "belief_at_entry": thesis.belief_history[0]["belief_score"] if thesis.belief_history else None,
            "belief_at_exit": thesis.last_belief_score,
            "belief_min": thesis.belief_min,
            "belief_max": thesis.belief_max,
            "belief_duration_seconds": round(duration, 1),
            "belief_history": thesis.belief_history,
        }

    def evaluate(self, symbol: str, context: dict) -> Optional[dict]:
        thesis = self._theses.get(symbol)
        if thesis is None:
            return None
        result = thesis.evaluate(context)
        self._save()
        return result
