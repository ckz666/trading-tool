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
"""
import math
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


class ThesisManager:
    """One per engine instance — tracks at most one open thesis per symbol,
    mirroring how these engines already track at most one open position per
    symbol."""

    def __init__(self):
        self._theses: dict[str, Thesis] = {}

    def open_thesis(self, thesis: Thesis):
        self._theses[thesis.symbol] = thesis

    def get(self, symbol: str) -> Optional[Thesis]:
        return self._theses.get(symbol)

    def close_thesis(self, symbol: str):
        self._theses.pop(symbol, None)

    def evaluate(self, symbol: str, context: dict) -> Optional[dict]:
        thesis = self._theses.get(symbol)
        if thesis is None:
            return None
        return thesis.evaluate(context)
