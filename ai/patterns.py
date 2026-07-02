import pandas as pd
import numpy as np


def detect_patterns(df: pd.DataFrame) -> dict:
    """Detect candlestick patterns on the last 1–5 candles."""
    if len(df) < 5:
        return {}

    c  = df.iloc[-1]   # current
    p  = df.iloc[-2]   # previous
    pp = df.iloc[-3]   # 2 back
    p3 = df.iloc[-4]   # 3 back
    p4 = df.iloc[-5]   # 4 back

    def _metrics(candle):
        body       = abs(candle["close"] - candle["open"])
        up_wick    = candle["high"] - max(candle["close"], candle["open"])
        dn_wick    = min(candle["close"], candle["open"]) - candle["low"]
        total      = candle["high"] - candle["low"]
        bullish    = candle["close"] > candle["open"]
        return body, up_wick, dn_wick, total, bullish

    body,  uw,  dw,  tr,  bull  = _metrics(c)
    pbody, puw, pdw, ptr, pbull = _metrics(p)

    # avoid division by zero
    tr_safe  = max(tr,  1e-10)
    ptr_safe = max(ptr, 1e-10)

    avg_range = (df["high"] - df["low"]).tail(10).mean()

    patterns: dict[str, str] = {}   # name → "bullish" | "bearish" | "neutral"

    # ── Single-candle patterns ────────────────────────────────────────────────

    # Doji: body < 10% of range
    if body / tr_safe < 0.10:
        if dw > 2 * uw:
            patterns["dragonfly_doji"] = "bullish"     # long lower wick → buyers took over
        elif uw > 2 * dw:
            patterns["gravestone_doji"] = "bearish"    # long upper wick → sellers took over
        elif uw > 0.35 * tr_safe and dw > 0.35 * tr_safe:
            if abs(uw - dw) < 0.15 * tr_safe:
                patterns["rickshaw_man"] = "neutral"       # long, near-symmetric wicks
            else:
                patterns["long_legged_doji"] = "neutral"   # long wicks on both sides
        else:
            patterns["doji"] = "neutral"

    # Marubozu: almost no wicks — strong momentum candle
    if body / tr_safe > 0.90:
        patterns["marubozu"] = "bullish" if bull else "bearish"

    # Spinning Top: small body, roughly equal wicks — indecision
    if 0.10 < body / tr_safe < 0.30 and abs(uw - dw) < body:
        patterns["spinning_top"] = "neutral"

    # High Wave: tiny body, very long wicks both sides, unusually wide range for the context
    if body / tr_safe < 0.15 and uw > 2 * body and dw > 2 * body and tr > avg_range * 1.5:
        patterns["high_wave"] = "neutral"

    # Hammer: long lower wick, small body at top of range (at support → bullish)
    if dw > 2 * body and uw < body * 0.5 and body / tr_safe < 0.35:
        patterns["hammer"] = "bullish"

    # Inverted Hammer: long upper wick, small body at bottom → bullish (buyers tried)
    if uw > 2 * body and dw < body * 0.5 and body / tr_safe < 0.35:
        patterns["inverted_hammer"] = "bullish"

    # Hanging Man: same shape as hammer but at top → bearish warning
    # (context-blind here; we check if prev trend was up)
    prev_trend_up   = df["close"].tail(10).iloc[-1] > df["close"].tail(10).iloc[0]
    prev_trend_down = not prev_trend_up
    if dw > 2 * body and uw < body * 0.5 and prev_trend_up:
        patterns["hanging_man"] = "bearish"

    # Shooting Star: long upper wick, small body at top → bearish
    if uw > 2 * body and dw < body * 0.5 and body / tr_safe < 0.35:
        patterns["shooting_star"] = "bearish"

    # Bullish Belt Hold: opens at/near the low, strong body, no lower wick, after a downtrend
    if bull and dw < body * 0.05 and body / tr_safe > 0.6 and prev_trend_down:
        patterns["bullish_belt_hold"] = "bullish"

    # Bearish Belt Hold: opens at/near the high, strong body, no upper wick, after an uptrend
    if not bull and uw < body * 0.05 and body / tr_safe > 0.6 and prev_trend_up:
        patterns["bearish_belt_hold"] = "bearish"

    # ── Two-candle patterns ───────────────────────────────────────────────────

    # Bullish Engulfing
    if (not pbull and bull
            and c["open"] <= p["close"] and c["close"] >= p["open"]
            and body > pbody):
        patterns["bullish_engulfing"] = "bullish"

    # Bearish Engulfing
    if (pbull and not bull
            and c["open"] >= p["close"] and c["close"] <= p["open"]
            and body > pbody):
        patterns["bearish_engulfing"] = "bearish"

    # Piercing Line: bearish candle followed by bullish that closes above midpoint
    if (not pbull and bull
            and c["open"] < p["close"]
            and c["close"] > (p["open"] + p["close"]) / 2
            and c["close"] < p["open"]):
        patterns["piercing_line"] = "bullish"

    # Dark Cloud Cover: bullish candle followed by bearish that closes below midpoint
    if (pbull and not bull
            and c["open"] > p["close"]
            and c["close"] < (p["open"] + p["close"]) / 2
            and c["close"] > p["open"]):
        patterns["dark_cloud_cover"] = "bearish"

    # Bullish Harami: large bearish candle, then small candle inside it
    if (not pbull and bull
            and c["open"] > p["close"] and c["close"] < p["open"]
            and body < pbody * 0.6):
        patterns["bullish_harami"] = "bullish"

    # Bearish Harami: large bullish candle, then small candle inside it
    if (pbull and not bull
            and c["open"] < p["close"] and c["close"] > p["open"]
            and body < pbody * 0.6):
        patterns["bearish_harami"] = "bearish"

    # Bullish Harami Cross: harami where the inside candle is a doji (direction-agnostic)
    if (not pbull and c["open"] > p["close"] and c["close"] < p["open"]
            and body / tr_safe < 0.10):
        patterns["bullish_harami_cross"] = "bullish"

    # Bearish Harami Cross
    if (pbull and c["open"] < p["close"] and c["close"] > p["open"]
            and body / tr_safe < 0.10):
        patterns["bearish_harami_cross"] = "bearish"

    # Homing Pigeon: small bearish candle contained within a prior larger bearish candle
    if (not pbull and not bull
            and c["open"] > p["close"] and c["close"] < p["open"]
            and body < pbody * 0.6):
        patterns["homing_pigeon"] = "bullish"

    # Tweezer Bottom: two candles with near-identical lows → support
    if (not pbull and bull
            and abs(c["low"] - p["low"]) / max(p["low"], 1e-10) < 0.002):
        patterns["tweezer_bottom"] = "bullish"

    # Tweezer Top: two candles with near-identical highs → resistance
    if (pbull and not bull
            and abs(c["high"] - p["high"]) / max(p["high"], 1e-10) < 0.002):
        patterns["tweezer_top"] = "bearish"

    # Bullish Kicker: bearish marubozu, then a bullish marubozu that gaps above its high
    if (not pbull and bull
            and pbody / ptr_safe > 0.7 and body / tr_safe > 0.7
            and c["low"] > p["high"]):
        patterns["bullish_kicker"] = "bullish"

    # Bearish Kicker: bullish marubozu, then a bearish marubozu that gaps below its low
    if (pbull and not bull
            and pbody / ptr_safe > 0.7 and body / tr_safe > 0.7
            and c["high"] < p["low"]):
        patterns["bearish_kicker"] = "bearish"

    # Bullish Counterattack: gap-down open that rallies back to match the prior close
    if (not pbull and bull
            and c["open"] < p["close"]
            and abs(c["close"] - p["close"]) / max(p["close"], 1e-10) < 0.002):
        patterns["bullish_counterattack"] = "bullish"

    # Bearish Counterattack: gap-up open that sells back down to match the prior close
    if (pbull and not bull
            and c["open"] > p["close"]
            and abs(c["close"] - p["close"]) / max(p["close"], 1e-10) < 0.002):
        patterns["bearish_counterattack"] = "bearish"

    # On-Neck / In-Neck / Thrusting: bearish continuation after a gap-down open that only
    # partially recovers into the prior candle's body
    if not pbull and bull and c["open"] < p["low"]:
        midpoint = (p["open"] + p["close"]) / 2
        if c["close"] <= p["low"] * 1.002:
            patterns["on_neck"] = "bearish"
        elif c["close"] <= p["close"] * 1.005:
            patterns["in_neck"] = "bearish"
        elif c["close"] < midpoint:
            patterns["thrusting"] = "bearish"

    # ── Three-candle patterns ─────────────────────────────────────────────────

    ppbody, _, _, _, ppbull = _metrics(pp)

    # Morning Star: bearish big → small doji/star → bullish big
    if (not ppbull
            and abs(p["close"] - p["open"]) < ppbody * 0.3
            and bull and c["close"] > (pp["open"] + pp["close"]) / 2):
        patterns["morning_star"] = "bullish"

    # Evening Star: bullish big → small doji/star → bearish big
    if (ppbull
            and abs(p["close"] - p["open"]) < ppbody * 0.3
            and not bull and c["close"] < (pp["open"] + pp["close"]) / 2):
        patterns["evening_star"] = "bearish"

    # Three White Soldiers: three consecutive bullish candles, each closing higher
    if (all(df.iloc[-i]["close"] > df.iloc[-i]["open"] for i in [1, 2, 3])
            and df.iloc[-1]["close"] > df.iloc[-2]["close"] > df.iloc[-3]["close"]):
        patterns["three_white_soldiers"] = "bullish"

    # Three Black Crows: three consecutive bearish candles, each closing lower
    if (all(df.iloc[-i]["close"] < df.iloc[-i]["open"] for i in [1, 2, 3])
            and df.iloc[-1]["close"] < df.iloc[-2]["close"] < df.iloc[-3]["close"]):
        patterns["three_black_crows"] = "bearish"

    # Three Inside Up: harami + bullish confirmation
    if (not ppbull and                                   # candle -3 bearish
            p["open"] > pp["close"] and p["close"] < pp["open"] and  # harami
            bull and c["close"] > p["close"]):           # bullish confirm
        patterns["three_inside_up"] = "bullish"

    # Three Inside Down: harami + bearish confirmation
    if (ppbull and
            p["open"] < pp["close"] and p["close"] > pp["open"] and
            not bull and c["close"] < p["close"]):
        patterns["three_inside_down"] = "bearish"

    # Identical Three Crows: three bearish candles, each opening ~at the prior close
    if (not bull and not pbull and not ppbull
            and c["close"] < p["close"] < pp["close"]
            and abs(c["open"] - p["close"]) / max(p["close"], 1e-10) < 0.002
            and abs(p["open"] - pp["close"]) / max(pp["close"], 1e-10) < 0.002):
        patterns["identical_three_crows"] = "bearish"

    # Stick Sandwich: bearish–bullish–bearish, 1st/3rd closing at nearly the same price (support)
    if (not ppbull and pbull and not bull
            and p["close"] > pp["close"]
            and abs(c["close"] - pp["close"]) / max(pp["close"], 1e-10) < 0.002):
        patterns["stick_sandwich"] = "bullish"

    # Tri-Star: three consecutive dojis, middle one gapped away from the outer two
    pp_range = max(pp["high"] - pp["low"], 1e-10)
    if (body / tr_safe < 0.10 and pbody / ptr_safe < 0.10 and ppbody / pp_range < 0.10):
        if p["high"] < min(pp["low"], c["low"]):
            patterns["tri_star"] = "bullish"
        elif p["low"] > max(pp["high"], c["high"]):
            patterns["tri_star"] = "bearish"

    # Unique Three River: long bearish, harami-like bearish with a new low, small bullish close-under
    if (not ppbull and not pbull
            and p["open"] < pp["open"] and p["close"] > pp["close"]
            and p["low"] < pp["low"]
            and bull and body < pbody and c["close"] < p["close"]):
        patterns["unique_three_river"] = "bullish"

    # Abandoned Baby: doji that gaps away from both neighbors — full reversal
    if (not ppbull and pbody / ptr_safe < 0.10
            and p["high"] < pp["low"]
            and bull and c["low"] > p["high"]
            and c["close"] > (pp["open"] + pp["close"]) / 2):
        patterns["bullish_abandoned_baby"] = "bullish"

    if (ppbull and pbody / ptr_safe < 0.10
            and p["low"] > pp["high"]
            and not bull and c["high"] < p["low"]
            and c["close"] < (pp["open"] + pp["close"]) / 2):
        patterns["bearish_abandoned_baby"] = "bearish"

    # Tasuki Gap: gap continuation where the gap isn't fully closed by the third candle
    if (ppbull and pbull and p["low"] > pp["high"]
            and not bull and c["open"] < p["close"] and c["close"] > pp["high"]):
        patterns["upside_tasuki_gap"] = "bullish"

    if (not ppbull and not pbull and p["high"] < pp["low"]
            and bull and c["open"] > p["close"] and c["close"] < pp["low"]):
        patterns["downside_tasuki_gap"] = "bearish"

    # Two Crows: bullish candle, gap-up bearish candle, third closes back inside the first's body
    if (ppbull and not pbull and p["open"] > pp["close"]
            and not bull and c["open"] < p["open"] and c["open"] > p["close"]
            and pp["open"] < c["close"] < pp["close"]):
        patterns["two_crows"] = "bearish"

    # Upside Gap Two Crows: true gap up, third candle engulfs the second but the gap holds
    if (ppbull and not pbull and p["low"] > pp["close"]
            and not bull and c["open"] > p["open"] and c["close"] < p["close"]
            and c["close"] > pp["close"]):
        patterns["upside_gap_two_crows"] = "bearish"

    # ── Multi-candle continuation patterns ────────────────────────────────────

    p3body, _, _, _, p3bull = _metrics(p3)
    p4body, _, _, _, p4bull = _metrics(p4)

    # Rising Three Methods: long bull, 3 small bears inside, long bull (5-candle)
    if (p4bull and p4body > p3body * 2
            and all(df.iloc[-i]["close"] < df.iloc[-i]["open"] for i in [2, 3, 4])
            and all(df.iloc[-i]["close"] > p4["close"] and
                    df.iloc[-i]["open"]  < p4["open"]  for i in [2, 3, 4])
            and bull and c["close"] > p4["close"]):
        patterns["rising_three_methods"] = "bullish"

    # Falling Three Methods: long bear, 3 small bulls inside, long bear (5-candle)
    if (not p4bull and p4body > p3body * 2
            and all(df.iloc[-i]["close"] > df.iloc[-i]["open"] for i in [2, 3, 4])
            and not bull and c["close"] < p4["close"]):
        patterns["falling_three_methods"] = "bearish"

    # Mat Hold: long bull, gap up, 3 small candles holding above the first candle's low,
    # then a breakout close beyond the most recent high
    if (p4bull and p3["low"] > p4["close"]
            and all(df.iloc[-i]["low"] > p4["low"] for i in [2, 3, 4])
            and bull and c["close"] > df.iloc[-2]["high"]):
        patterns["mat_hold"] = "bullish"

    # Ladder Bottom: three falling bears, a small-bodied bear with a long upper wick, then reversal
    if (not p4bull and not p3bull and not ppbull
            and p4["close"] > p3["close"] > pp["close"]
            and not pbull and puw > 2 * pbody
            and bull and c["open"] > p["high"] and c["close"] > p["open"]):
        patterns["ladder_bottom"] = "bullish"

    return patterns
