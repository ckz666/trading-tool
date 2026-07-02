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

    patterns: dict[str, str] = {}   # name → "bullish" | "bearish" | "neutral"

    # ── Single-candle patterns ────────────────────────────────────────────────

    # Doji: body < 10% of range
    if body / tr_safe < 0.10:
        if dw > 2 * uw:
            patterns["dragonfly_doji"] = "bullish"     # long lower wick → buyers took over
        elif uw > 2 * dw:
            patterns["gravestone_doji"] = "bearish"    # long upper wick → sellers took over
        else:
            patterns["doji"] = "neutral"

    # Marubozu: almost no wicks — strong momentum candle
    if body / tr_safe > 0.90:
        patterns["marubozu"] = "bullish" if bull else "bearish"

    # Spinning Top: small body, roughly equal wicks — indecision
    if 0.10 < body / tr_safe < 0.30 and abs(uw - dw) < body:
        patterns["spinning_top"] = "neutral"

    # Hammer: long lower wick, small body at top of range (at support → bullish)
    if dw > 2 * body and uw < body * 0.5 and body / tr_safe < 0.35:
        patterns["hammer"] = "bullish"

    # Inverted Hammer: long upper wick, small body at bottom → bullish (buyers tried)
    if uw > 2 * body and dw < body * 0.5 and body / tr_safe < 0.35:
        patterns["inverted_hammer"] = "bullish"

    # Hanging Man: same shape as hammer but at top → bearish warning
    # (context-blind here; we check if prev trend was up)
    prev_trend_up = df["close"].tail(10).iloc[-1] > df["close"].tail(10).iloc[0]
    if dw > 2 * body and uw < body * 0.5 and prev_trend_up:
        patterns["hanging_man"] = "bearish"

    # Shooting Star: long upper wick, small body at top → bearish
    if uw > 2 * body and dw < body * 0.5 and body / tr_safe < 0.35:
        patterns["shooting_star"] = "bearish"

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

    # Tweezer Bottom: two candles with near-identical lows → support
    if (not pbull and bull
            and abs(c["low"] - p["low"]) / max(p["low"], 1e-10) < 0.002):
        patterns["tweezer_bottom"] = "bullish"

    # Tweezer Top: two candles with near-identical highs → resistance
    if (pbull and not bull
            and abs(c["high"] - p["high"]) / max(p["high"], 1e-10) < 0.002):
        patterns["tweezer_top"] = "bearish"

    # Bullish Kicker: gap up after bearish candle → strong bullish
    if (not pbull and bull and c["open"] > p["open"]):
        patterns["bullish_kicker"] = "bullish"

    # Bearish Kicker: gap down after bullish candle → strong bearish
    if (pbull and not bull and c["open"] < p["open"]):
        patterns["bearish_kicker"] = "bearish"

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

    return patterns
