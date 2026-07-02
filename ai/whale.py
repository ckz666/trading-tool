import asyncio
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime
import aiohttp

def _coin(symbol: str) -> str:
    """BTC/USDT → BTC"""
    return symbol.split("/")[0].upper()

def _binance_sym(symbol: str) -> str:
    """BTC/USDT → BTCUSDT"""
    return symbol.replace("/", "").upper()

LARGE_TRADE_USD = 200_000

BULLISH_KW = {
    "moon","pump","buy","bull","long","breakout","accumulate","ath","hodl",
    "bullish","surge","rally","bounce","support","recovery","rebound","rise",
    "gains","gain","soar","soars","climbs","climb","higher","upside","growth",
    "adoption","bullrun","all-time","record","milestone","launch","upgrade",
}
BEARISH_KW = {
    "dump","crash","bear","short","sell","rekt","correction","collapse","drop",
    "tank","bearish","resistance","overvalued","scam","rug","decline","falls",
    "fall","dip","dips","dives","dive","loss","losses","plunge","plunges",
    "warning","risk","hack","exploit","ban","ban","regulation","crackdown",
    "liquidation","liquidations","fear","panic","selloff","downturn","slump",
}


# ── 1. Top Trader L/S Ratio (Binance public futures, no key) ─────────────────
async def fetch_top_trader_ratio(symbol: str) -> dict:
    bsym = _binance_sym(symbol)
    results = {}
    async with aiohttp.ClientSession() as s:
        try:
            # Account ratio (how many accounts are long vs short)
            async with s.get("https://fapi.binance.com/futures/data/topLongShortAccountRatio",
                             params={"symbol": bsym, "period": "1h", "limit": "3"},
                             timeout=aiohttp.ClientTimeout(total=6)) as r:
                rows = await r.json()
                latest = rows[0] if rows else {}
                long_pct  = float(latest.get("longAccount", 0.5)) * 100
                short_pct = float(latest.get("shortAccount", 0.5)) * 100
                ratio     = float(latest.get("longShortRatio", 1.0))
                bias = "very_long"  if ratio > 2.5 else \
                       "long"       if ratio > 1.4 else \
                       "very_short" if ratio < 0.4 else \
                       "short"      if ratio < 0.7 else "neutral"
                results["account_ratio"] = {
                    "ratio": round(ratio, 3), "long_pct": round(long_pct, 1),
                    "short_pct": round(short_pct, 1), "bias": bias,
                }
        except Exception as e:
            results["account_ratio"] = {"bias": "unavailable", "error": str(e)}

        try:
            # Position ratio (volume-weighted, stronger signal)
            async with s.get("https://fapi.binance.com/futures/data/topLongShortPositionRatio",
                             params={"symbol": bsym, "period": "1h", "limit": "3"},
                             timeout=aiohttp.ClientTimeout(total=6)) as r:
                rows = await r.json()
                latest = rows[0] if rows else {}
                pratio = float(latest.get("longShortRatio", 1.0))
                pbias  = "very_long"  if pratio > 2.5 else \
                         "long"       if pratio > 1.4 else \
                         "very_short" if pratio < 0.4 else \
                         "short"      if pratio < 0.7 else "neutral"
                results["position_ratio"] = {
                    "ratio": round(pratio, 3),
                    "long_pct": round(float(latest.get("longAccount", 0.5)) * 100, 1),
                    "bias": pbias,
                }
        except Exception as e:
            results["position_ratio"] = {"bias": "unavailable", "error": str(e)}

        try:
            # Global L/S ratio (all traders, not just top)
            async with s.get("https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
                             params={"symbol": bsym, "period": "1h", "limit": "1"},
                             timeout=aiohttp.ClientTimeout(total=6)) as r:
                rows = await r.json()
                latest = rows[0] if rows else {}
                gratio = float(latest.get("longShortRatio", 1.0))
                results["global_ratio"] = {
                    "ratio": round(gratio, 3),
                    "long_pct": round(float(latest.get("longAccount", 0.5)) * 100, 1),
                }
        except Exception as e:
            results["global_ratio"] = {"ratio": 1.0, "error": str(e)}

    # composite bias (contrarian: extreme longs → bearish signal)
    ar_bias = results.get("account_ratio", {}).get("bias", "neutral")
    pr_bias = results.get("position_ratio", {}).get("bias", "neutral")
    extreme = ar_bias in ("very_long", "very_short") or pr_bias in ("very_long", "very_short")
    results["contrarian_note"] = (
        "⚠ VERY LONG: market makers may squeeze longs → bearish risk" if ar_bias == "very_long" else
        "⚠ VERY SHORT: short squeeze possible → bullish risk"          if ar_bias == "very_short" else
        "Ratio within normal range"
    )
    results["extreme_reading"] = extreme
    return results


# ── 2. Large Trade Scanner (Binance public aggTrades, no key needed) ─────────
async def fetch_large_trades(symbol: str) -> dict:
    bsym = _binance_sym(symbol)
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(
                "https://fapi.binance.com/fapi/v1/aggTrades",
                params={"symbol": bsym, "limit": "1000"},
                timeout=aiohttp.ClientTimeout(total=8),
            ) as r:
                if r.status != 200:
                    raise Exception(f"HTTP {r.status}")
                trades = await r.json()

        whale_buys = whale_sells = 0
        buy_vol = sell_vol = total_vol = 0.0

        for t in trades:
            price    = float(t.get("p", 0))
            qty      = float(t.get("q", 0))
            notional = price * qty
            total_vol += notional
            if notional >= LARGE_TRADE_USD:
                # isBuyerMaker=True means market seller hit the buyer → sell pressure
                if t.get("m", False):
                    whale_sells += 1; sell_vol += notional
                else:
                    whale_buys  += 1; buy_vol  += notional

        total_whale = buy_vol + sell_vol
        buy_pct = buy_vol / max(total_whale, 1)
        bias = "strongly_bullish" if buy_pct > 0.70 else \
               "bullish"          if buy_pct > 0.58 else \
               "strongly_bearish" if buy_pct < 0.30 else \
               "bearish"          if buy_pct < 0.42 else "neutral"

        return {
            "whale_buys": whale_buys, "whale_sells": whale_sells,
            "whale_buy_vol_usd": round(buy_vol),
            "whale_sell_vol_usd": round(sell_vol),
            "whale_bias": bias,
            "whale_pct_of_volume": round(total_whale / max(total_vol, 1) * 100, 1),
            "threshold_usd": LARGE_TRADE_USD,
            "source": "Binance aggTrades",
        }
    except Exception as e:
        return {"whale_buys": 0, "whale_sells": 0, "whale_bias": "unavailable", "error": str(e)}




# ── 4. Liquidations — estimated via OI drop + price direction ────────────────
async def fetch_liquidations(symbol: str) -> dict:
    """
    Estimate liquidation pressure from Open Interest change + price direction.
    OI drop while price drops  → long liquidations dominant
    OI drop while price rises  → short liquidations dominant
    OI stable / up             → no notable liquidations
    Uses Binance public klines (OI proxy via volume spike) as fallback.
    """
    bsym = _binance_sym(symbol)
    try:
        async with aiohttp.ClientSession() as s:
            # Fetch OI history (1h candles, last 12)
            oi_task = s.get(
                "https://fapi.binance.com/futures/data/openInterestHist",
                params={"symbol": bsym, "period": "1h", "limit": "12"},
                timeout=aiohttp.ClientTimeout(total=8),
            )
            # Fetch recent 24h price change
            px_task = s.get(
                "https://fapi.binance.com/fapi/v1/ticker/24hr",
                params={"symbol": bsym},
                timeout=aiohttp.ClientTimeout(total=6),
            )
            async with oi_task as oi_r, px_task as px_r:
                oi_data = await oi_r.json() if oi_r.status == 200 else []
                px_data = await px_r.json() if px_r.status == 200 else {}

        if not oi_data or len(oi_data) < 2:
            return {"long_liq_usd": 0, "short_liq_usd": 0, "source": "oi_unavailable"}

        oi_values   = [float(r["sumOpenInterestValue"]) for r in oi_data]
        oi_latest   = oi_values[-1]
        oi_prev     = oi_values[-2]
        oi_change   = oi_latest - oi_prev          # negative = OI dropped = liquidations
        price_chg   = float(px_data.get("priceChangePercent", 0))

        # Rough estimate: assume OI drop translates 1:1 to liquidated notional
        liq_estimate = max(0, -oi_change)

        if oi_change < -oi_latest * 0.005 and price_chg < -0.3:
            long_liq  = round(liq_estimate)
            short_liq = 0
            signal    = "long_squeeze"
        elif oi_change < -oi_latest * 0.005 and price_chg > 0.3:
            long_liq  = 0
            short_liq = round(liq_estimate)
            signal    = "short_squeeze"
        else:
            long_liq = short_liq = 0
            signal   = "none"

        return {
            "long_liq_usd":  long_liq,
            "short_liq_usd": short_liq,
            "oi_usd":        round(oi_latest),
            "oi_change_usd": round(oi_change),
            "signal":        signal,
            "source":        "OI+price estimate",
        }
    except Exception as e:
        return {"long_liq_usd": 0, "short_liq_usd": 0, "source": "unavailable", "error": str(e)}


# ── 5. News Sentiment (RSS: CoinDesk + Cointelegraph) ─────────────────────────
RSS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
]

_COIN_ALIASES = {
    "BTC": {"btc", "bitcoin"},
    "ETH": {"eth", "ethereum", "ether"},
    "SOL": {"sol", "solana"},
    "XRP": {"xrp", "ripple"},
    "BNB": {"bnb", "binance"},
    "ADA": {"ada", "cardano"},
    "DOGE": {"doge", "dogecoin"},
    "AVAX": {"avax", "avalanche"},
}
_GENERIC = {"crypto", "market", "defi", "nft", "altcoin", "blockchain", "token"}

def _score_title(title: str, coin: str) -> int:
    """Return +1 bullish, -1 bearish, 0 neutral/irrelevant."""
    t     = title.lower()
    words = set(re.findall(r'\w+', t))
    aliases = _COIN_ALIASES.get(coin.upper(), {coin.lower()})
    if not (aliases & words) and not (_GENERIC & words):
        return 0
    b_hits = len(words & BULLISH_KW)
    s_hits = len(words & BEARISH_KW)
    if b_hits > s_hits: return 1
    if s_hits > b_hits: return -1
    return 0

async def fetch_news_sentiment(symbol: str) -> dict:
    coin    = _coin(symbol)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; TradingBot/1.0)"}
    bull = bear = 0
    headlines: list[str] = []

    async with aiohttp.ClientSession(headers=headers) as s:
        for feed_url in RSS_FEEDS:
            try:
                async with s.get(feed_url, timeout=aiohttp.ClientTimeout(total=8)) as r:
                    if r.status != 200:
                        continue
                    xml_text = await r.text()
                    root = ET.fromstring(xml_text)
                    ns   = {"atom": "http://www.w3.org/2005/Atom"}
                    # RSS 2.0 items
                    items = root.findall(".//item")
                    # Atom entries
                    if not items:
                        items = root.findall(".//atom:entry", ns)
                    for item in items[:15]:
                        title = (item.findtext("title") or
                                 item.findtext("atom:title", namespaces=ns) or "").strip()
                        if not title:
                            continue
                        score = _score_title(title, coin)
                        if score > 0:   bull += 1
                        elif score < 0: bear += 1
                        if score != 0 and len(headlines) < 6:
                            headlines.append(title[:90])
            except Exception:
                continue

    if not headlines and bull == 0 and bear == 0:
        return {"bias": "unavailable", "bull": 0, "bear": 0, "headlines": [], "count": 0}

    bias = "bullish" if bull > bear + 1 else \
           "bearish" if bear > bull + 1 else "neutral"
    return {"bias": bias, "bull": bull, "bear": bear,
            "headlines": headlines[:5], "count": bull + bear}


# ── Composite ─────────────────────────────────────────────────────────────────
async def get_all_whale_data(symbol: str) -> dict:
    from ai.cmc import fetch_cmc_data

    results = await asyncio.gather(
        fetch_top_trader_ratio(symbol),
        fetch_large_trades(symbol),
        fetch_liquidations(symbol),
        fetch_news_sentiment(symbol),
        fetch_cmc_data(symbol),
        return_exceptions=True,
    )

    def safe(r, fb): return r if isinstance(r, dict) else fb

    tr  = safe(results[0], {})
    lt  = safe(results[1], {"whale_bias": "unavailable"})
    liq = safe(results[2], {"long_liq_usd": 0, "short_liq_usd": 0})
    ns  = safe(results[3], {"bias": "unavailable"})
    cmc = safe(results[4], {"available": False})

    score = 0.0
    bmap = {"strongly_bullish": 2, "bullish": 1, "slightly_bullish": 0.5, "neutral": 0,
            "unavailable": 0,
            "slightly_bearish": -0.5, "bearish": -1, "strongly_bearish": -2}

    score += bmap.get(lt.get("whale_bias", "neutral"), 0) * 1.5   # exchange trades = high weight
    score += bmap.get(ns.get("bias", "neutral"), 0) * 0.5         # news = lower weight
    score += bmap.get(cmc.get("cmc_signal", {}).get("bias", "neutral"), 0) * 1.0

    # top trader L/S is CONTRARIAN
    ar = tr.get("account_ratio", {}).get("bias", "neutral")
    if ar == "very_long":    score -= 1.5
    elif ar == "very_short": score += 1.5
    elif ar == "long":       score -= 0.5
    elif ar == "short":      score += 0.5

    composite = "strongly_bullish" if score >= 2.5 else \
                "bullish"          if score >= 1.0 else \
                "slightly_bullish" if score > 0.2 else \
                "strongly_bearish" if score <= -2.5 else \
                "bearish"          if score <= -1.0 else \
                "slightly_bearish" if score < -0.2 else "neutral"

    return {
        "composite_score":  round(score, 2),
        "composite_bias":   composite,
        "top_trader_ratio": tr,
        "large_trades":     lt,
        "liquidations":     liq,
        "news_sentiment":   ns,
        "cmc":              cmc,
        "ts": datetime.now().isoformat(),
    }
