import os
import aiohttp
from datetime import datetime

CMC_BASE = "https://pro-api.coinmarketcap.com"


def _headers() -> dict:
    return {
        "X-CMC_PRO_API_KEY": os.getenv("COINMARKETCAP_API_KEY", ""),
        "Accept": "application/json",
    }


async def fetch_cmc_data(symbol: str) -> dict:
    """Fetch Fear&Greed, global market metrics, and coin quote from CMC."""
    coin_sym  = symbol.split("/")[0].upper()   # BTC/USDT → BTC
    coin_slug = coin_sym.lower()               # used in fallback only; CMC API accepts symbol directly
    hdrs      = _headers()

    if not hdrs["X-CMC_PRO_API_KEY"]:
        return {"available": False, "note": "COINMARKETCAP_API_KEY not set"}

    results: dict = {"available": True, "ts": datetime.now().isoformat()}

    async with aiohttp.ClientSession(headers=hdrs) as s:

        # ── Fear & Greed Index ───────────────────────────────────────────
        try:
            async with s.get(f"{CMC_BASE}/v3/fear-and-greed/latest",
                             timeout=aiohttp.ClientTimeout(total=6)) as r:
                d = await r.json()
                fg = d.get("data", {})
                val   = int(fg.get("value", 50))
                label = fg.get("value_classification", "Neutral")
                results["fear_greed"] = {
                    "value": val,
                    "label": label,
                    "bias": _fg_bias(val),
                    "source": "CoinMarketCap",
                }
        except Exception as e:
            results["fear_greed"] = {"value": 50, "label": "Neutral", "bias": "neutral", "error": str(e)}

        # ── Global Market Metrics ────────────────────────────────────────
        try:
            async with s.get(f"{CMC_BASE}/v1/global-metrics/quotes/latest",
                             timeout=aiohttp.ClientTimeout(total=6)) as r:
                d = (await r.json()).get("data", {})
                q  = d.get("quote", {}).get("USD", {})
                btc_dom  = d.get("btc_dominance", 0)
                eth_dom  = d.get("eth_dominance", 0)
                total_mc = q.get("total_market_cap", 0)
                mc_chg   = q.get("total_market_cap_yesterday_percentage_change", 0)
                vol_24h  = q.get("total_volume_24h", 0)
                active   = d.get("active_cryptocurrencies", 0)
                results["global_market"] = {
                    "total_market_cap_usd": round(total_mc),
                    "total_volume_24h_usd": round(vol_24h),
                    "market_cap_change_24h_pct": round(mc_chg, 3),
                    "btc_dominance_pct": round(btc_dom, 2),
                    "eth_dominance_pct": round(eth_dom, 2),
                    "active_cryptos": active,
                    # BTC dominance rising = risk-off (altcoins bleed), falling = risk-on
                    "btc_dom_signal": "risk_off" if btc_dom > 55 else
                                      "risk_on"  if btc_dom < 45 else "neutral",
                    "market_trend": "bullish" if mc_chg > 2 else
                                    "bearish" if mc_chg < -2 else "neutral",
                }
        except Exception as e:
            results["global_market"] = {"error": str(e)}

        # ── Coin Quote (price, volume, market cap, dominance) ────────────
        try:
            async with s.get(f"{CMC_BASE}/v2/cryptocurrency/quotes/latest",
                             params={"symbol": coin_sym},
                             timeout=aiohttp.ClientTimeout(total=6)) as r:
                d     = (await r.json()).get("data", {})
                coins = d.get(coin_sym, [])
                coin  = coins[0] if isinstance(coins, list) and coins else d.get(coin_sym, {})
                if isinstance(coin, dict) and coin:
                    q = coin.get("quote", {}).get("USD", {})
                    results["coin"] = {
                        "name":            coin.get("name"),
                        "cmc_rank":        coin.get("cmc_rank"),
                        "price":           q.get("price"),
                        "change_1h_pct":   round(q.get("percent_change_1h",  0), 3),
                        "change_24h_pct":  round(q.get("percent_change_24h", 0), 3),
                        "change_7d_pct":   round(q.get("percent_change_7d",  0), 3),
                        "volume_24h_usd":  round(q.get("volume_24h", 0)),
                        "volume_change_pct": round(q.get("volume_change_24h", 0), 2),
                        "market_cap_usd":  round(q.get("market_cap", 0)),
                        "market_cap_dominance": round(q.get("market_cap_dominance", 0), 3),
                        "fully_diluted_mc": round(q.get("fully_diluted_market_cap", 0)),
                        # momentum signals
                        "momentum": "strong_bull" if q.get("percent_change_24h", 0) > 5 else
                                    "bull"        if q.get("percent_change_24h", 0) > 1 else
                                    "strong_bear" if q.get("percent_change_24h", 0) < -5 else
                                    "bear"        if q.get("percent_change_24h", 0) < -1 else "neutral",
                        "volume_spike": q.get("volume_change_24h", 0) > 50,
                    }
        except Exception as e:
            results["coin"] = {"error": str(e)}

        # ── Trending (top gainers signal) ─────────────────────────────────
        try:
            async with s.get(f"{CMC_BASE}/v1/cryptocurrency/trending/gainers-losers",
                             params={"time_period": "24h", "limit": "5"},
                             timeout=aiohttp.ClientTimeout(total=6)) as r:
                d = await r.json()
                if d.get("status", {}).get("error_code") == 0:
                    gainers = d.get("data", {}).get("gainers", [])
                    losers  = d.get("data", {}).get("losers",  [])
                    results["trending"] = {
                        "top_gainers": [f"{c.get('symbol')} +{c.get('quote',{}).get('USD',{}).get('percent_change_24h',0):.1f}%" for c in gainers[:3]],
                        "top_losers":  [f"{c.get('symbol')} {c.get('quote',{}).get('USD',{}).get('percent_change_24h',0):.1f}%"  for c in losers[:3]],
                        "note": "gainers in alt season = risk-on; major coins in gainers = rotation"
                    }
                else:
                    results["trending"] = {"note": "not available on free tier"}
        except Exception:
            results["trending"] = {"note": "unavailable"}

    # ── composite CMC signal ──────────────────────────────────────────────
    fg_val     = results.get("fear_greed", {}).get("value", 50)
    coin_mom   = results.get("coin", {}).get("momentum", "neutral")
    mkt_trend  = results.get("global_market", {}).get("market_trend", "neutral")
    btc_sig    = results.get("global_market", {}).get("btc_dom_signal", "neutral")

    score = 0.0
    score += (fg_val - 50) / 50 * 2          # -2..+2 from F&G
    score += {"strong_bull": 1.5, "bull": 0.5, "neutral": 0, "bear": -0.5, "strong_bear": -1.5}.get(coin_mom, 0)
    score += {"bullish": 0.5, "neutral": 0, "bearish": -0.5}.get(mkt_trend, 0)
    if btc_sig == "risk_off" and coin_sym != "BTC": score -= 0.5  # alts suffer when BTC dom rises

    results["cmc_signal"] = {
        "score": round(score, 2),
        "bias":  "strongly_bullish" if score >= 2.5 else
                 "bullish"          if score >= 1.0 else
                 "slightly_bullish" if score > 0.2  else
                 "strongly_bearish" if score <= -2.5 else
                 "bearish"          if score <= -1.0 else
                 "slightly_bearish" if score < -0.2  else "neutral",
    }

    return results


def _fg_bias(v: int) -> str:
    if v <= 20: return "extreme_fear"
    if v <= 40: return "fear"
    if v <= 60: return "neutral"
    if v <= 80: return "greed"
    return "extreme_greed"
