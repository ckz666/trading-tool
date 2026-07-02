import aiohttp
import asyncio
from datetime import datetime


async def fetch_fear_greed() -> dict:
    """Fear & Greed Index from alternative.me (free, no key needed)."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.alternative.me/fng/?limit=1", timeout=aiohttp.ClientTimeout(total=5)) as r:
                data = await r.json()
                d = data["data"][0]
                return {
                    "value": int(d["value"]),
                    "label": d["value_classification"],
                    "ts": d["timestamp"],
                }
    except Exception:
        return {"value": 50, "label": "Neutral", "ts": "unavailable"}


async def fetch_crypto_headlines(symbol: str = "bitcoin") -> list[str]:
    """Fetch recent crypto headlines from CryptoPanic (free tier)."""
    try:
        url = f"https://cryptopanic.com/api/v1/posts/?auth_token=free&currencies={symbol}&kind=news&public=true"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
                if r.status != 200:
                    return []
                data = await r.json()
                headlines = [p["title"] for p in data.get("results", [])[:5]]
                return headlines
    except Exception:
        return []


async def get_market_sentiment(symbol: str = "BTC/USDT") -> dict:
    coin = symbol.split("/")[0].lower()
    coin_map = {"btc": "bitcoin", "eth": "ethereum", "sol": "solana"}
    coin_name = coin_map.get(coin, coin)

    fg, headlines = await asyncio.gather(
        fetch_fear_greed(),
        fetch_crypto_headlines(coin.upper()),
    )

    return {
        "fear_greed": fg,
        "headlines": headlines,
        "sentiment_bias": _bias_from_fg(fg["value"]),
    }


def _bias_from_fg(value: int) -> str:
    if value <= 25:
        return "extreme_fear"
    elif value <= 45:
        return "fear"
    elif value <= 55:
        return "neutral"
    elif value <= 75:
        return "greed"
    else:
        return "extreme_greed"
