"""Reddit sentiment via OAuth app-only (client_credentials) grant — no free
anonymous access works from this VM (verified 2026-07-23: unauthenticated
requests to reddit.com/*.json get a 403 from this IP, likely a datacenter-IP
block), so a free "script" app (client_id/secret, reddit.com/prefs/apps) is
required even for read-only public data. No username/password needed — the
client_credentials grant reads public subreddits without a logged-in user.

Simple keyword-count sentiment over recent hot post titles, not an ML model
— DeepSeek's assessment (2026-07-23, project memory) was that crypto
subreddit sentiment is "extrem verrauscht" without dedicated NLP, so this is
built and wired in as a low-weight, gracefully-degrading confluence input,
not treated as a strong signal.
"""
import os
import time
from datetime import datetime

import aiohttp

TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
API_BASE = "https://oauth.reddit.com"
USER_AGENT = "trading-tool-sentiment/1.0 (by /u/trading_tool_bot)"

# symbol -> extra subreddit to check alongside the general r/CryptoCurrency
SYMBOL_SUBREDDITS = {
    "BTC": "Bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "XRP": "XRP",
    "DOGE": "dogecoin",
    "ADA": "cardano",
}

BULLISH_WORDS = ["moon", "bullish", "pump", "breakout", "rally", "ath", "long",
                  "buy the dip", "accumulate", "undervalued", "surge", "explode"]
BEARISH_WORDS = ["crash", "bearish", "dump", "capitulation", "rekt", "short",
                  "correction", "collapse", "plunge", "sell off", "selloff", "overvalued"]

_token_cache: dict = {"token": None, "expires_at": 0.0}


async def _get_token(session: aiohttp.ClientSession) -> str | None:
    client_id = os.getenv("REDDIT_CLIENT_ID", "")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        return None
    now = time.time()
    if _token_cache["token"] and now < _token_cache["expires_at"]:
        return _token_cache["token"]
    try:
        auth = aiohttp.BasicAuth(client_id, client_secret)
        async with session.post(
            TOKEN_URL, auth=auth,
            data={"grant_type": "client_credentials"},
            headers={"User-Agent": USER_AGENT},
            timeout=aiohttp.ClientTimeout(total=8),
        ) as r:
            if r.status != 200:
                return None
            d = await r.json()
            token = d.get("access_token")
            _token_cache["token"] = token
            _token_cache["expires_at"] = now + d.get("expires_in", 3600) - 60
            return token
    except Exception:
        return None


async def _fetch_hot_titles(session: aiohttp.ClientSession, token: str, subreddit: str, limit: int = 25) -> list[str]:
    try:
        async with session.get(
            f"{API_BASE}/r/{subreddit}/hot",
            params={"limit": str(limit)},
            headers={"Authorization": f"bearer {token}", "User-Agent": USER_AGENT},
            timeout=aiohttp.ClientTimeout(total=8),
        ) as r:
            if r.status != 200:
                return []
            d = await r.json()
            return [c["data"]["title"].lower() for c in d.get("data", {}).get("children", [])]
    except Exception:
        return []


def _score_titles(titles: list[str]) -> tuple[int, int]:
    bull = sum(1 for t in titles for w in BULLISH_WORDS if w in t)
    bear = sum(1 for t in titles for w in BEARISH_WORDS if w in t)
    return bull, bear


async def fetch_reddit_sentiment(symbol: str) -> dict:
    """symbol: 'BTC/USDT' format. Returns {'available': False} if
    REDDIT_CLIENT_ID/SECRET aren't set or any fetch fails — never raises."""
    if not os.getenv("REDDIT_CLIENT_ID") or not os.getenv("REDDIT_CLIENT_SECRET"):
        return {"available": False, "note": "REDDIT_CLIENT_ID/SECRET not set"}

    coin = symbol.split("/")[0].upper()
    subreddits = ["CryptoCurrency"]
    if coin in SYMBOL_SUBREDDITS:
        subreddits.append(SYMBOL_SUBREDDITS[coin])

    async with aiohttp.ClientSession() as session:
        token = await _get_token(session)
        if not token:
            return {"available": False, "note": "OAuth token fetch failed"}

        all_titles = []
        for sub in subreddits:
            all_titles.extend(await _fetch_hot_titles(session, token, sub))

    if not all_titles:
        return {"available": False, "note": "no posts fetched"}

    bull, bear = _score_titles(all_titles)
    total = bull + bear
    if total == 0:
        bias = "neutral"
    else:
        ratio = bull / total
        bias = "bullish" if ratio >= 0.65 else "bearish" if ratio <= 0.35 else "neutral"

    return {
        "available": True,
        "ts": datetime.now().isoformat(),
        "subreddits": subreddits,
        "post_count": len(all_titles),
        "bull_mentions": bull,
        "bear_mentions": bear,
        "bias": bias,
    }
