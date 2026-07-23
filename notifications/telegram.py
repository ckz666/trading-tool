"""Best-effort Telegram push notifications for trade events.

No-ops silently (returns False) if TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID aren't
set in .env — never raises, a notification failure must never interrupt
trading. Read lazily via os.getenv() on every call (not module-level
constants) so a key added to .env after the process started still needs a
restart like every other credential in this project, but the module itself
stays importable/testable without env vars present.
"""
import asyncio
import os
import aiohttp

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


async def send_telegram(text: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return False
    url = TELEGRAM_API.format(token=token)
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as session:
            async with session.post(url, data={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }) as r:
                if r.status != 200:
                    body = await r.text()
                    print(f"[Telegram] send failed ({r.status}): {body[:200]}")
                    return False
                return True
    except Exception as e:
        print(f"[Telegram] send error: {e}")
        return False


def notify_fire_and_forget(text: str):
    """Schedule send_telegram() without awaiting it — for call sites that are
    sync (e.g. an engine's _log()) but run inside a live asyncio event loop.
    Swallows the case where no loop is running (e.g. a script/test context)."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(send_telegram(text))
    except RuntimeError:
        pass
