import os
from dotenv import load_dotenv

load_dotenv()

BITGET_API_KEY = os.getenv("BITGET_API_KEY", "")
BITGET_SECRET = os.getenv("BITGET_SECRET", "")
BITGET_PASSPHRASE = os.getenv("BITGET_PASSPHRASE", "")

PAPER_BALANCE = float(os.getenv("PAPER_TRADING_BALANCE", "10000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]
DEFAULT_TIMEFRAME = "1h"
DEFAULT_SYMBOL = "BTC/USDT"
