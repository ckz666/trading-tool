import pandas as pd
import numpy as np


def sma_crossover(df: pd.DataFrame, fast: int = 10, slow: int = 30) -> pd.Series:
    """Buy when fast SMA crosses above slow SMA, sell when it crosses below."""
    fast_sma = df["close"].rolling(fast).mean()
    slow_sma = df["close"].rolling(slow).mean()
    signal = pd.Series(0, index=df.index)
    signal[fast_sma > slow_sma] = 1
    signal[fast_sma < slow_sma] = -1
    # only trigger on crossover, not while holding
    crossover = signal.diff()
    result = pd.Series(0, index=df.index)
    result[crossover == 2] = 1   # crossed to bullish
    result[crossover == -2] = -1  # crossed to bearish
    return result


def rsi_strategy(df: pd.DataFrame, period: int = 14, oversold: int = 30, overbought: int = 70) -> pd.Series:
    """Buy on RSI oversold, sell on RSI overbought."""
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)

    signal = pd.Series(0, index=df.index)
    signal[rsi < oversold] = 1
    signal[rsi > overbought] = -1
    return signal


def bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    """Buy when price touches lower band, sell at upper band."""
    mid = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    lower = mid - std_dev * std
    upper = mid + std_dev * std

    signal = pd.Series(0, index=df.index)
    signal[df["close"] <= lower] = 1
    signal[df["close"] >= upper] = -1
    return signal


STRATEGIES = {
    "sma_crossover": sma_crossover,
    "rsi": rsi_strategy,
    "bollinger_bands": bollinger_bands,
}
