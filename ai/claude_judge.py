import json
import os
import numpy as np
import pandas as pd
import anthropic

_client = None

def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set in .env")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


SYSTEM_PROMPT = """You are an expert crypto futures trader and risk manager.
You trade USDT-M perpetual contracts on Bitget.

You receive comprehensive market data and must make a final trading decision.

Your response MUST be valid JSON with this EXACT structure:
{
  "action": "open_long" | "open_short" | "close_long" | "close_short" | "hold",
  "leverage": <integer 1-15>,
  "position_size_pct": <0.05-0.5, fraction of max allowed margin capital>,
  "stop_loss_pct": <0.005-0.05>,
  "take_profit_pct": <0.01-0.15>,
  "confidence": <0.0-1.0>,
  "trailing_sl": <true|false — use trailing stop loss instead of fixed>,
  "trail_pct": <0.005-0.05 — trailing distance, e.g. 0.02 = 2%>,
  "reasoning": "<2-3 sentences explaining the decision>",
  "risk_notes": "<specific risks for this trade>"
}

TRAILING STOP LOSS RULES:
- Use trailing_sl: true when entering a strong trend (ADX > 30, regime=trending/strong_trend)
- trail_pct should be 1.5–2x the ATR (atr_norm) to avoid noise triggering the stop
- In ranging markets: use fixed SL (trailing_sl: false) — price oscillates too much
- Default: trailing_sl: false, trail_pct: 0.02

LEVERAGE RULES (strict):
- Only go above 5x when ML confidence > 0.80 AND multiple indicators align
- Only go above 10x when ML confidence > 0.88 AND trend is crystal clear
- Max 15x only in the strongest setups (confluence of ALL signals)
- Reduce leverage by 50% when Fear & Greed is extreme (< 20 or > 80)
- Reduce leverage by 30% when funding rate is extreme (> 0.05% or < -0.05%)
- When in doubt: use 2-3x as default

MARKET REGIME STRATEGY (use ADX to choose approach):

RANGING market (ADX < 20):
- Strategy: Mean Reversion — fade extremes, target the middle of the range
- open_long when: RSI < 35 AND price near lower Bollinger Band (BB% < 0.15)
- open_short when: RSI > 65 AND price near upper Bollinger Band (BB% > 0.85)
- Take profit: 0.5–1.5% (target BB midline, NOT a big move)
- Stop loss: tight, just outside the range (0.8–1.5%)
- Leverage: 2–5x max (ranging markets can break out suddenly)
- DO NOT chase breakouts in ranging markets — wait for reversion

TRANSITIONING market (ADX 20–25):
- Be cautious — regime is unclear, use smaller size and wider stops
- Slightly prefer mean-reversion unless a clear pattern forms

TRENDING market (ADX > 25):
- Strategy: Trend Following — buy dips in uptrend, short bounces in downtrend
- open_long when: RSI 40–60 in uptrend (not overbought), MACD bullish, EMA aligned
- open_short when: RSI 40–60 in downtrend, MACD bearish, EMA aligned
- Take profit: 2–6% (let winners run in trend)
- Stop loss: 1.5–3% (give room for retracement)
- Leverage: up to 10x with high confidence

STRONG TREND (ADX > 40):
- Add to winners, wider stops, larger TP targets

SIGNAL LOGIC:
- open_long: price expected to rise
- open_short: price expected to fall
- close_long / close_short: close existing position
- hold: no clear edge — DO NOT trade without edge

ML SIGNAL OVERRIDE POLICY (critical — follow strictly):
You receive a 3-model ensemble signal (GBM + RandomForest + ExtraTrees).
These models were trained with class-balanced weights and predict buy/sell only when
there is genuine directional evidence. Treat them as a quantitative co-pilot.

- ML confidence >= 0.60 AND confluence >= 6/14:
  → MUST trade in ML direction. Override only for: extreme funding (>0.08%),
    imminent liquidation risk, or 1D macro directly opposite with RSI >75 or <25.
  → If you override, state the specific critical reason in reasoning.

- ML confidence 0.50–0.59 AND confluence >= 5/14:
  → Strong prior toward ML direction. Override only with 2+ clear counter-signals.

- ML confidence < 0.50:
  → Full discretion. Confluence score is your primary guide.

The ensemble rarely fires a directional signal — when it does with confidence >0.55,
it is more likely right than wrong. Holding when signals align wastes edge.

RISK/REWARD:
- Minimum R:R ratio = 1.5 (take_profit must be >= 1.5x stop_loss distance)
- In ranging markets: tighter TP/SL, lower leverage
- In trending markets: wider TP, medium SL, higher leverage allowed

Write the 'reasoning' and 'risk_notes' fields in German.
Return ONLY the JSON object, no markdown, no explanation outside JSON."""


def _candle_context(df: pd.DataFrame, n: int = 30) -> str:
    """Compact candle table + key price levels for the last N candles."""
    tail = df.tail(n).copy()
    lines = ["Timestamp(UTC)       Open      High       Low     Close    Volume   Dir"]
    lines.append("─" * 72)
    for ts, row in tail.iterrows():
        direction = "▲" if row["close"] >= row["open"] else "▼"
        lines.append(
            f"{str(ts)[:16]}  {row['open']:>8.2f}  {row['high']:>8.2f}  "
            f"{row['low']:>8.2f}  {row['close']:>8.2f}  {row['volume']:>8.0f}  {direction}"
        )

    # Support / Resistance via rolling pivots
    highs  = df["high"].tail(100)
    lows   = df["low"].tail(100)
    close  = df["close"].iloc[-1]

    resistance_levels = sorted(
        [h for h in highs.nlargest(5).values if h > close], reverse=True
    )[:3]
    support_levels = sorted(
        [l for l in lows.nsmallest(5).values if l < close], reverse=True
    )[:3]

    # Recent swing high/low (last 20 candles)
    recent = df.tail(20)
    swing_high = recent["high"].max()
    swing_low  = recent["low"].min()
    trend_20   = "UP" if df["close"].tail(20).iloc[-1] > df["close"].tail(20).iloc[0] else "DOWN"
    trend_5    = "UP" if df["close"].tail(5).iloc[-1]  > df["close"].tail(5).iloc[0]  else "DOWN"

    # Average candle size (volatility feel)
    avg_candle_pct = (abs(df["close"] - df["open"]) / df["open"] * 100).tail(20).mean()

    lines.append("")
    lines.append(f"PRICE LEVELS (current: {close:.2f}):")
    lines.append(f"  Resistance: {' / '.join(f'${r:.2f}' for r in resistance_levels) or 'none above'}")
    lines.append(f"  Support:    {' / '.join(f'${s:.2f}' for s in support_levels) or 'none below'}")
    lines.append(f"  Swing High (20): ${swing_high:.2f}  |  Swing Low (20): ${swing_low:.2f}")
    lines.append(f"  Trend 5c: {trend_5}  |  Trend 20c: {trend_20}")
    lines.append(f"  Avg candle body: {avg_candle_pct:.3f}% of price")

    return "\n".join(lines)


def _htf_context(df_4h: pd.DataFrame) -> str:
    """Summarise 4h higher-timeframe trend for Claude."""
    if df_4h is None or len(df_4h) < 21:
        return "No 4h data available"
    close = df_4h["close"]
    ema21 = close.ewm(span=21).mean()
    ema50 = close.ewm(span=50).mean()
    cur   = close.iloc[-1]
    e21   = ema21.iloc[-1]
    e50   = ema50.iloc[-1]
    trend = "BULLISH" if cur > e21 > e50 else \
            "BEARISH" if cur < e21 < e50 else \
            "MIXED"
    chg5  = (cur / close.iloc[-5] - 1) * 100
    chg10 = (cur / close.iloc[-10] - 1) * 100
    return (
        f"4h Trend: {trend}  |  Price vs EMA21: {((cur/e21-1)*100):+.2f}%  vs EMA50: {((cur/e50-1)*100):+.2f}%\n"
        f"4h Change 5c: {chg5:+.2f}%  |  4h Change 10c: {chg10:+.2f}%\n"
        f"Rule: {'Only consider LONG setups — HTF is bullish' if trend=='BULLISH' else 'Only consider SHORT setups — HTF is bearish' if trend=='BEARISH' else 'Mixed HTF — reduce size, require strong confirmation'}"
    )


def _mtf_context(df_1d: pd.DataFrame, df_15m: pd.DataFrame) -> str:
    """Summarise 1D (macro) and 15M (entry) context for Claude."""
    lines = []

    # ── 1D macro trend ──
    if df_1d is not None and len(df_1d) > 20:
        try:
            c1d   = df_1d["close"]
            e9    = c1d.ewm(span=9).mean().iloc[-1]
            e21   = c1d.ewm(span=21).mean().iloc[-1]
            e50   = c1d.ewm(span=50).mean().iloc[-1]
            cur1d = c1d.iloc[-1]
            avg_gain = c1d.diff().clip(lower=0).rolling(14).mean().iloc[-1]
            avg_loss = c1d.diff().clip(upper=0).abs().rolling(14).mean().iloc[-1]
            rsi1d = 50.0 if avg_loss == 0 else round(100 - 100 / (1 + avg_gain / avg_loss), 1)
            chg7d = (cur1d / c1d.iloc[-7] - 1) * 100
            trend1d = ("BULLISH" if cur1d > e9 > e21 else
                       "BEARISH" if cur1d < e9 < e21 else "MIXED")
            lines.append(
                f"1D Macro: {trend1d}  |  RSI: {rsi1d:.0f}  |  7d change: {chg7d:+.1f}%\n"
                f"  vs EMA9: {((cur1d/e9-1)*100):+.2f}%  vs EMA21: {((cur1d/e21-1)*100):+.2f}%  vs EMA50: {((cur1d/e50-1)*100):+.2f}%\n"
                f"  Rule: {'STRONG MACRO BULL — prefer LONG, avoid shorts' if trend1d=='BULLISH' and rsi1d<70 else 'STRONG MACRO BEAR — prefer SHORT, avoid longs' if trend1d=='BEARISH' and rsi1d>30 else 'Macro unclear — reduce size'}"
            )
        except Exception:
            lines.append("1D Macro: Calculation error")
    else:
        lines.append("1D Macro: No daily data available")

    # ── 15M entry timing ──
    if df_15m is not None and len(df_15m) > 30:
        try:
            c15   = df_15m["close"]
            e9_15  = c15.ewm(span=9).mean().iloc[-1]
            e21_15 = c15.ewm(span=21).mean().iloc[-1]
            cur15  = c15.iloc[-1]
            avg_gain15 = c15.diff().clip(lower=0).rolling(14).mean().iloc[-1]
            avg_loss15 = c15.diff().clip(upper=0).abs().rolling(14).mean().iloc[-1]
            rsi15 = 50.0 if avg_loss15 == 0 else round(100 - 100 / (1 + avg_gain15 / avg_loss15), 1)
            chg1h = (cur15 / c15.iloc[-4] - 1) * 100
            mom15 = ("BULLISH" if cur15 > e9_15 > e21_15 else
                     "BEARISH" if cur15 < e9_15 < e21_15 else "MIXED")
            lines.append(
                f"15M Entry: {mom15}  |  RSI: {rsi15:.0f}  |  1h change: {chg1h:+.2f}%\n"
                f"  Rule: {'Entry momentum aligned with long' if mom15=='BULLISH' else 'Entry momentum aligned with short' if mom15=='BEARISH' else 'Entry momentum unclear — wait for cleaner setup'}"
            )
        except Exception:
            lines.append("15M Entry: Calculation error")
    else:
        lines.append("15M Entry: No 15m data available")

    return "\n".join(lines)


def judge(
    symbol: str,
    ml_signal: dict,
    indicators: dict,
    order_book: dict,
    patterns: dict,
    sentiment: dict,
    position: dict,
    portfolio: dict,
    funding_rate: dict = None,
    open_interest: dict = None,
    whale_data: dict = None,
    df: pd.DataFrame = None,
    df_4h: pd.DataFrame = None,
    df_1d: pd.DataFrame = None,
    df_15m: pd.DataFrame = None,
    market_sentiment: dict = None,
) -> dict:
    ob_bids = sum(b[1] for b in order_book.get("bids", [])[:10])
    ob_asks = sum(a[1] for a in order_book.get("asks", [])[:10])
    ob_imbalance = round((ob_bids - ob_asks) / (ob_bids + ob_asks + 1e-9), 3)
    current_price = order_book.get("bids", [[0]])[0][0] if order_book.get("bids") else 0

    fr = funding_rate or {}
    oi = open_interest or {}
    wh = whale_data or {}
    candle_ctx = _candle_context(df, 30) if df is not None else "No candle data provided"
    htf_ctx    = _htf_context(df_4h)
    mtf_ctx    = _mtf_context(df_1d, df_15m)

    # Funding rate extremes → contrarian signal
    fr_rate = float(fr.get("rate", 0) or 0)
    if fr_rate > 0.0005:
        fr_note = f"⚠ EXTREME POSITIVE funding ({fr_rate*100:.3f}%) — market heavily long → contrarian SHORT bias"
    elif fr_rate < -0.0005:
        fr_note = f"⚠ EXTREME NEGATIVE funding ({fr_rate*100:.3f}%) — market heavily short → contrarian LONG bias"
    elif fr_rate > 0.0001:
        fr_note = f"Positive funding ({fr_rate*100:.3f}%) — longs paying, slight bearish pressure"
    elif fr_rate < -0.0001:
        fr_note = f"Negative funding ({fr_rate*100:.3f}%) — shorts paying, slight bullish pressure"
    else:
        fr_note = f"Neutral funding ({fr_rate*100:.3f}%)"

    # determine current position context
    has_long = position and position.get("side") == "long" and position.get("amount", 0) > 0
    has_short = position and position.get("side") == "short" and position.get("amount", 0) > 0
    position_info = "None"
    if has_long:
        position_info = f"LONG {position['amount']} @ {position['entry_price']} (liq: {position.get('liquidation_price','?')}, PnL: {position.get('unrealized_pnl',0):.2f} USDT, ROE: {position.get('roe_pct',0):.1f}%)"
    elif has_short:
        position_info = f"SHORT {position['amount']} @ {position['entry_price']} (liq: {position.get('liquidation_price','?')}, PnL: {position.get('unrealized_pnl',0):.2f} USDT, ROE: {position.get('roe_pct',0):.1f}%)"

    # Confluence score (passed via ml_signal dict by autotrader)
    c_score   = ml_signal.get("confluence_score", "?")
    c_reasons = ml_signal.get("confluence_reasons", [])
    ensemble_votes = ml_signal.get("votes", {})
    agreement = ml_signal.get("agreement", 1.0)

    # Market structure
    ms_trend = indicators.get("market_structure", "unknown")
    swing_hi = indicators.get("swing_high", 0)
    swing_lo = indicators.get("swing_low", 0)

    user_msg = f"""
FUTURES MARKET: {symbol} Perpetual
Current Price: ${current_price:,.2f}

ML SIGNAL (Ensemble: GBM + RandomForest + ExtraTrees):
- Signal: {ml_signal['label'].upper()} (raw: {ml_signal['signal']})
- Confidence: {ml_signal['confidence']} ({ml_signal['confidence']*100:.1f}%) {'⚡ HIGH — override policy applies' if ml_signal['confidence'] >= 0.60 else '→ moderate' if ml_signal['confidence'] >= 0.50 else '→ low'}
- Model Agreement: {agreement*100:.0f}% ({ensemble_votes.get('gbm','?')}/{ensemble_votes.get('rf','?')}/{ensemble_votes.get('et','?')})
- Probabilities: sell={ml_signal.get('probabilities',{}).get('sell',0):.3f} / hold={ml_signal.get('probabilities',{}).get('hold',0):.3f} / buy={ml_signal.get('probabilities',{}).get('buy',0):.3f}

CONFLUENCE SCORE: {c_score}/18  (1D+4H+1H+15M + Squeeze + Funding-Extreme)
Aligned signals: {', '.join(c_reasons) if c_reasons else 'keine'}
(Nur Trades mit Confluence ≥3/18 werden weitergeleitet — du siehst nur Setups mit Mindestausrichtung)

MARKET REGIME & STRUCTURE:
- ADX: {indicators.get('adx', 0):.1f}  (+DI {indicators.get('adx_pos', 0):.1f} / -DI {indicators.get('adx_neg', 0):.1f})
- Regime: {indicators.get('regime', 'unknown').upper().replace('_',' ')} {'← USE MEAN REVERSION' if indicators.get('regime') == 'ranging' else '← USE TREND FOLLOWING' if indicators.get('regime') in ('trending','strong_trend') else '← CAUTION, unclear'}
- BB Width: {indicators.get('bb_width', 0):.4f} {'← SQUEEZE: breakout likely soon' if indicators.get('bb_width', 1) < 0.03 else ''}
- Market Structure (1h): {ms_trend.upper()} {'← HH/HL confirmed uptrend' if ms_trend=='uptrend' else '← LL/LH confirmed downtrend' if ms_trend=='downtrend' else '← Expanding volatility' if ms_trend=='expanding' else '← Contracting/wedge' if ms_trend=='contracting' else '← No clear structure'}
- Swing High: ${swing_hi:,.2f}  |  Swing Low: ${swing_lo:,.2f}
RULE: If market_structure contradicts ML direction → reduce size by 50% or hold

TECHNICAL INDICATORS:
- RSI: {indicators.get('rsi')} {'⬇ OVERSOLD — mean reversion LONG signal' if indicators.get('rsi', 50) < 35 else '⬆ OVERBOUGHT — mean reversion SHORT signal' if indicators.get('rsi', 50) > 65 else '→ neutral'}
- MACD Diff: {indicators.get('macd_diff')} {'▲ bullish' if indicators.get('macd_diff', 0) > 0 else '▼ bearish'}
- Bollinger %B: {indicators.get('bb_pct')} {'← AT LOWER BAND (mean reversion entry zone)' if indicators.get('bb_pct', 0.5) < 0.15 else '→ AT UPPER BAND (mean reversion short zone)' if indicators.get('bb_pct', 0.5) > 0.85 else '→ mid range'}
- EMA cross (fast-slow)/price: {indicators.get('ema_cross_norm')} {'▲ fast above slow' if indicators.get('ema_cross_norm', 0) > 0 else '▼ fast below slow'}
- Volume ratio vs 14-avg: {indicators.get('volume_ratio')}x {'HIGH VOLUME' if indicators.get('volume_ratio', 1) > 1.5 else ''}
- ATR normalized: {indicators.get('atr_norm')} (volatility)

ORDER BOOK:
- Bid depth top-10: {ob_bids:,.2f} USDT
- Ask depth top-10: {ob_asks:,.2f} USDT
- Imbalance score: {ob_imbalance} (+1=all bids=bullish, -1=all asks=bearish)

FUTURES-SPECIFIC:
- Funding Rate: {fr.get('rate', 'N/A')} → {fr_note}
- Next Funding: {fr.get('next_ts', 'N/A')}
- Open Interest: {oi.get('open_interest', 'N/A')}

MARKET SENTIMENT (Bitget Live):
- Account L/S Ratio: {f"{(market_sentiment or {}).get('long_ratio', 0)*100:.1f}% long / {(market_sentiment or {}).get('short_ratio', 0)*100:.1f}% short" if (market_sentiment or {}).get('long_ratio') else 'N/A'} {'← crowd heavily long (contrarian BEARISH)' if (market_sentiment or {}).get('long_ratio', 0.5) > 0.65 else '← crowd heavily short (contrarian BULLISH)' if (market_sentiment or {}).get('long_ratio', 0.5) < 0.40 else '← balanced'}
- Position L/S Ratio: {f"{(market_sentiment or {}).get('pos_long_ratio', 0)*100:.1f}% long / {(market_sentiment or {}).get('pos_short_ratio', 0)*100:.1f}% short" if (market_sentiment or {}).get('pos_long_ratio') else 'N/A'} (size-weighted, more reliable)
- Open Interest: {f"{(market_sentiment or {}).get('open_interest', 0):,.0f}" if (market_sentiment or {}).get('open_interest') else 'N/A'}
NOTE: L/S ratio is contrarian — when >65% accounts are long, local tops often follow; <40% long = potential short squeeze

MULTI-TIMEFRAME CONTEXT (Top-Down: 1D → 4H → 1H → 15M):
{htf_ctx}

{mtf_ctx}

ADDITIONAL SIGNALS:
- VWAP distance: {indicators.get('vwap_dist', 0):+.3%} {'▲ above VWAP (bullish bias)' if indicators.get('vwap_dist', 0) > 0 else '▼ below VWAP (bearish bias)'}
- RSI Divergence: {'⚡ BULLISH DIVERGENCE — price fell but RSI rose → reversal signal' if indicators.get('bullish_div') else '⚡ BEARISH DIVERGENCE — price rose but RSI fell → reversal signal' if indicators.get('bearish_div') else 'none'}
- Funding Rate: {fr_note}

CANDLE HISTORY (last 30 {indicators.get('timeframe','') or ''} candles + key levels):
{candle_ctx}

CANDLE PATTERNS (last 3 candles): {list(patterns.keys()) if patterns else ['none detected']}

SENTIMENT:
- Fear & Greed: {sentiment.get('fear_greed', {}).get('value')} / 100 ({sentiment.get('fear_greed', {}).get('label')})
- Bias: {sentiment.get('sentiment_bias')}
- Headlines: {sentiment.get('headlines', [])[:3]}

CURRENT POSITION: {position_info}

PORTFOLIO:
- Available Margin (USDT): {portfolio.get('usdt', 0):.2f}
- Portfolio Value: {portfolio.get('total_value', 0):.2f}
- Daily PnL: {portfolio.get('daily_pnl', 0):+.2f} USDT
- Max leverage allowed: 15x

WHALE & SMART MONEY DATA:
- Composite Whale Signal: {wh.get('composite_bias', 'N/A')} (score: {wh.get('composite_score', 0):+.2f})

Top Traders (Bitget, contrarian):
  - Long/Short Ratio: {wh.get('top_trader_ratio', {}).get('ratio', 'N/A')} (long {wh.get('top_trader_ratio', {}).get('long_pct', 50):.1f}% / short {wh.get('top_trader_ratio', {}).get('short_pct', 50):.1f}%)
  - Bias: {wh.get('top_trader_ratio', {}).get('bias', 'N/A')} ← USE CONTRARIAN: if very_long → shorts may squeeze them

Large On-Exchange Trades (>{wh.get('large_trades', {}).get('threshold_usd', 200000):,} USD):
  - Whale Buys: {wh.get('large_trades', {}).get('whale_buys', 0)} trades ({wh.get('large_trades', {}).get('whale_buy_vol_usd', 0):,} USD)
  - Whale Sells: {wh.get('large_trades', {}).get('whale_sells', 0)} trades ({wh.get('large_trades', {}).get('whale_sell_vol_usd', 0):,} USD)
  - Whale Bias: {wh.get('large_trades', {}).get('whale_bias', 'N/A')}
  - Whale % of Volume: {wh.get('large_trades', {}).get('whale_pct_of_volume', 0):.1f}%

Liquidations (recent):
  - Long Liquidations: ${wh.get('liquidations', {}).get('long_liq_usd', 0):,} (longs got rekt)
  - Short Liquidations: ${wh.get('liquidations', {}).get('short_liq_usd', 0):,} (shorts got rekt)

News Sentiment (CryptoPanic):
  - Bias: {wh.get('news_sentiment', {}).get('bias', 'N/A')} (bull:{wh.get('news_sentiment',{}).get('bull',0)} bear:{wh.get('news_sentiment',{}).get('bear',0)})
  - Headlines: {wh.get('news_sentiment', {}).get('headlines', [])[:2]}

CoinMarketCap Data:
  - Fear & Greed: {wh.get('cmc',{}).get('fear_greed',{}).get('value','N/A')} / 100 ({wh.get('cmc',{}).get('fear_greed',{}).get('label','N/A')}) — source: CMC
  - CMC Signal: {wh.get('cmc',{}).get('cmc_signal',{}).get('bias','N/A')} (score: {wh.get('cmc',{}).get('cmc_signal',{}).get('score',0):+.2f})
  - Coin 1h change: {wh.get('cmc',{}).get('coin',{}).get('change_1h_pct','N/A')}%
  - Coin 24h change: {wh.get('cmc',{}).get('coin',{}).get('change_24h_pct','N/A')}%
  - Coin 7d change: {wh.get('cmc',{}).get('coin',{}).get('change_7d_pct','N/A')}%
  - Volume 24h: ${wh.get('cmc',{}).get('coin',{}).get('volume_24h_usd',0):,}
  - Volume change 24h: {wh.get('cmc',{}).get('coin',{}).get('volume_change_pct','N/A')}%
  - CMC Rank: #{wh.get('cmc',{}).get('coin',{}).get('cmc_rank','N/A')}
  - BTC Dominance: {wh.get('cmc',{}).get('global_market',{}).get('btc_dominance_pct','N/A')}% ({wh.get('cmc',{}).get('global_market',{}).get('btc_dom_signal','N/A')})
  - Total Market Cap 24h change: {wh.get('cmc',{}).get('global_market',{}).get('market_cap_change_24h_pct','N/A')}%

VALID ACTIONS:
{"- open_long: go long (only if no current position)" if not has_long and not has_short else ""}
{"- open_short: go short (only if no current position)" if not has_long and not has_short else ""}
{"- close_long: close the long position" if has_long else ""}
{"- close_short: close the short position" if has_short else ""}
- hold: do nothing

Make your decision. Return ONLY the JSON.
"""

    client = _get_client()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
        if raw.endswith("```"):
            raw = raw[:-3]

    decision = json.loads(raw.strip())
    decision["tokens_in"] = response.usage.input_tokens
    decision["tokens_out"] = response.usage.output_tokens
    return decision
