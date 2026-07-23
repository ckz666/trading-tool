import asyncio
import concurrent.futures
import pandas as pd
from datetime import datetime, date
from typing import Callable, Optional

from exchange.futures_client import FuturesClient
from exchange.binance_data import fetch_ohlcv_binance, fetch_funding_rate_history_binance
from exchange.market_scanner import get_trending_symbols
from trading.futures_paper import FuturesPaperEngine, MAINTENANCE_MARGIN
from trading.risk import RiskManager, RiskConfig
from ai.ml_signal import (
    predict, get_indicators, detect_market_structure, train as ml_train,
    _funding_to_series, build_features,
)
from ai.patterns import detect_patterns
from ai.vol_regime import classify_vol_regime
from ai.whale import fetch_news_sentiment
from ai.cmc import fetch_cmc_data
from ai.reddit_sentiment import fetch_reddit_sentiment
from exchange.liquidation_stream import get_collector as get_liquidation_collector
from notifications.telegram import notify_fire_and_forget
from trading.journal import get_journal

DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "HYPE/USDT"]


class AutoTrader:
    def __init__(
        self,
        symbols: list[str] = None,
        timeframe: str = "1h",
        interval_seconds: int = 300,
        engine: FuturesPaperEngine = None,
        risk_config: RiskConfig = None,
        max_leverage: int = 15,
        max_open_positions: int = 3,
        max_same_direction: int = 3,
        retrain_every_cycles: int = 24,
        min_claude_confidence: float = 0.65,
        min_ml_conf: float = 0.35,
        min_confluence: int = 9,   # scaled from 8/24 for the +3 News/CMC points added 2026-07-22 (max now 27)
        min_dir_precision: float = 0.30,
        max_stale_cycles: int = 12,
        stale_conf_floor: float = 0.15,
        # Soft-gate entry params (2026-07-23) — see _confluence_score/_ml_contribution
        # and the confluence-computation block in _run_symbol for the full rationale.
        # Validated via 192-combo walk-forward sweep + BTC/ETH/HYPE out-of-sample check
        # (project memory), TOP#4 candidate: min_confluence=4, hold_offset=2,
        # neutral_zone=1, ml_weight=1, skip_contra=True on the BACKTEST's smaller
        # confluence scale (~11 max/side). Live's _confluence_score has more criteria
        # (News/CMC/OI/CVD/L-S-ratio/funding-tiered/15M/squeeze — max ~26 after moving
        # ML's own 3 pts out into _ml_contribution) that were never part of that sweep,
        # so hold_offset/neutral_zone are scaled ~2.36x (26/11) as a REASONED ADAPTATION,
        # not a directly re-validated number — min_confluence itself is left at its
        # already-tuned live value above rather than rescaled from scratch. Run in
        # parallel paper-trading against the old hard-gate behavior before trusting
        # this as proven, per DeepSeek's condition #3 (see memory).
        hold_offset: int = 5,
        neutral_zone: int = 2,
        ml_weight: float = 1,
        skip_contra: bool = True,
        # Order-flow ML feature (2026-07-23, see ai/ml_signal.py::cvd_zscore_from_ohlcv
        # + project memory) — wired end-to-end but OFF by default: models were trained
        # without it (neutral 0.0 fallback matches that), and the model would need
        # retraining + the same walk-forward validation rigor as the soft-gate change
        # before this should influence real decisions. Flip on to start feeding a live
        # z-score of Bitget's cvd_ratio (ring-buffer-normalised, see _run_symbol) once
        # that validation has happened.
        use_cvd_feature: bool = False,
        # Injected by web/app.py: returns the TRUE combined equity across every
        # engine sharing the wallet (AutoTrader, Funding Harvest, Grid, Mean
        # Reversion, Pairs Trading), not just this engine's own cash+positions.
        # Without this, self.engine.portfolio_value() below is blind to capital
        # the OTHER engines have locked in their own open positions — that capital
        # is missing from shared_wallet.balance too (it's not "lost", it's just
        # held elsewhere), so risk checks against the narrow view can read a
        # portfolio as deeply underwater when it's actually fine (found live,
        # 2026-07-23: Mean Reversion held $1,677 in an open position, invisible
        # to AutoTrader's own view, which then saw a false 16% "daily loss" and
        # blocked all new entries — see project memory). None keeps the old
        # narrow (and now known-wrong-in-a-multi-engine-world) behaviour, for
        # standalone use/testing without the rest of the app wired up.
        portfolio_value_fn: Optional[Callable[[], float]] = None,
    ):
        self.symbols = list(symbols or DEFAULT_SYMBOLS)
        self.timeframe = timeframe
        self.interval = interval_seconds
        self.engine = engine or FuturesPaperEngine()
        self.portfolio_value_fn = portfolio_value_fn
        self.risk = RiskManager(risk_config or RiskConfig(), state_file="data/autotrader_risk_state.json")
        if self.risk.peak_equity <= 0:
            # First-ever boot (no prior risk-state file): seed with the wallet's
            # starting balance instead of 0, so the drawdown breaker reflects the
            # true since-inception drawdown immediately rather than "discovering"
            # a fake new peak at whatever the equity happens to be right now.
            self.risk.peak_equity = self.engine.wallet.initial_balance
        self.max_leverage = max_leverage
        self.max_open_positions = max_open_positions
        self.max_same_direction = max_same_direction
        self.running = False
        self._task: Optional[asyncio.Task] = None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self._position_lock = asyncio.Lock()   # prevents _monitor_loop + _run_symbol double-close

        self.min_claude_confidence: float = min_claude_confidence  # kept for API compat
        self.min_ml_conf: float = min_ml_conf
        self.min_confluence: int = min_confluence
        self.min_dir_precision: float = min_dir_precision
        self.hold_offset: int = hold_offset
        self.neutral_zone: int = neutral_zone
        self.ml_weight: float = ml_weight
        self.skip_contra: bool = skip_contra
        self.use_cvd_feature: bool = use_cvd_feature
        # A position rides its hard SL/TP even if the model's own conviction
        # in that direction decays to ~nothing, as long as it never flips to
        # an outright counter-signal — observed in prod as a TAIKO/USDT long
        # held ~40h through a slow drift to a -217 USDT stop-loss while
        # confidence sat at 0.00 the whole time. This closes it early instead.
        self.max_stale_cycles: int = max_stale_cycles
        self.stale_conf_floor: float = stale_conf_floor
        self.position_stale_cycles: dict[str, int] = {}
        self.claude_calls_saved: int = 0   # kept for API compat
        self.retrain_every: int = retrain_every_cycles
        self.monitor_interval: int = 5    # SL/TP/Liq check every 5 seconds
        self._monitor_task: Optional[asyncio.Task] = None
        self.live_prices: dict[str, float] = {}
        self.training_progress: dict[str, int] = {}
        self.log: list[dict] = []
        self.last_decisions: dict[str, dict] = {}
        self.last_ml_signals: dict[str, dict] = {}
        self.cycle_count: int = 0
        self.model_accuracy: dict[str, float] = {}
        self.model_dir_precision: dict[str, float] = {}
        self._funding_tick: int = 0
        self.last_retrain_cycle: int = 0
        self.last_retrain_ts: str = None
        self.next_retrain_cycle: int = retrain_every_cycles

        # Dynamic symbol discovery
        self.dynamic_symbols: bool = True
        self.symbol_refresh_cycles: int = 48   # every 48 cycles ≈ 4h at 5min
        self.max_symbols: int = 10
        self.anchor_symbols: list[str] = ["BTC/USDT", "ETH/USDT", "HYPE/USDT"]
        self._symbol_blocklist: set[str] = set()
        self.trending_data: list[dict] = []            # latest scanner results
        self.last_symbol_refresh: str = None
        self._oi_buffer: dict[str, list] = {}         # {symbol: [oi_float, ...]} rolling 300 entries
        self._cvd_ratio_buffer: dict[str, list] = {}  # {symbol: [cvd_ratio, ...]} rolling 100 entries, see use_cvd_feature
        # News/CMC/Fear&Greed change on a macro timescale, not per 5-min cycle —
        # cached per symbol to avoid hammering RSS feeds and burning CMC's rate-
        # limited free-tier API credits every single cycle for every symbol.
        self._news_cmc_cache: dict[str, tuple[float, dict]] = {}
        self._news_cmc_ttl_seconds: int = 900   # 15 min
        # Reddit sentiment — same reasoning/TTL as news/CMC above, plus it's an
        # OAuth-token-limited API not meant to be hit every 5min per symbol.
        self._reddit_cache: dict[str, tuple[float, dict]] = {}
        self._reddit_ttl_seconds: int = 900     # 15 min

    # ── logging ───────────────────────────────────────────────────────────────
    def _log(self, level: str, msg: str, symbol: str = None, data: dict = None):
        entry = {
            "ts": datetime.now().isoformat(),
            "level": level,
            "symbol": symbol or "ALL",
            "msg": msg,
            **(data or {}),
        }
        self.log.append(entry)
        self.log = self.log[-500:]
        tag = f"[{symbol}] " if symbol else ""
        print(f"[{entry['ts'][11:19]}] [{level}] {tag}{msg}")
        if level == "TRADE":
            notify_fire_and_forget(f"🤖 <b>AutoTrader</b> {tag}\n{msg}")

    # ── model training ────────────────────────────────────────────────────────
    async def train_model(self, symbol: str, limit: int = 2000, client: FuturesClient = None) -> dict:
        self._log("INFO", f"Training ML model — {symbol} {self.timeframe} x{limit}", symbol)
        self.training_progress[symbol] = 0

        async def _fetch_funding(c: FuturesClient):
            return await c.fetch_funding_rate_history(symbol, 500)

        # Bitget's futures OHLCV history caps at ~1200 1h candles (~50 days) no
        # matter what's requested. Binance Futures has the same pairs with ~4x
        # more history (verified: ~208 days) — use it for training data only;
        # live execution, position management and funding stay on Bitget (the
        # actual trading venue). Falls back to Bitget for symbols too new/small
        # to be listed on Binance yet.
        #
        # Funding history: same reasoning applies and then some — Bitget caps
        # at ~100 records (~33 days), which left funding_norm/funding_trend
        # constant (0.0 feature importance, verified 2026-07-22) over most of
        # any longer training window. Binance funding history goes back much
        # further, so it's fetched from there whenever the OHLCV came from
        # there too.
        ohlcv, funding_raw = await asyncio.gather(
            fetch_ohlcv_binance(symbol, self.timeframe, limit),
            fetch_funding_rate_history_binance(symbol, 1000),
        )

        # Reuse a shared client when training a batch (train_all) so ccxt's
        # load_markets() happens once, not once per symbol — firing it
        # concurrently per-symbol is what was tripping Bitget's rate limit.
        if not ohlcv:
            self._log("WARN", f"{symbol} not found on Binance — training on Bitget history instead", symbol)
            async def _fetch_bitget(c: FuturesClient):
                return await asyncio.gather(
                    c.fetch_ohlcv(symbol, self.timeframe, limit),
                    _fetch_funding(c),
                )
            if client is not None:
                ohlcv, funding_raw = await _fetch_bitget(client)
            else:
                async with FuturesClient() as c:
                    ohlcv, funding_raw = await _fetch_bitget(c)
        df = _to_df(ohlcv)
        funding_series = _funding_to_series(funding_raw)

        def _progress(pct: int):
            self.training_progress[symbol] = pct
            self._log("INFO", f"Training {pct}%", symbol)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._executor, lambda: ml_train(df, symbol, progress_cb=_progress, funding_series=funding_series)
        )
        self.training_progress[symbol] = 100
        self.model_accuracy[symbol] = result["accuracy"]
        self.model_dir_precision[symbol] = result.get("dir_precision", 0.0)
        self._log("INFO",
            f"ML ready — bal_acc {result['accuracy']*100:.1f}% | dir_prec {result.get('dir_precision',0)*100:.1f}% | f1 {result.get('f1_macro',0):.2f} | {result['samples']} samples",
            symbol)
        return result

    async def train_all(self, limit: int = 3000, symbols: list = None) -> list:
        targets = symbols if symbols is not None else self.symbols
        async with FuturesClient() as client:
            return await asyncio.gather(*[self.train_model(sym, limit, client) for sym in targets])

    # ── single symbol cycle ───────────────────────────────────────────────────
    async def _run_symbol(self, client: FuturesClient, symbol: str):
        try:
            # These wrappers normalise both raised exceptions AND the explicit
            # None-on-failure now returned by fetch_funding_rate/fetch_cvd/
            # fetch_current_oi (2026-07-23 fix, audit finding H-05 — those used
            # to return fake plausible-looking data like {"rate": 0.0001} or
            # {"cvd_ratio": 0.5} on failure, indistinguishable from a real
            # reading) down to the same "no data" empty/zero value this code
            # already treats as neutral. Confluence-score criteria still fall
            # back to their own defaults on missing keys either way — closing
            # that loop fully is a separate, larger follow-up (see project memory).
            async def _safe_funding_history():
                try:
                    return await asyncio.wait_for(client.fetch_funding_rate_history(symbol, 100), timeout=10)
                except Exception:
                    return []

            async def _safe_funding_rate():
                try:
                    fr = await asyncio.wait_for(client.fetch_funding_rate(symbol), timeout=10)
                    return fr if fr is not None else {}
                except Exception:
                    return {}

            async def _safe_sentiment():
                try:
                    return await asyncio.wait_for(client.fetch_market_sentiment(symbol), timeout=10)
                except Exception:
                    return {}

            async def _safe_cvd():
                try:
                    cvd = await asyncio.wait_for(client.fetch_cvd(symbol, 500), timeout=10)
                    return cvd if cvd is not None else {}
                except Exception:
                    return {}

            async def _safe_oi_current():
                try:
                    oi = await asyncio.wait_for(client.fetch_current_oi(symbol), timeout=6)
                    return oi if oi is not None else 0.0
                except Exception:
                    return 0.0

            async def _safe_news_cmc():
                cached = self._news_cmc_cache.get(symbol)
                now = datetime.now().timestamp()
                if cached and (now - cached[0]) < self._news_cmc_ttl_seconds:
                    return cached[1]
                try:
                    news, cmc = await asyncio.wait_for(
                        asyncio.gather(fetch_news_sentiment(symbol), fetch_cmc_data(symbol)),
                        timeout=12,
                    )
                except Exception:
                    news, cmc = {}, {}
                result = {"news": news, "cmc": cmc}
                self._news_cmc_cache[symbol] = (now, result)
                return result

            async def _safe_reddit():
                cached = self._reddit_cache.get(symbol)
                now = datetime.now().timestamp()
                if cached and (now - cached[0]) < self._reddit_ttl_seconds:
                    return cached[1]
                try:
                    result = await asyncio.wait_for(fetch_reddit_sentiment(symbol), timeout=12)
                except Exception:
                    result = {"available": False}
                self._reddit_cache[symbol] = (now, result)
                return result

            ohlcv, ohlcv_4h, ohlcv_1d, ohlcv_15m, fr, funding_raw, market_sentiment, cvd_data, oi_now, news_cmc, reddit = await asyncio.gather(
                client.fetch_ohlcv(symbol, self.timeframe, 300),
                client.fetch_ohlcv(symbol, "4h", 100),
                client.fetch_ohlcv(symbol, "1d", 100),
                client.fetch_ohlcv(symbol, "15m", 200),
                _safe_funding_rate(),
                _safe_funding_history(),
                _safe_sentiment(),
                _safe_cvd(),
                _safe_oi_current(),
                _safe_news_cmc(),
                _safe_reddit(),
            )
            # Liquidation flow: read-only from the always-running background
            # WebSocket collector (exchange/liquidation_stream.py) — no network
            # call here, so no caching needed, unlike the fetch-based sources above.
            liq_flow = get_liquidation_collector().flow_score(symbol, window_hours=1.0)

            # ── OI buffer: accumulate per-cycle snapshots, compute deltas ──
            buf = self._oi_buffer.setdefault(symbol, [])
            if oi_now > 0:
                buf.append(oi_now)
                if len(buf) > 300:
                    buf.pop(0)
            # 1 cycle = 5min → 12 cycles ≈ 1H, 48 ≈ 4H, 288 ≈ 24H
            def _oi_delta(n):
                if len(buf) >= n + 1 and buf[-n - 1] > 0:
                    return (buf[-1] - buf[-n - 1]) / buf[-n - 1]
                return 0.0

            market_sentiment["oi_4h_delta"]  = _oi_delta(48)
            market_sentiment["oi_24h_delta"] = _oi_delta(288)
            market_sentiment["oi_current"]   = buf[-1] if buf else 0.0
            cvd_ratio_now = cvd_data.get("cvd_ratio", 0.5)
            market_sentiment["cvd_ratio"]    = cvd_ratio_now
            market_sentiment["cvd_net"]      = cvd_data.get("cvd_net", 0.0)

            # ── CVD ring buffer → rolling z-score, feeds the ML cvd_zscore feature
            # when use_cvd_feature is on (see ai/ml_signal.py::cvd_zscore_from_ohlcv
            # docstring for why this needs to be z-scored rather than used raw).
            cvd_buf = self._cvd_ratio_buffer.setdefault(symbol, [])
            cvd_buf.append(cvd_ratio_now)
            if len(cvd_buf) > 100:
                cvd_buf.pop(0)
            cvd_zscore_live = 0.0
            if len(cvd_buf) >= 20:
                cvd_arr = pd.Series(cvd_buf)
                cvd_std = cvd_arr.std()
                if cvd_std > 0:
                    cvd_zscore_live = float((cvd_arr.iloc[-1] - cvd_arr.mean()) / cvd_std)

            # News headlines (RSS-scraped) + CoinMarketCap (Fear&Greed, global
            # market, coin-specific momentum) — cached up to 15min, see _safe_news_cmc.
            news = news_cmc.get("news", {})
            cmc  = news_cmc.get("cmc", {})
            market_sentiment["news_bias"] = news.get("bias", "unavailable")
            market_sentiment["news_bull"] = news.get("bull", 0)
            market_sentiment["news_bear"] = news.get("bear", 0)
            if cmc.get("available"):
                fg = cmc.get("fear_greed", {})
                market_sentiment["fear_greed_value"] = fg.get("value")
                market_sentiment["fear_greed_bias"]  = fg.get("bias", "neutral")
                cmc_sig = cmc.get("cmc_signal", {})
                market_sentiment["cmc_signal_score"] = cmc_sig.get("score", 0.0)
                market_sentiment["cmc_signal_bias"]  = cmc_sig.get("bias", "neutral")
            if reddit.get("available"):
                market_sentiment["reddit_bias"] = reddit.get("bias", "neutral")
                market_sentiment["reddit_bull"] = reddit.get("bull_mentions", 0)
                market_sentiment["reddit_bear"] = reddit.get("bear_mentions", 0)
            if liq_flow.get("available"):
                market_sentiment["liq_dominant_side"] = liq_flow.get("dominant_side")
                market_sentiment["liq_dominance_ratio"] = liq_flow.get("dominance_ratio", 1.0)
                market_sentiment["liq_long_usd"] = liq_flow.get("long_liq_usd", 0)
                market_sentiment["liq_short_usd"] = liq_flow.get("short_liq_usd", 0)
            df     = _to_df(ohlcv)
            df_4h  = _to_df(ohlcv_4h)
            df_1d  = _to_df(ohlcv_1d)
            df_15m = _to_df(ohlcv_15m)
            funding_series = _funding_to_series(funding_raw)
            market_sentiment["funding_rate"] = fr.get("rate", 0)
            price = df["close"].iloc[-1]
            prices = {symbol: price}

            # ── funding tick ──
            if self._funding_tick % 8 == 0:
                self.engine.apply_funding(symbol, price)

            # ── SL / TP / Liquidation ──
            async with self._position_lock:
                trigger = self.engine.check_sl_tp_liquidation(symbol, price)
                if trigger == "take_profit_1":
                    # Partial close 50% at TP1, move SL to breakeven
                    pos_d  = self.engine.get_position(symbol, price)
                    record = self._partial_close(symbol, price, 0.5, "take_profit_1")
                    # move SL to breakeven so the rest of the trade is risk-free
                    cur_pos = self.engine.positions.get(symbol)
                    if cur_pos:
                        cur_pos.stop_loss = cur_pos.entry_price
                    self._log("TRADE",
                        f"PARTIAL TP1 — 50% geschlossen @ ${price:,.2f} | PnL: {record['pnl']:+.2f} USDT | SL → Breakeven ({cur_pos.entry_price:,.2f})",
                        symbol, {"type": "take_profit_1", **record})
                    # continue to let the remaining position run (don't return)
                elif trigger:
                    pos_d  = self.engine.get_position(symbol, price)
                    record = self._close(symbol, price, trigger)
                    self._log("TRADE",
                        f"{trigger.upper()} — closed {pos_d['side'].upper()} @ ${price:,.2f} | PnL: {record['pnl']:+.2f} USDT | ROE: {record['roe_pct']:+.1f}%",
                        symbol, {"type": trigger, **record})
                    return

            # ── risk checks (daily loss + drawdown) ──
            # Deliberately the TRUE cross-engine equity when available (see
            # portfolio_value_fn docstring above) — a 5%/15% loss breaker should
            # reflect the whole account, not just what AutoTrader itself is
            # holding, or it can trip on money that's simply parked in another
            # engine's position rather than actually lost.
            risk_portfolio_value = (self.portfolio_value_fn() if self.portfolio_value_fn
                                     else self.engine.portfolio_value(self.live_prices | {symbol: price}))
            open_count  = len(self.engine.positions)
            has_position = symbol in self.engine.positions

            if not has_position:   # only block NEW entries, not position management
                if not self.risk.check_daily_loss(risk_portfolio_value):
                    self._log("WARN", f"Blocked (daily loss): {self.risk.block_reason}", symbol)
                    return
                if not self.risk.check_drawdown(risk_portfolio_value):
                    self._log("WARN", f"Blocked (drawdown): {self.risk.block_reason}", symbol)
                    return
            else:
                # Still update peak equity tracking even when managing existing positions
                self.risk.check_drawdown(risk_portfolio_value)

            # ── max positions check ──
            if open_count >= self.max_open_positions and not has_position:
                self._log("INFO", f"Max positions ({self.max_open_positions}) reached — skipping new entry", symbol)
                return

            # ── ML signal ──
            # cvd_zscore feature only fed when use_cvd_feature is on (default off,
            # see constructor docstring) — models were trained without it, so the
            # neutral 0.0 fallback in build_features() is what they expect otherwise.
            loop = asyncio.get_event_loop()
            def _predict_this_bar():
                feats = None
                if self.use_cvd_feature:
                    cvd_series = pd.Series(cvd_zscore_live, index=df.index)
                    feats = build_features(df, funding_series=funding_series,
                                            precomputed_cvd_zscore=cvd_series)
                return predict(df, symbol, funding_series=funding_series, features=feats)
            ml_signal = await loop.run_in_executor(self._executor, _predict_this_bar)
            indicators_quick = get_indicators(df)
            ml_signal["regime"] = indicators_quick.get("regime", "unknown")
            ml_signal["adx"]    = indicators_quick.get("adx", 0)
            self.last_ml_signals[symbol] = ml_signal

            has_position = symbol in self.engine.positions
            conf         = ml_signal["confidence"]
            label        = ml_signal["label"]   # "buy" | "sell" | "hold"

            # ── MTF indicators (1D + 15M) ─────────────────────────────────────
            # >30, not >20: ADX needs >=2x its window (28 for window=14) or the ta
            # library indexes past the end of the series instead of returning NaN
            # (see ai/ml_signal.py::_resample_htf_indicators for the same guard) —
            # was inconsistent with ind_15m's threshold below, caused an IndexError
            # crash on thin-history symbols (e.g. SKHY/USDT, freshly listed).
            ind_1d  = get_indicators(df_1d)  if len(df_1d)  > 30 else None
            ind_15m = get_indicators(df_15m) if len(df_15m) > 30 else None

            # ── Confluence — bidirectional soft-gate (2026-07-23) ─────────────
            # Computed for BOTH directions unconditionally (not gated on the ML
            # label) so a strong technical signal can still open a trade when ML
            # says "hold", and so open-position management always has a real
            # is_long-conditioned score to compare against. See _confluence_score
            # and _ml_contribution docstrings + the constructor's soft-gate params
            # for the full rationale and validation history.
            patterns_quick = detect_patterns(df)
            score_long, reasons_long = self._confluence_score(
                True, indicators_quick, patterns_quick, df_4h,
                ind_1d=ind_1d, ind_15m=ind_15m, sentiment=market_sentiment,
            )
            score_short, reasons_short = self._confluence_score(
                False, indicators_quick, patterns_quick, df_4h,
                ind_1d=ind_1d, ind_15m=ind_15m, sentiment=market_sentiment,
            )
            ml_long, ml_short = self._ml_contribution(
                label, conf, ml_signal.get("agreement", 0.0), self.min_ml_conf, self.ml_weight
            )
            score_long  += ml_long
            score_short += ml_short
            confluence_score   = score_long if score_long >= score_short else score_short
            confluence_reasons = reasons_long if score_long >= score_short else reasons_short

            # ── Rule-based decision (no Claude) ──────────────────────────────────
            MIN_CONFLUENCE = self.min_confluence

            di_score   = ml_signal.get("di_score", 0.0)
            di_blocked = ml_signal.get("di_blocked", False)

            self._log("INFO",
                f"ML → {label.upper()} conf={conf:.2f} | DI={di_score:.2f} | "
                f"C_long={score_long} C_short={score_short}/28",
                symbol)

            action = "hold"
            skip_reason = None
            trade_mode = "trend"

            close_reason = "counter_signal"
            if has_position:
                # Already in a position — only manage via SL/TP (handled above)
                # Optionally close on strong counter-signal
                cur_pos = self.engine.positions.get(symbol)
                if cur_pos:
                    is_long = cur_pos.side == "long"
                    pos_confluence = score_long if is_long else score_short
                    counter = (is_long and label == "sell") or (not is_long and label == "buy")
                    supportive = (is_long and label == "buy") or (not is_long and label == "sell")
                    if counter and conf >= 0.65 and pos_confluence >= MIN_CONFLUENCE:
                        action = "close_long" if is_long else "close_short"
                        skip_reason = None
                    elif supportive and conf >= self.stale_conf_floor:
                        # model still backs this direction — reset the decay counter
                        self.position_stale_cycles[symbol] = 0
                        skip_reason = "position open — managed by SL/TP"
                    else:
                        # label is "hold", or points the right way but too weakly to
                        # count as supportive — the model has lost conviction without
                        # flipping to a hard counter-signal. Count consecutive cycles
                        # of this and exit early rather than riding to the hard SL/TP.
                        stale = self.position_stale_cycles.get(symbol, 0) + 1
                        self.position_stale_cycles[symbol] = stale
                        if stale >= self.max_stale_cycles:
                            action = "close_long" if is_long else "close_short"
                            close_reason = "confidence_decay"
                            skip_reason = None
                            self._log("INFO",
                                f"Confidence-Decay-Exit nach {stale} Zyklen ohne Modell-Rückhalt (conf={conf:.2f})",
                                symbol)
                        else:
                            skip_reason = "position open — managed by SL/TP"
            elif di_blocked:
                skip_reason = f"DI={di_score:.2f} — Regime-Shift, kein Entry"
            elif self.model_dir_precision.get(symbol, 0.0) < self.min_dir_precision:
                skip_reason = (
                    f"Modellqualität zu niedrig (dir_precision="
                    f"{self.model_dir_precision.get(symbol, 0.0):.0%} < {self.min_dir_precision:.0%})"
                )
            else:
                ml_was_hold     = (label == "hold")
                eff_min         = MIN_CONFLUENCE + (self.hold_offset if ml_was_hold else 0)
                best            = max(score_long, score_short)
                entry_direction = "long" if score_long >= score_short else "short"
                label_direction = "long" if label == "buy" else ("short" if label == "sell" else None)

                if best < eff_min:
                    skip_reason = f"C={best}/28 < {eff_min}" + (" (ML=hold — höhere Hürde)" if ml_was_hold else "")
                elif abs(score_long - score_short) < self.neutral_zone:
                    skip_reason = f"Neutralzone |{score_long}-{score_short}| < {self.neutral_zone}"
                elif self.skip_contra and label_direction and entry_direction != label_direction:
                    skip_reason = f"konträr zu ML-Label ({label}) — Skip-Filter aktiv"
                elif self._same_direction_count("buy" if entry_direction == "long" else "sell") >= self.max_same_direction:
                    skip_reason = (
                        f"Richtungslimit erreicht ({entry_direction}: "
                        f"{self._same_direction_count('buy' if entry_direction == 'long' else 'sell')}/{self.max_same_direction}) — Korrelationsrisiko"
                    )
                else:
                    action = "open_long" if entry_direction == "long" else "open_short"
                    confluence_score   = best
                    confluence_reasons = reasons_long if entry_direction == "long" else reasons_short
                    if ml_was_hold:
                        # keep `label` in sync with the direction we're actually
                        # trading so downstream logging/_same_direction_count stay
                        # consistent for entries the soft-gate opened without an
                        # ML opinion (label was "hold" going into this branch)
                        label = "buy" if entry_direction == "long" else "sell"

            # ── Scalp fallback: trend path declined, but the market is ranging
            # (ADX < 20 → no real trend) — try mean-reversion instead of sitting
            # idle. Independent of ML confidence/confluence, so it can fire even
            # when the trend model says HOLD.
            if skip_reason and not has_position and indicators_quick.get("regime") == "ranging":
                scalp_label, scalp_reason = self._scalp_signal(indicators_quick)
                if scalp_label != "hold" and self._same_direction_count(scalp_label) < self.max_same_direction:
                    action        = "open_long" if scalp_label == "buy" else "open_short"
                    skip_reason   = None
                    trade_mode    = "scalp"
                    label         = scalp_label
                    confluence_reasons = [scalp_reason]

            if skip_reason:
                decision = {
                    "action": "hold", "confidence": conf,
                    "reasoning": skip_reason,
                    "confluence_score": confluence_score,
                    "confluence_reasons": confluence_reasons,
                    "di_score": di_score,
                    "ts": datetime.now().isoformat(), "price": price,
                }
                self.last_decisions[symbol] = decision
                return

            # ── ATR-based SL/TP from indicators ──────────────────────────────────
            indicators_full = indicators_quick
            atr = indicators_full.get("atr", price * 0.015)
            if trade_mode == "scalp":
                # ranging market — smaller targets sized to the range, not a trend leg
                sl_pct   = max((atr * 0.5) / price, 0.003)   # 0.5x ATR, min 0.3%
                tp_pct   = atr * 1.0 / price                  # 1.0x ATR (R:R 1:2)
                tp1_pct  = atr * 0.5 / price                  # TP1 at 0.5x ATR (partial close)
                leverage = self._scaled_leverage(self.min_confluence, atr / price)
            else:
                sl_pct   = max((atr * 1.5) / price, 0.005)   # 1.5x ATR, min 0.5%
                tp_pct   = atr * 3.0 / price                  # 3.0x ATR (R:R 1:2)
                tp1_pct  = atr * 1.5 / price                  # TP1 at 1.5x ATR (partial close)
                leverage = self._scaled_leverage(confluence_score, atr / price)

            # Stop-loss must trigger before liquidation, or it never fires — the position
            # rides to near-total margin loss instead of the sized risk_amount. Cap the SL
            # distance to 80% of the leverage's liquidation buffer.
            liq_buffer_pct = 1 / leverage - MAINTENANCE_MARGIN
            sl_pct   = min(sl_pct, liq_buffer_pct * 0.8)

            trailing_sl = True
            trail_pct   = sl_pct

            reasons_str = " | ".join(confluence_reasons[:4])
            mode_tag = "SCALP " if trade_mode == "scalp" else ""
            self._log("INFO",
                f"SIGNAL {mode_tag}{action.upper()} | C={confluence_score}/28 | conf={conf:.2f} | SL={sl_pct:.1%} TP={tp_pct:.1%} | {reasons_str}",
                symbol)

            reasoning = (reasons_str if trade_mode == "scalp" else
                         f"Rule: C={confluence_score}/28 ≥ {MIN_CONFLUENCE}. {reasons_str}")
            decision = {
                "action": action, "confidence": conf, "mode": trade_mode,
                "reasoning": reasoning,
                "leverage": leverage, "stop_loss_pct": sl_pct, "take_profit_pct": tp_pct,
                "tp1_pct": tp1_pct, "trailing_sl": trailing_sl, "trail_pct": trail_pct,
                "confluence_score": confluence_score, "confluence_reasons": confluence_reasons,
                "di_score": di_score, "ts": datetime.now().isoformat(), "price": price,
            }
            self.last_decisions[symbol] = decision

            # ── ATR-based position sizing (Kelly-scaled risk-per-trade, ─────────
            #    de-rated in a STORM volatility regime) ──────────────────────────
            equity      = self.engine.portfolio_value(self.live_prices | {symbol: price})
            risk_pct    = self._kelly_risk_pct(symbol)
            vol_regime  = classify_vol_regime(df)
            risk_pct    = risk_pct * vol_regime["risk_multiplier"]
            risk_amount = equity * risk_pct
            sl_dist     = price * sl_pct         # stop-loss distance in USDT
            sl_dist     = max(sl_dist, price * 0.005)  # minimum 0.5%

            # base amount so that a full SL hit costs exactly risk_amount
            raw_amount  = risk_amount / sl_dist
            max_margin  = equity * self.risk.config.max_position_pct
            cap_amount  = (max_margin * leverage) / price
            amount      = min(raw_amount, cap_amount)
            margin_use  = (amount * price) / leverage

            # ── execute ──
            cur_pos = self.engine.positions.get(symbol)

            if action == "open_long" and not cur_pos:
                self.position_stale_cycles[symbol] = 0
                # Real per-symbol maintenance margin rate instead of a fixed 0.5%
                # guess for every symbol (audit finding H-01, 2026-07-23, see
                # project memory) — matters most for this bot's thinly-traded
                # small-cap symbols, which can have meaningfully different tiers
                # than BTC/ETH. Cached 1h, falls back to the old default on failure.
                mmr = await client.fetch_maintenance_margin_rate(symbol, notional=amount * price)
                record = self.engine.open_position(
                    symbol, "long", amount, price, leverage,
                    price * (1 - sl_pct), price * (1 + tp_pct),
                    trailing_sl=trailing_sl, trail_pct=trail_pct, mode=trade_mode,
                    maintenance_margin=mmr)
                regime_tag = f" | {vol_regime['regime'].upper()}" if vol_regime["regime"] == "storm" else ""
                self._log("TRADE",
                    f"OPEN {mode_tag}LONG {amount:.6f} @ ${price:,.2f} | {leverage}x | Margin ${margin_use:.0f} | Risk ${risk_amount:.0f} ({risk_pct:.2%}){regime_tag} | Liq ${record['liq_price']:,.0f}",
                    symbol, {"type": "open_long", **record})
                get_journal().record("autotrader", symbol, "open_long", reasoning)

            elif action == "open_short" and not cur_pos:
                self.position_stale_cycles[symbol] = 0
                mmr = await client.fetch_maintenance_margin_rate(symbol, notional=amount * price)
                record = self.engine.open_position(
                    symbol, "short", amount, price, leverage,
                    price * (1 + sl_pct), price * (1 - tp_pct),
                    trailing_sl=trailing_sl, trail_pct=trail_pct, mode=trade_mode,
                    maintenance_margin=mmr)
                regime_tag = f" | {vol_regime['regime'].upper()}" if vol_regime["regime"] == "storm" else ""
                self._log("TRADE",
                    f"OPEN {mode_tag}SHORT {amount:.6f} @ ${price:,.2f} | {leverage}x | Margin ${margin_use:.0f} | Risk ${risk_amount:.0f} ({risk_pct:.2%}){regime_tag} | Liq ${record['liq_price']:,.0f}",
                    symbol, {"type": "open_short", **record})
                get_journal().record("autotrader", symbol, "open_short", reasoning)

            elif action == "close_long" and cur_pos and cur_pos.side == "long":
                record = self._close(symbol, price, close_reason)
                self.position_stale_cycles.pop(symbol, None)
                self._log("TRADE",
                    f"CLOSE LONG ({close_reason}) @ ${price:,.2f} | PnL {record['pnl']:+.2f} USDT | ROE {record['roe_pct']:+.1f}%",
                    symbol, {"type": "close_long", **record})
                get_journal().record("autotrader", symbol, "close_long", close_reason, pnl=record["pnl"])

            elif action == "close_short" and cur_pos and cur_pos.side == "short":
                record = self._close(symbol, price, close_reason)
                self.position_stale_cycles.pop(symbol, None)
                self._log("TRADE",
                    f"CLOSE SHORT ({close_reason}) @ ${price:,.2f} | PnL {record['pnl']:+.2f} USDT | ROE {record['roe_pct']:+.1f}%",
                    symbol, {"type": "close_short", **record})
                get_journal().record("autotrader", symbol, "close_short", close_reason, pnl=record["pnl"])

            else:
                self._log("INFO", f"HOLD (action={action}, pos={'open' if cur_pos else 'none'})", symbol)

        except Exception as e:
            self._log("ERROR", f"Cycle error: {e}", symbol)
            import traceback; traceback.print_exc()

    # ── fast monitor loop (SL/TP/Liq every 15s using live prices) ───────────
    async def _monitor_loop(self):
        while self.running:
            await asyncio.sleep(self.monitor_interval)
            if not self.engine.positions:
                continue
            prices = self.live_prices
            if not prices:
                continue
            try:
                for symbol in list(self.engine.positions.keys()):
                    price = prices.get(symbol)
                    if price is None:
                        continue
                    async with self._position_lock:
                        trigger = self.engine.check_sl_tp_liquidation(symbol, price)
                        if trigger == "take_profit_1":
                            record = self._partial_close(symbol, price, 0.5, "take_profit_1")
                            cur_pos = self.engine.positions.get(symbol)
                            if cur_pos:
                                cur_pos.stop_loss = cur_pos.entry_price
                            self._log("TRADE",
                                f"PARTIAL TP1 (monitor) — 50% @ ${price:,.2f} | PnL: {record['pnl']:+.2f} | SL → Breakeven",
                                symbol, {"type": "take_profit_1", **record})
                            get_journal().record("autotrader", symbol, "partial_take_profit_1",
                                                  "TP1 @ 1.5x ATR erreicht, 50% geschlossen, SL auf Breakeven",
                                                  pnl=record["pnl"])
                        elif trigger:
                            pos = self.engine.get_position(symbol, price)
                            record = self._close(symbol, price, trigger)
                            self._log("TRADE",
                                f"{trigger.upper()} (monitor) — closed {pos['side'].upper()} @ ${price:,.2f} | PnL: {record['pnl']:+.2f} USDT | ROE: {record['roe_pct']:+.1f}%",
                                symbol, {"type": trigger, **record})
                            get_journal().record("autotrader", symbol, f"close_{pos['side']}", trigger,
                                                  pnl=record["pnl"])
            except Exception as e:
                self._log("ERROR", f"Monitor-Loop Fehler: {e}", "ALL")

    # ── dynamic symbol discovery ──────────────────────────────────────────────
    async def _refresh_symbols(self):
        """Replace watchlist with top trending USDT-perp pairs, keeping anchors + open positions."""
        try:
            trending = await get_trending_symbols(top_n=self.max_symbols + 2, min_volume=20_000_000)
            self.trending_data = trending
            if not trending:
                return

            # Build new symbol list
            protected = set(self.anchor_symbols) | set(self.engine.positions.keys())
            new_syms = list(protected)
            for t in trending:
                sym = t["symbol"]
                if sym in self._symbol_blocklist:
                    continue
                if sym not in new_syms and len(new_syms) < self.max_symbols:
                    new_syms.append(sym)

            # Fill remaining slots with anchors if not already included
            for a in self.anchor_symbols:
                if a not in new_syms and len(new_syms) < self.max_symbols:
                    new_syms.append(a)

            added   = [s for s in new_syms if s not in self.symbols]
            removed = [s for s in self.symbols if s not in new_syms]
            self.symbols = new_syms
            self.last_symbol_refresh = datetime.now().isoformat()

            if added or removed:
                self._log("INFO",
                    f"Symbols aktualisiert — neu: {added} | entfernt: {removed} | aktiv: {self.symbols}")
                # Train models for newly added symbols
                if added:
                    await self.train_all(symbols=added)
            else:
                self._log("INFO", f"Symbols unverändert: {self.symbols}")
        except Exception as e:
            self._log("WARN", f"Symbol-Refresh fehlgeschlagen: {e}")

    # ── main loop ─────────────────────────────────────────────────────────────
    async def _loop(self):
        try:
            async with FuturesClient() as client:
                while self.running:
                    self.cycle_count += 1
                    self._log("INFO", f"══ Cycle #{self.cycle_count} — {len(self.symbols)} symbols ══")
                    self._funding_tick += 1

                    # ── dynamic symbol refresh (cycle_count==1 handled by startup()) ──
                    if self.dynamic_symbols and self.cycle_count % self.symbol_refresh_cycles == 0:
                        await self._refresh_symbols()

                    # ── auto-retrain ──
                    if self.retrain_every > 0 and self.cycle_count > 1 and self.cycle_count % self.retrain_every == 0:
                        self._log("INFO", f"Auto-retrain triggered (every {self.retrain_every} cycles)")
                        try:
                            await self.train_all()
                            self.last_retrain_cycle = self.cycle_count
                            self.last_retrain_ts    = datetime.now().isoformat()
                            self.next_retrain_cycle = self.cycle_count + self.retrain_every
                        except Exception as e:
                            # a transient fetch/rate-limit error here must not kill the whole
                            # trading loop — skip this retrain, keep the existing models
                            self._log("WARN", f"Auto-retrain fehlgeschlagen, behalte alte Modelle: {e}", "ALL")

                    await asyncio.gather(
                        *[self._run_symbol(client, sym) for sym in self.symbols],
                        return_exceptions=True,
                    )
                    # equity snapshot after each full cycle — use live_prices where available
                    all_prices = {s: self.last_decisions.get(s, {}).get("price", 0) for s in self.symbols}
                    all_prices.update(self.live_prices)
                    self.engine.record_equity(all_prices)
                    await asyncio.sleep(self.interval)
        except Exception as e:
            self._log("ERROR", f"_loop crashed — AutoTrader gestoppt: {e}", "ALL")
            import traceback; traceback.print_exc()
            self.running = False

    async def startup(self) -> list:
        """Scan for lucrative symbols, train all (including new), then start the loop."""
        if self.dynamic_symbols:
            self._log("INFO", "Startup-Scan: suche lukrative Symbole…")
            await self._refresh_symbols()
        results = await self.train_all()
        self.start()
        return results

    def start(self):
        if self.running:
            return
        self.running = True
        self._task         = asyncio.create_task(self._loop())
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        self._log("INFO", f"AutoTrader started — {self.symbols} | {self.timeframe} | every {self.interval}s | monitor every {self.monitor_interval}s | max {self.max_leverage}x | max {self.max_open_positions} positions")

    async def stop(self):
        self.running = False
        for t in (self._task, self._monitor_task):
            if t:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
        self._log("INFO", "AutoTrader stopped")

    # ── status ────────────────────────────────────────────────────────────────
    def _daily_pnl_from_engine(self) -> float:
        """Compute today's realised PNL from engine trade history (source of truth)."""
        today = date.today().isoformat()
        return sum(
            t.get("pnl", 0) or 0
            for t in self.engine.trade_history
            if (t.get("ts") or "")[:10] == today and t.get("pnl") is not None
        )

    def status(self) -> dict:
        # merge live_prices (5s) over last_decisions prices (up to 5min stale)
        last_prices = {sym: d.get("price", 0) for sym, d in self.last_decisions.items()}
        last_prices.update(self.live_prices)
        engine_status = self.engine.status(last_prices)

        # always compute daily_pnl from engine history (survives restarts)
        risk_status = self.risk.status()
        risk_status["daily_pnl"] = round(self._daily_pnl_from_engine(), 2)

        return {
            "running": self.running,
            "symbols": self.symbols,
            "timeframe": self.timeframe,
            "interval_seconds": self.interval,
            "max_leverage": self.max_leverage,
            "max_open_positions": self.max_open_positions,
            "max_same_direction": self.max_same_direction,
            "cycle_count": self.cycle_count,
            "retrain_every": self.retrain_every,
            "last_retrain_cycle": self.last_retrain_cycle,
            "last_retrain_ts": self.last_retrain_ts,
            "next_retrain_cycle": self.next_retrain_cycle,
            "min_claude_confidence": self.min_claude_confidence,
            "min_ml_conf": self.min_ml_conf,
            "min_confluence": self.min_confluence,
            "hold_offset": self.hold_offset,
            "neutral_zone": self.neutral_zone,
            "ml_weight": self.ml_weight,
            "skip_contra": self.skip_contra,
            "use_cvd_feature": self.use_cvd_feature,
            "min_dir_precision": self.min_dir_precision,
            "max_stale_cycles": self.max_stale_cycles,
            "stale_conf_floor": self.stale_conf_floor,
            "position_stale_cycles": self.position_stale_cycles,
            "claude_calls_saved": self.claude_calls_saved,
            "training_progress": self.training_progress,
            "model_accuracy": self.model_accuracy,
            "model_dir_precision": self.model_dir_precision,
            "last_decisions": self.last_decisions,
            "last_ml_signals": self.last_ml_signals,
            "risk": risk_status,
            "engine": engine_status,
            "trending_data": self.trending_data,
            "last_symbol_refresh": self.last_symbol_refresh,
            "dynamic_symbols": self.dynamic_symbols,
        }

    # ── close helpers (always sync risk.daily_pnl) ───────────────────────────
    def _close(self, symbol: str, price: float, reason: str) -> dict:
        record = self.engine.close_position(symbol, price, reason=reason)
        self.risk.daily_pnl += record["pnl"]
        return record

    def _partial_close(self, symbol: str, price: float, fraction: float, reason: str) -> dict:
        record = self.engine.partial_close_position(symbol, price, fraction, reason)
        self.risk.daily_pnl += record["pnl"]
        return record

    # ── Kelly-scaled position sizing ─────────────────────────────────────────
    def _trade_pnls(self, symbol: str = None) -> list[float]:
        return [
            t["pnl"] for t in self.engine.trade_history
            if t.get("pnl") is not None and (symbol is None or t.get("symbol") == symbol)
        ]

    def _scalp_signal(self, indicators: dict) -> tuple[str, str]:
        """Mean-reversion fallback for ranging markets (ADX < 20, no real trend):
        buy near the lower Bollinger band on RSI oversold, sell near the upper
        band on RSI overbought. Independent of the ML trend model — pure rule-based,
        since trend-following signals aren't meaningful when there's no trend."""
        rsi    = indicators.get("rsi", 50)
        bb_pct = indicators.get("bb_pct", 0.5)
        if rsi <= 32 and bb_pct <= 0.15:
            return "buy", f"Scalp: RSI überverkauft ({rsi:.0f}) nahe unterem BB-Band ({bb_pct:.2f})"
        if rsi >= 68 and bb_pct >= 0.85:
            return "sell", f"Scalp: RSI überkauft ({rsi:.0f}) nahe oberem BB-Band ({bb_pct:.2f})"
        return "hold", ""

    def _same_direction_count(self, label: str) -> int:
        """Count open positions on the same side as a prospective new entry.
        Symbols move together (esp. alts vs. BTC) — many positions all betting
        the same direction isn't diversification, it's one bet repeated."""
        side = "long" if label == "buy" else "short"
        return sum(1 for p in self.engine.positions.values() if p.side == side)

    def _scaled_leverage(self, confluence_score: int, atr_pct: float) -> int:
        """Scale leverage down for marginal signals and volatile symbols instead of
        always using max_leverage flat. Full leverage requires both strong confluence
        (>= 2x the min threshold) and below-average volatility (ATR <= 1.5% of price).
        """
        conviction = confluence_score / (self.min_confluence * 2) if self.min_confluence else 1.0
        conviction_factor = max(min(conviction, 1.0), 0.5)

        if atr_pct <= 0.015:
            vol_factor = 1.0
        elif atr_pct >= 0.03:
            vol_factor = 0.5
        else:
            vol_factor = 1.0 - 0.5 * (atr_pct - 0.015) / (0.03 - 0.015)

        scaled = self.max_leverage * conviction_factor * vol_factor
        return max(int(round(scaled)), 2)

    def _kelly_risk_pct(self, symbol: str, default_pct: float = 0.015) -> float:
        """Half-Kelly risk-per-trade from realised PnL history (per-symbol, else portfolio-wide).
        Falls back to a fixed 1.5% until enough trade history exists to trust the estimate.
        """
        sym_pnls = self._trade_pnls(symbol)
        pnls = sym_pnls if len(sym_pnls) >= 20 else self._trade_pnls()
        kelly_pct = self.risk.kelly_risk_pct(pnls)
        return kelly_pct if kelly_pct is not None else default_pct

    # ── confluence score ──────────────────────────────────────────────────────
    @staticmethod
    def _ml_contribution(label: str, conf: float, agreement: float,
                          min_conf: float, ml_weight: float) -> tuple[int, int]:
        """
        ML's own weighted vote — added on top of the (direction-independent)
        confluence score below instead of hard-gating it. Confidence only gates
        whether ML has a usable opinion at all (proven flat/uninformative in
        [0, 0.40], see ML diagnosis 2026-07-22) — NOT used to scale the point
        size. Agreement (fraction of the 3-model ensemble voting the same way)
        sets the magnitude instead. Mirrors ai/backtest.py::_ml_contribution /
        ai/sweep.py::_ml_points — same design, kept as a separate copy per this
        repo's live/backtest mirroring convention (see Kelly/Ichimoku).
        Returns (points_to_long, points_to_short) — 0/0 when ML has no usable opinion.
        """
        if label == "hold" or conf < min_conf or agreement < 0.67:
            return 0, 0
        pts = ml_weight * (2 if agreement >= 0.99 else 1)
        return (pts, 0) if label == "buy" else (0, pts)

    def _confluence_score(
        self,
        is_long: bool,
        indicators: dict,
        patterns: dict,
        df_4h,
        ind_1d:  dict = None,
        ind_15m: dict = None,
        sentiment: dict = None,
    ) -> tuple[int, list[str]]:
        """
        Score across 1D/4H/1H/15M + on-chain + flow + macro/news signals that support
        the given direction (~26 max between this and _ml_contribution combined).
        Direction-independent (soft-gate, 2026-07-23): caller decides is_long and calls
        this once per side per bar so both directions can be compared even when the ML
        label is 'hold' — see the confluence-computation block in _run_symbol and
        _ml_contribution for how ML itself weighs in.
        Layers: 1H-RSI(1) + 1H-MACD(1) + 1H-EMA(1) + VWAP(1) + 4H-structure(2) +
                Candle(1) + 1D-trend(2) + 15M-mom(2) + Squeeze(2) + Funding-tiered(2)
                + L/S-ratio(1) + OI-delta(2) + CVD(2) + Ichimoku(2) + News(1) + CMC-composite(2)
        """
        score   = 0
        reasons = []

        # 1. RSI positioning (0-1 pt)
        rsi = indicators.get("rsi", 50)
        if is_long and rsi < 42:
            score += 1; reasons.append(f"RSI überverkauft ({rsi:.0f})")
        elif not is_long and rsi > 58:
            score += 1; reasons.append(f"RSI überkauft ({rsi:.0f})")

        # 4. MACD (0-1 pt)
        macd = indicators.get("macd_diff", 0)
        if is_long and macd > 0:
            score += 1; reasons.append("MACD bullisch")
        elif not is_long and macd < 0:
            score += 1; reasons.append("MACD bärisch")

        # 5. EMA cross (0-1 pt)
        ema = indicators.get("ema_cross_norm", 0)
        if is_long and ema > 0:
            score += 1; reasons.append("EMA9 über EMA21")
        elif not is_long and ema < 0:
            score += 1; reasons.append("EMA9 unter EMA21")

        # 6. VWAP position (0-1 pt)
        vwap = indicators.get("vwap_dist", 0)
        if is_long and vwap > 0.001:
            score += 1; reasons.append("Über VWAP")
        elif not is_long and vwap < -0.001:
            score += 1; reasons.append("Unter VWAP")

        # 7. Market structure from 4h (0-2 pts)
        try:
            if df_4h is not None and len(df_4h) > 12:
                ms = detect_market_structure(df_4h)
                trend = ms.get("trend", "unknown")
                if is_long and trend == "uptrend":
                    score += 2; reasons.append("4H: HH/HL Aufwärtsstruktur")
                elif not is_long and trend == "downtrend":
                    score += 2; reasons.append("4H: LL/LH Abwärtsstruktur")
                elif is_long and trend == "sideways":
                    score -= 1; reasons.append("4H: Seitwärts (kein Rückenwind)")
                elif not is_long and trend == "sideways":
                    score -= 1; reasons.append("4H: Seitwärts (kein Rückenwind)")
        except Exception:
            pass

        # 8. Candle pattern alignment (0-1 pt)
        for pname, pinfo in (patterns or {}).items():
            ptype = pinfo if isinstance(pinfo, str) else pinfo.get("type", "")
            if is_long and ptype == "bullish":
                score += 1; reasons.append(f"Muster: {pname}")
                break
            elif not is_long and ptype == "bearish":
                score += 1; reasons.append(f"Muster: {pname}")
                break

        # 9. Daily trend (0-2 pts, -1 if counter-trend)
        if ind_1d:
            d_ema  = ind_1d.get("ema_cross_norm", 0)
            d_rsi  = ind_1d.get("rsi", 50)
            d_macd = ind_1d.get("macd_diff", 0)
            d_bull  = d_ema > 0 and d_rsi > 50
            d_bear  = d_ema < 0 and d_rsi < 50
            d_str  = f"EMA {'▲' if d_ema>0 else '▼'} RSI {d_rsi:.0f}"
            if is_long:
                if d_bull:
                    score += 2; reasons.append(f"1D bullisch ({d_str})")
                elif d_macd > 0 or d_rsi > 50:
                    score += 1; reasons.append(f"1D neutral-bullisch ({d_str})")
                elif d_bear:
                    score -= 1; reasons.append(f"1D Gegentrend bärisch ({d_str}) ⚠")
            else:
                if d_bear:
                    score += 2; reasons.append(f"1D bärisch ({d_str})")
                elif d_macd < 0 or d_rsi < 50:
                    score += 1; reasons.append(f"1D neutral-bärisch ({d_str})")
                elif d_bull:
                    score -= 1; reasons.append(f"1D Gegentrend bullisch ({d_str}) ⚠")

        # 10. 15M momentum (0-2 pts)
        if ind_15m:
            m_macd = ind_15m.get("macd_diff", 0)
            m_rsi  = ind_15m.get("rsi", 50)
            m_ema  = ind_15m.get("ema_cross_norm", 0)
            if is_long:
                if m_macd > 0: score += 1; reasons.append(f"15M MACD bullisch")
                if m_rsi > 50 and m_ema > 0: score += 1; reasons.append(f"15M RSI/EMA bullisch ({m_rsi:.0f})")
            else:
                if m_macd < 0: score += 1; reasons.append(f"15M MACD bärisch")
                if m_rsi < 50 and m_ema < 0: score += 1; reasons.append(f"15M RSI/EMA bärisch ({m_rsi:.0f})")

        # 11. Volatility Squeeze fired — compression release (0-2 pts)
        sq_fired    = indicators.get("squeeze_fired", 0)
        sq_momentum = indicators.get("squeeze_momentum", 0)
        sq_active   = indicators.get("squeeze_active", 0)
        if sq_fired:
            if is_long and sq_momentum > 0:
                score += 2; reasons.append(f"Squeeze gefeuert — bullisches Momentum ({sq_momentum*100:+.2f}%)")
            elif not is_long and sq_momentum < 0:
                score += 2; reasons.append(f"Squeeze gefeuert — bärisches Momentum ({sq_momentum*100:+.2f}%)")
            else:
                # fired but momentum opposes ML direction
                score -= 1; reasons.append(f"Squeeze gefeuert gegen ML-Richtung ⚠ (mom={sq_momentum*100:+.2f}%)")
        elif sq_active:
            reasons.append("Squeeze aktiv — Kompression läuft (kein Punkt, aber Vorsicht)")

        # 12. Funding rate — tiered contrarian (0-2 pts, or -1 counter)
        if sentiment:
            fund_rate = sentiment.get("funding_rate")
            if fund_rate is not None:
                if not is_long:
                    if fund_rate > 0.001:    # > 10x normal — extreme crowded longs
                        score += 2; reasons.append(f"Funding EXTREM positiv ({fund_rate*100:.3f}%) — Longs massiv überhitzt")
                    elif fund_rate > 0.0003: # > 3x normal — elevated
                        score += 1; reasons.append(f"Funding hoch positiv ({fund_rate*100:.3f}%) — Longs überhitzt")
                    elif fund_rate < -0.0002:
                        score -= 1; reasons.append(f"Funding negativ ({fund_rate*100:.3f}%) — Gegenwind für Short ⚠")
                else:
                    if fund_rate < -0.001:   # extreme negative — shorts crowded
                        score += 2; reasons.append(f"Funding EXTREM negativ ({fund_rate*100:.3f}%) — Shorts massiv überhitzt")
                    elif fund_rate < -0.0001:
                        score += 1; reasons.append(f"Funding negativ ({fund_rate*100:.3f}%) — Shorts dominieren")
                    elif fund_rate > 0.0005:
                        score -= 1; reasons.append(f"Funding extrem ({fund_rate*100:.3f}%) — Gegenwind für Long ⚠")

            # 13. Long/Short ratio — contrarian (0-1 pt)
            long_ratio = sentiment.get("long_ratio")
            if long_ratio is not None:
                if not is_long and long_ratio > 0.65:
                    score += 1; reasons.append(f"L/S: {long_ratio:.0%} long (Masse bullisch → konträr bearish)")
                elif is_long and long_ratio < 0.40:
                    score += 1; reasons.append(f"L/S: {long_ratio:.0%} long (Masse bärisch → konträr bullisch)")

            # 14. OI Delta — Liquiditätszufluss (0-2 pts, -1 if heavy outflow)
            oi_4h = sentiment.get("oi_4h_delta", 0.0)
            if oi_4h is not None:
                if oi_4h > 0.03:    # > +3% in 4H — stark ansteigendes OI = neue Konviction
                    score += 2; reasons.append(f"OI +{oi_4h*100:.1f}% (4H) — starker Kapitalzufluss")
                elif oi_4h > 0.01:  # > +1%
                    score += 1; reasons.append(f"OI +{oi_4h*100:.1f}% (4H) — Kapitalzufluss")
                elif oi_4h < -0.03: # > -3% — Positionen werden massiv abgebaut
                    score -= 1; reasons.append(f"OI {oi_4h*100:.1f}% (4H) — Kapitalabfluss ⚠")

            # 15. CVD — Cumulative Volume Delta (0-2 pts, -1 if opposing flow)
            cvd_ratio = sentiment.get("cvd_ratio", 0.5)
            if cvd_ratio is not None:
                if is_long:
                    if cvd_ratio > 0.65:
                        score += 2; reasons.append(f"CVD {cvd_ratio:.0%} Buy — starker Kaufdruck")
                    elif cvd_ratio > 0.55:
                        score += 1; reasons.append(f"CVD {cvd_ratio:.0%} Buy — Kaufdruck")
                    elif cvd_ratio < 0.38:
                        score -= 1; reasons.append(f"CVD {cvd_ratio:.0%} Buy — Gegenwind Verkaufsdruck ⚠")
                else:
                    if cvd_ratio < 0.35:
                        score += 2; reasons.append(f"CVD {cvd_ratio:.0%} Buy — starker Verkaufsdruck")
                    elif cvd_ratio < 0.45:
                        score += 1; reasons.append(f"CVD {cvd_ratio:.0%} Buy — Verkaufsdruck")
                    elif cvd_ratio > 0.62:
                        score -= 1; reasons.append(f"CVD {cvd_ratio:.0%} Buy — Gegenwind Kaufdruck ⚠")

        # 16. Ichimoku Cloud — price vs. cloud + TK-cross (0-2 pts, -1 counter-trend)
        ichi = indicators.get("ichimoku") or {}
        if ichi.get("available"):
            pos_ = ichi["price_vs_cloud"]
            tk   = ichi["tk_cross"]
            if is_long:
                if pos_ == "above" and tk == "bullish":
                    score += 2; reasons.append("Ichimoku: über Cloud + bullischer TK-Cross")
                elif pos_ == "above":
                    score += 1; reasons.append("Ichimoku: über Cloud")
                elif pos_ == "below":
                    score -= 1; reasons.append("Ichimoku: unter Cloud ⚠")
            else:
                if pos_ == "below" and tk == "bearish":
                    score += 2; reasons.append("Ichimoku: unter Cloud + bärischer TK-Cross")
                elif pos_ == "below":
                    score += 1; reasons.append("Ichimoku: unter Cloud")
                elif pos_ == "above":
                    score -= 1; reasons.append("Ichimoku: über Cloud ⚠")

        # 17. News headlines (RSS, keyword-scored) — confirming, not contrarian (0-1 pt)
        if sentiment:
            news_bias = sentiment.get("news_bias")
            if news_bias == "bullish" and is_long:
                score += 1; reasons.append(f"News bullisch ({sentiment.get('news_bull',0)}↑/{sentiment.get('news_bear',0)}↓)")
            elif news_bias == "bearish" and not is_long:
                score += 1; reasons.append(f"News bärisch ({sentiment.get('news_bull',0)}↑/{sentiment.get('news_bear',0)}↓)")

            # 18. CoinMarketCap composite (Fear&Greed + global market + coin momentum) (0-2 pts)
            cmc_bias = sentiment.get("cmc_signal_bias")
            if cmc_bias:
                bullish_tiers = {"strongly_bullish": 2, "bullish": 1, "slightly_bullish": 1}
                bearish_tiers = {"strongly_bearish": 2, "bearish": 1, "slightly_bearish": 1}
                fg = sentiment.get("fear_greed_value")
                fg_str = f", F&G={fg}" if fg is not None else ""
                if is_long and cmc_bias in bullish_tiers:
                    pts = bullish_tiers[cmc_bias]
                    score += pts; reasons.append(f"CMC {cmc_bias} (score={sentiment.get('cmc_signal_score',0):+.1f}{fg_str})")
                elif not is_long and cmc_bias in bearish_tiers:
                    pts = bearish_tiers[cmc_bias]
                    score += pts; reasons.append(f"CMC {cmc_bias} (score={sentiment.get('cmc_signal_score',0):+.1f}{fg_str})")

            # 19. Reddit sentiment (keyword-count over r/CryptoCurrency + coin
            # subreddit hot posts) — low weight (0-1 pt) since this is a coarse
            # keyword count, not real NLP (DeepSeek's assessment 2026-07-23, see
            # project memory: crypto subreddit sentiment is noisy on its own).
            reddit_bias = sentiment.get("reddit_bias")
            if reddit_bias == "bullish" and is_long:
                score += 1; reasons.append(f"Reddit bullisch ({sentiment.get('reddit_bull',0)}↑/{sentiment.get('reddit_bear',0)}↓)")
            elif reddit_bias == "bearish" and not is_long:
                score += 1; reasons.append(f"Reddit bärisch ({sentiment.get('reddit_bull',0)}↑/{sentiment.get('reddit_bear',0)}↓)")

            # 20. Liquidation flow (Binance forceOrder stream, free, no API key —
            # see exchange/liquidation_stream.py; NOT a Coinglass price-level
            # heatmap, that needs a $699+/mo Coinglass plan, checked 2026-07-23,
            # no free tier despite an earlier assumption otherwise). Interpreted
            # as momentum confirmation, not a contrarian "magnet" signal: heavy
            # recent SHORT liquidations = a squeeze = bullish momentum; heavy
            # recent LONG liquidations = a flush = bearish momentum. Only counts
            # when the dominant side's liquidation volume is at least 1.5x the
            # other side's (dominance_ratio) — otherwise it's just noise.
            liq_side = sentiment.get("liq_dominant_side")
            liq_ratio = sentiment.get("liq_dominance_ratio", 1.0)
            if liq_side and liq_ratio >= 1.5:
                if liq_side == "short" and is_long:
                    score += 1; reasons.append(f"Short-Squeeze-Flow (${sentiment.get('liq_short_usd',0):,.0f} liquidiert, {liq_ratio:.1f}x)")
                elif liq_side == "long" and not is_long:
                    score += 1; reasons.append(f"Long-Flush-Flow (${sentiment.get('liq_long_usd',0):,.0f} liquidiert, {liq_ratio:.1f}x)")

        # No clamp-to-0 here (unlike the old single-direction version) — this gets
        # compared against the opposite direction's score via argmax in _run_symbol,
        # so a negative score (net-bearish for the direction being scored) needs to
        # stay negative for that comparison to be meaningful.
        return score, reasons

    def get_log(self, limit: int = 50, symbol: str = None) -> list:
        logs = self.log
        if symbol:
            logs = [e for e in logs if e.get("symbol") == symbol]
        return list(reversed(logs[-limit:]))

    def get_open_positions(self) -> list:
        result = []
        for sym, pos in self.engine.positions.items():
            # prefer live_prices (5s poll) over last_decisions price (up to 5min stale)
            p = self.live_prices.get(sym) or self.last_decisions.get(sym, {}).get("price", pos.entry_price)
            d = pos.to_dict(p)
            d["current_price"] = p
            result.append(d)
        return result

    def get_trade_history(self, limit: int = 100) -> list:
        return list(reversed(self.engine.trade_history[-limit:]))


def _to_df(ohlcv: list) -> pd.DataFrame:
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df
