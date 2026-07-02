import asyncio
import concurrent.futures
import pandas as pd
from datetime import datetime, date
from typing import Optional

from exchange.futures_client import FuturesClient
from exchange.market_scanner import get_trending_symbols
from trading.futures_paper import FuturesPaperEngine, MAINTENANCE_MARGIN
from trading.risk import RiskManager, RiskConfig
from ai.ml_signal import predict, get_indicators, detect_market_structure, train as ml_train, _funding_to_series
from ai.patterns import detect_patterns

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
        retrain_every_cycles: int = 24,
        min_claude_confidence: float = 0.65,
        min_ml_conf: float = 0.35,
        min_confluence: int = 5,
    ):
        self.symbols = list(symbols or DEFAULT_SYMBOLS)
        self.timeframe = timeframe
        self.interval = interval_seconds
        self.engine = engine or FuturesPaperEngine()
        self.risk = RiskManager(risk_config or RiskConfig())
        self.max_leverage = max_leverage
        self.max_open_positions = max_open_positions
        self.running = False
        self._task: Optional[asyncio.Task] = None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self._position_lock = asyncio.Lock()   # prevents _monitor_loop + _run_symbol double-close

        self.min_claude_confidence: float = min_claude_confidence  # kept for API compat
        self.min_ml_conf: float = min_ml_conf
        self.min_confluence: int = min_confluence
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

    # ── model training ────────────────────────────────────────────────────────
    async def train_model(self, symbol: str, limit: int = 2000) -> dict:
        self._log("INFO", f"Training ML model — {symbol} {self.timeframe} x{limit}", symbol)
        self.training_progress[symbol] = 0
        async with FuturesClient() as client:
            ohlcv, funding_raw = await asyncio.gather(
                client.fetch_ohlcv(symbol, self.timeframe, limit),
                client.fetch_funding_rate_history(symbol, 500),
            )
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
        self._log("INFO",
            f"ML ready — bal_acc {result['accuracy']*100:.1f}% | dir_prec {result.get('dir_precision',0)*100:.1f}% | f1 {result.get('f1_macro',0):.2f} | {result['samples']} samples",
            symbol)
        return result

    async def train_all(self, limit: int = 1000, symbols: list = None) -> list:
        targets = symbols if symbols is not None else self.symbols
        return await asyncio.gather(*[self.train_model(sym, limit) for sym in targets])

    # ── single symbol cycle ───────────────────────────────────────────────────
    async def _run_symbol(self, client: FuturesClient, symbol: str):
        try:
            async def _safe_funding_history():
                try:
                    return await asyncio.wait_for(client.fetch_funding_rate_history(symbol, 100), timeout=10)
                except Exception:
                    return []

            async def _safe_sentiment():
                try:
                    return await asyncio.wait_for(client.fetch_market_sentiment(symbol), timeout=10)
                except Exception:
                    return {}

            async def _safe_cvd():
                try:
                    return await asyncio.wait_for(client.fetch_cvd(symbol, 500), timeout=10)
                except Exception:
                    return {}

            async def _safe_oi_current():
                try:
                    return await asyncio.wait_for(client.fetch_current_oi(symbol), timeout=6)
                except Exception:
                    return 0.0

            ohlcv, ohlcv_4h, ohlcv_1d, ohlcv_15m, fr, funding_raw, market_sentiment, cvd_data, oi_now = await asyncio.gather(
                client.fetch_ohlcv(symbol, self.timeframe, 300),
                client.fetch_ohlcv(symbol, "4h", 100),
                client.fetch_ohlcv(symbol, "1d", 100),
                client.fetch_ohlcv(symbol, "15m", 200),
                client.fetch_funding_rate(symbol),
                _safe_funding_history(),
                _safe_sentiment(),
                _safe_cvd(),
                _safe_oi_current(),
            )

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
            market_sentiment["cvd_ratio"]    = cvd_data.get("cvd_ratio", 0.5)
            market_sentiment["cvd_net"]      = cvd_data.get("cvd_net", 0.0)
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
            portfolio_value = self.engine.portfolio_value(self.live_prices | {symbol: price})
            open_count  = len(self.engine.positions)
            has_position = symbol in self.engine.positions

            if not has_position:   # only block NEW entries, not position management
                if not self.risk.check_daily_loss(portfolio_value):
                    self._log("WARN", f"Blocked (daily loss): {self.risk.block_reason}", symbol)
                    return
                if not self.risk.check_drawdown(portfolio_value):
                    self._log("WARN", f"Blocked (drawdown): {self.risk.block_reason}", symbol)
                    return
            else:
                # Still update peak equity tracking even when managing existing positions
                self.risk.check_drawdown(portfolio_value)

            # ── max positions check ──
            if open_count >= self.max_open_positions and not has_position:
                self._log("INFO", f"Max positions ({self.max_open_positions}) reached — skipping new entry", symbol)
                return

            # ── ML signal ──
            loop = asyncio.get_event_loop()
            ml_signal = await loop.run_in_executor(
                self._executor, lambda: predict(df, symbol, funding_series=funding_series)
            )
            indicators_quick = get_indicators(df)
            ml_signal["regime"] = indicators_quick.get("regime", "unknown")
            ml_signal["adx"]    = indicators_quick.get("adx", 0)
            self.last_ml_signals[symbol] = ml_signal

            has_position = symbol in self.engine.positions
            conf         = ml_signal["confidence"]
            label        = ml_signal["label"]   # "buy" | "sell" | "hold"

            # ── MTF indicators (1D + 15M) ─────────────────────────────────────
            ind_1d  = get_indicators(df_1d)  if len(df_1d)  > 20 else None
            ind_15m = get_indicators(df_15m) if len(df_15m) > 30 else None

            # ── Confluence filter ─────────────────────────────────────────────
            # Always compute when ML has a direction (not hold) or position open.
            # Confluence replaces the old confidence-only gate.
            confluence_score, confluence_reasons = 0, []
            if label != "hold" or has_position:
                patterns_quick = detect_patterns(df)
                confluence_score, confluence_reasons = self._confluence_score(
                    ml_signal, indicators_quick, patterns_quick, df_4h,
                    ind_1d=ind_1d, ind_15m=ind_15m, sentiment=market_sentiment,
                )

            # ── Rule-based decision (no Claude) ──────────────────────────────────
            MIN_CONFLUENCE = self.min_confluence
            MIN_CONF       = self.min_ml_conf

            di_score   = ml_signal.get("di_score", 0.0)
            di_blocked = ml_signal.get("di_blocked", False)

            self._log("INFO",
                f"ML → {label.upper()} conf={conf:.2f} | DI={di_score:.2f} | C={confluence_score}/24",
                symbol)

            action = "hold"
            skip_reason = None

            if has_position:
                # Already in a position — only manage via SL/TP (handled above)
                # Optionally close on strong counter-signal
                cur_pos = self.engine.positions.get(symbol)
                if cur_pos:
                    is_long = cur_pos.side == "long"
                    counter = (is_long and label == "sell") or (not is_long and label == "buy")
                    if counter and conf >= 0.65 and confluence_score >= MIN_CONFLUENCE:
                        action = "close_long" if is_long else "close_short"
                        skip_reason = None
                    else:
                        skip_reason = "position open — managed by SL/TP"
            elif label == "hold":
                skip_reason = "ML → HOLD"
            elif di_blocked:
                skip_reason = f"DI={di_score:.2f} — Regime-Shift, kein Entry"
            elif conf < MIN_CONF:
                skip_reason = f"conf={conf:.2f} < {MIN_CONF}"
            elif confluence_score < MIN_CONFLUENCE:
                skip_reason = f"C={confluence_score}/24 < {MIN_CONFLUENCE}"
            else:
                action = "open_long" if label == "buy" else "open_short"

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
            indicators_full = get_indicators(df)
            atr      = indicators_full.get("atr", price * 0.015)
            sl_pct   = max((atr * 1.5) / price, 0.005)   # 1.5x ATR, min 0.5%
            tp_pct   = atr * 3.0 / price                  # 3.0x ATR (R:R 1:2)
            tp1_pct  = atr * 1.5 / price                  # TP1 at 1.5x ATR (partial close)
            leverage = self.max_leverage

            # Stop-loss must trigger before liquidation, or it never fires — the position
            # rides to near-total margin loss instead of the sized risk_amount. Cap the SL
            # distance to 80% of the leverage's liquidation buffer.
            liq_buffer_pct = 1 / leverage - MAINTENANCE_MARGIN
            sl_pct   = min(sl_pct, liq_buffer_pct * 0.8)

            trailing_sl = True
            trail_pct   = sl_pct

            reasons_str = " | ".join(confluence_reasons[:4])
            self._log("INFO",
                f"SIGNAL {action.upper()} | C={confluence_score}/24 | conf={conf:.2f} | SL={sl_pct:.1%} TP={tp_pct:.1%} | {reasons_str}",
                symbol)

            decision = {
                "action": action, "confidence": conf,
                "reasoning": f"Rule: C={confluence_score}/24 ≥ {MIN_CONFLUENCE}, conf={conf:.2f} ≥ {MIN_CONF}. {reasons_str}",
                "leverage": leverage, "stop_loss_pct": sl_pct, "take_profit_pct": tp_pct,
                "tp1_pct": tp1_pct, "trailing_sl": trailing_sl, "trail_pct": trail_pct,
                "confluence_score": confluence_score, "confluence_reasons": confluence_reasons,
                "di_score": di_score, "ts": datetime.now().isoformat(), "price": price,
            }
            self.last_decisions[symbol] = decision

            # ── ATR-based position sizing (Kelly-scaled risk-per-trade) ─────────
            equity      = self.engine.portfolio_value(self.live_prices | {symbol: price})
            risk_pct    = self._kelly_risk_pct(symbol)
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
                record = self.engine.open_position(
                    symbol, "long", amount, price, leverage,
                    price * (1 - sl_pct), price * (1 + tp_pct),
                    trailing_sl=trailing_sl, trail_pct=trail_pct)
                self._log("TRADE",
                    f"OPEN LONG {amount:.6f} @ ${price:,.2f} | {leverage}x | Margin ${margin_use:.0f} | Risk ${risk_amount:.0f} ({risk_pct:.2%}) | Liq ${record['liq_price']:,.0f}",
                    symbol, {"type": "open_long", **record})

            elif action == "open_short" and not cur_pos:
                record = self.engine.open_position(
                    symbol, "short", amount, price, leverage,
                    price * (1 + sl_pct), price * (1 - tp_pct),
                    trailing_sl=trailing_sl, trail_pct=trail_pct)
                self._log("TRADE",
                    f"OPEN SHORT {amount:.6f} @ ${price:,.2f} | {leverage}x | Margin ${margin_use:.0f} | Risk ${risk_amount:.0f} ({risk_pct:.2%}) | Liq ${record['liq_price']:,.0f}",
                    symbol, {"type": "open_short", **record})

            elif action == "close_long" and cur_pos and cur_pos.side == "long":
                record = self._close(symbol, price, "counter_signal")
                self._log("TRADE",
                    f"CLOSE LONG (counter-signal) @ ${price:,.2f} | PnL {record['pnl']:+.2f} USDT | ROE {record['roe_pct']:+.1f}%",
                    symbol, {"type": "close_long", **record})

            elif action == "close_short" and cur_pos and cur_pos.side == "short":
                record = self._close(symbol, price, "counter_signal")
                self._log("TRADE",
                    f"CLOSE SHORT (counter-signal) @ ${price:,.2f} | PnL {record['pnl']:+.2f} USDT | ROE {record['roe_pct']:+.1f}%",
                    symbol, {"type": "close_short", **record})

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
                        elif trigger:
                            pos = self.engine.get_position(symbol, price)
                            record = self._close(symbol, price, trigger)
                            self._log("TRADE",
                                f"{trigger.upper()} (monitor) — closed {pos['side'].upper()} @ ${price:,.2f} | PnL: {record['pnl']:+.2f} USDT | ROE: {record['roe_pct']:+.1f}%",
                                symbol, {"type": trigger, **record})
            except Exception as e:
                self._log("ERROR", f"Monitor-Loop Fehler: {e}", "ALL")

    # ── dynamic symbol discovery ──────────────────────────────────────────────
    async def _refresh_symbols(self):
        """Replace watchlist with top trending USDT-perp pairs, keeping anchors + open positions."""
        try:
            trending = await get_trending_symbols(top_n=self.max_symbols + 2, min_volume=75_000_000)
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
                        await self.train_all()
                        self.last_retrain_cycle = self.cycle_count
                        self.last_retrain_ts    = datetime.now().isoformat()
                        self.next_retrain_cycle = self.cycle_count + self.retrain_every

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
            "cycle_count": self.cycle_count,
            "retrain_every": self.retrain_every,
            "last_retrain_cycle": self.last_retrain_cycle,
            "last_retrain_ts": self.last_retrain_ts,
            "next_retrain_cycle": self.next_retrain_cycle,
            "min_claude_confidence": self.min_claude_confidence,
            "min_ml_conf": self.min_ml_conf,
            "min_confluence": self.min_confluence,
            "claude_calls_saved": self.claude_calls_saved,
            "training_progress": self.training_progress,
            "model_accuracy": self.model_accuracy,
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

    def _kelly_risk_pct(self, symbol: str, default_pct: float = 0.015) -> float:
        """Half-Kelly risk-per-trade from realised PnL history (per-symbol, else portfolio-wide).
        Falls back to a fixed 1.5% until enough trade history exists to trust the estimate.
        """
        sym_pnls = self._trade_pnls(symbol)
        pnls = sym_pnls if len(sym_pnls) >= 20 else self._trade_pnls()
        kelly_pct = self.risk.kelly_risk_pct(pnls)
        return kelly_pct if kelly_pct is not None else default_pct

    # ── confluence score ──────────────────────────────────────────────────────
    def _confluence_score(
        self,
        ml_signal: dict,
        indicators: dict,
        patterns: dict,
        df_4h,
        ind_1d:  dict = None,
        ind_15m: dict = None,
        sentiment: dict = None,
    ) -> tuple[int, list[str]]:
        """
        Score 0-24: signals across 1D/4H/1H/15M + on-chain + flow align with ML direction.
        Layers: ML(2) + Ensemble(1) + 1H-RSI(1) + 1H-MACD(1) + 1H-EMA(1)
                + VWAP(1) + 4H-structure(2) + Candle(1) + 1D-trend(2) + 15M-mom(2)
                + Squeeze(2) + Funding-tiered(2) + L/S-ratio(1) + OI-delta(2) + CVD(2)
                + Ichimoku(2)
        """
        label = ml_signal.get("label")
        if label == "hold":
            return 0, []

        is_long = (label == "buy")
        score   = 0
        reasons = []

        # 1. ML confidence (0-2 pts)
        conf = ml_signal.get("confidence", 0)
        if conf >= 0.72:
            score += 2; reasons.append(f"ML sehr zuversichtlich ({conf:.0%})")
        elif conf >= 0.65:
            score += 1; reasons.append(f"ML zuversichtlich ({conf:.0%})")

        # 2. Ensemble agreement (0-1 pt)
        if ml_signal.get("agreement", 1.0) >= 0.67:
            score += 1; reasons.append(f"3 Modelle einig ({ml_signal['agreement']:.0%})")

        # 3. RSI positioning (0-1 pt)
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

        return max(score, 0), reasons

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
