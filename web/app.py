import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

import config
from exchange.client import BitgetClient
from exchange.futures_client import FuturesClient
from trading.paper import PaperEngine
from trading.futures_paper import FuturesPaperEngine
from trading.backtest import run_backtest
from ai.backtest import run_backtest as run_mtf_backtest
from trading.autotrader import AutoTrader
from trading.funding_harvest import FundingHarvestEngine, FundingHarvester
from trading.grid import GridEngine, GridTrader
from trading.wallet import SharedWallet
from trading.risk import RiskConfig
from strategies.base import STRATEGIES
from monitoring.alerts import AlertManager
from ai.whale import get_all_whale_data
from exchange.market_scanner import get_trending_symbols, get_all_market_overview


# ── shared state ─────────────────────────────────────────────────────────────
paper = PaperEngine()
# One real pool of capital shared by the three automated engines (AutoTrader,
# Funding Harvest, Grid) — opening a position in any one of them draws down
# the same balance, instead of each running its own independent account.
shared_wallet = SharedWallet()
futures_paper = FuturesPaperEngine(wallet=shared_wallet)
alert_mgr = AlertManager()
autotrader: AutoTrader = None
_autotrader_starting = False
funding_harvest_engine = FundingHarvestEngine(wallet=shared_wallet)
funding_harvester: FundingHarvester = None
_funding_harvester_starting = False
grid_engine = GridEngine(wallet=shared_wallet)
grid_trader: GridTrader = None
_grid_trader_starting = False
_price_cache: dict[str, dict] = {}
_ws_clients: list[WebSocket] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_price_poll_loop())
    yield
    task.cancel()


app = FastAPI(title="Bitget Trading Tool", lifespan=lifespan)

WATCH_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


def _add_watch_symbols(symbols: list[str]):
    for s in symbols:
        if s not in WATCH_SYMBOLS:
            WATCH_SYMBOLS.append(s)


async def _price_poll_loop():
    # Futures client, not spot: AutoTrader positions are USDT-M perpetuals, and
    # some symbols it trades (e.g. KORU/USDT) don't exist as spot markets at all —
    # fetching those via the spot client throws BadSymbol. One unhandled failure
    # used to abort this whole loop early, silently freezing every symbol later
    # in WATCH_SYMBOLS for good (never reached again, every 5s, forever).
    async with FuturesClient() as client:
        while True:
            try:
                # include any symbols the autotrader/grid_trader added dynamically
                if autotrader:
                    for sym in autotrader.symbols:
                        if sym not in WATCH_SYMBOLS:
                            WATCH_SYMBOLS.append(sym)
                if grid_trader:
                    for sym in grid_trader.symbols:
                        if sym not in WATCH_SYMBOLS:
                            WATCH_SYMBOLS.append(sym)

                for sym in list(WATCH_SYMBOLS):
                    try:
                        ticker = await client.fetch_ticker(sym)
                    except Exception:
                        continue   # don't let one bad symbol block the rest
                    price  = ticker["last"]
                    _price_cache[sym] = {
                        "symbol": sym,
                        "price": price,
                        "change_pct": ticker.get("percentage", 0),
                        "volume": ticker.get("quoteVolume", 0),
                        "high": ticker.get("high", 0),
                        "low": ticker.get("low", 0),
                        "ts": datetime.now().isoformat(),
                    }
                    alert_mgr.check(sym, price)
                    # push live price into autotrader/grid_trader for fast SL/TP/fill monitoring
                    if autotrader:
                        autotrader.live_prices[sym] = price
                    if grid_trader:
                        grid_trader.live_prices[sym] = price

                await _broadcast({"type": "prices", "data": list(_price_cache.values())})
                # equity snapshot every cycle (works with or without AutoTrader)
                prices = {sym: d["price"] for sym, d in _price_cache.items()}
                engine = autotrader.engine if autotrader else futures_paper
                engine.record_equity(prices)

                # broadcast live position updates so frontend stays current
                if autotrader and autotrader.engine.positions:
                    positions = autotrader.get_open_positions()
                    await _broadcast({"type": "positions", "data": positions})

            except Exception:
                pass
            await asyncio.sleep(5)


async def _broadcast(msg: dict):
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)


# ── WebSocket ─────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.append(ws)
    if _price_cache:
        await ws.send_json({"type": "prices", "data": list(_price_cache.values())})
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


# ── Prices ────────────────────────────────────────────────────────────────────
@app.get("/api/prices")
async def get_prices():
    if not _price_cache:
        async with FuturesClient() as client:
            for sym in WATCH_SYMBOLS:
                try:
                    t = await client.fetch_ticker(sym)
                except Exception:
                    continue
                _price_cache[sym] = {"symbol": sym, "price": t["last"], "change_pct": t.get("percentage", 0)}
    return list(_price_cache.values())


@app.get("/api/ohlcv")
async def get_ohlcv(symbol: str = "BTC/USDT", timeframe: str = "1h", limit: int = 100):
    async with BitgetClient() as client:
        data = await client.fetch_ohlcv(symbol, timeframe, limit)
    return [{"t": d[0], "o": d[1], "h": d[2], "l": d[3], "c": d[4], "v": d[5]} for d in data]


@app.get("/api/orderbook")
async def get_orderbook(symbol: str = "BTC/USDT"):
    async with BitgetClient() as client:
        ob = await client.fetch_order_book(symbol, 10)
    return ob


# ── Spot Paper Trading ────────────────────────────────────────────────────────
class OrderRequest(BaseModel):
    symbol: str
    side: str
    amount: float
    price: Optional[float] = None


@app.get("/api/paper/balance")
def get_paper_balance():
    prices = {sym: d["price"] for sym, d in _price_cache.items()}
    return {
        "balance": paper.get_balance(),
        "positions": paper.get_positions(),
        "portfolio_value": paper.portfolio_value(prices),
        "pnl": paper.pnl(prices),
    }


@app.post("/api/paper/order")
async def place_paper_order(req: OrderRequest):
    if req.price is None:
        if req.symbol not in _price_cache:
            raise HTTPException(400, "No price available")
        req.price = _price_cache[req.symbol]["price"]
    try:
        order = paper.place_order(req.symbol, req.side, req.amount, req.price)
        await _broadcast({"type": "order", "data": order})
        return order
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/paper/orders")
def get_paper_orders():
    return paper.trade_history


# ── Futures Paper ─────────────────────────────────────────────────────────────
@app.get("/api/futures/status")
def get_futures_status():
    prices = {sym: d["price"] for sym, d in _price_cache.items()}
    return futures_paper.status(prices)


@app.get("/api/futures/trades")
def get_futures_trades():
    return futures_paper.trade_history


@app.post("/api/futures/close/{symbol:path}")
async def close_futures_position_manual(symbol: str):
    if symbol not in futures_paper.positions:
        raise HTTPException(400, f"No open position for {symbol}")
    async with FuturesClient() as client:
        t = await client.fetch_ticker(symbol)
        price = t["last"]
    record = futures_paper.close_position(symbol, price, "manual")
    return {"status": "closed", "record": record}


@app.get("/api/whale")
async def get_whale_data(symbol: str = "BTC/USDT"):
    return await get_all_whale_data(symbol)


# ── Backtest ──────────────────────────────────────────────────────────────────
class BacktestRequest(BaseModel):
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    limit: int = 500
    strategy: str = "sma_crossover"
    initial_balance: float = 10000.0


@app.post("/api/backtest")
async def run_backtest_endpoint(req: BacktestRequest):
    if req.strategy not in STRATEGIES:
        raise HTTPException(400, f"Unknown strategy. Available: {list(STRATEGIES.keys())}")
    async with BitgetClient() as client:
        ohlcv = await client.fetch_ohlcv(req.symbol, req.timeframe, req.limit)
    result = run_backtest(ohlcv, STRATEGIES[req.strategy], req.initial_balance)
    return {"summary": result.summary(), "equity_curve": result.equity_curve[-100:], "trades": result.trades}


class MtfBacktestRequest(BaseModel):
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    limit: int = 8000           # Binance: verified available back to ~333 days 1H (2026-07-22); Bitget: max ~1200
    data_source: str = "binance"  # "binance" (more history) or "bitget"
    train_pct: float = 0.70
    min_confluence: int = 4   # calibrated 2026-07-22, see ai/backtest.py::run_backtest
    min_conf: float = 0.40
    atr_sl_mult: float = 1.5
    atr_tp_mult: float = 3.0
    leverage: int = 5


@app.post("/api/backtest/mtf")
async def run_mtf_backtest_endpoint(req: MtfBacktestRequest):
    from exchange.futures_client import FuturesClient
    from exchange.binance_data import fetch_ohlcv_binance
    from ai.ml_signal import _funding_to_series
    import pandas as pd
    import concurrent.futures

    if req.data_source == "binance":
        # Binance: public, no key, years of history — only for backtest data
        ohlcv = await fetch_ohlcv_binance(req.symbol, req.timeframe, req.limit)
        # Funding history still from Bitget (8H records, up to 33 days)
        async with FuturesClient() as client:
            funding_raw = await client.fetch_funding_rate_history(req.symbol, 500)
    else:
        async with FuturesClient() as client:
            ohlcv, funding_raw = await asyncio.gather(
                client.fetch_ohlcv(req.symbol, req.timeframe, req.limit),
                client.fetch_funding_rate_history(req.symbol, 500),
            )

    if not ohlcv:
        raise HTTPException(400, "No OHLCV data received")

    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("ts", inplace=True)

    funding_series = _funding_to_series(funding_raw)

    loop = asyncio.get_event_loop()

    def _progress(msg: str):
        # called from the worker thread — hop back onto the event loop to broadcast
        asyncio.run_coroutine_threadsafe(
            _broadcast({"type": "backtest_log", "msg": msg}), loop
        )

    _progress(f"Backtest gestartet — {req.symbol} {req.timeframe} x{req.limit} ({req.data_source})")
    with concurrent.futures.ThreadPoolExecutor() as ex:
        result = await loop.run_in_executor(
            ex,
            lambda: run_mtf_backtest(
                df, req.symbol, funding_series=funding_series,
                train_pct=req.train_pct, min_confluence=req.min_confluence,
                min_conf=req.min_conf, atr_sl_mult=req.atr_sl_mult,
                atr_tp_mult=req.atr_tp_mult, leverage=req.leverage,
                progress_cb=_progress,
            )
        )
    result["data_source"] = req.data_source
    return result


# ── Alerts ────────────────────────────────────────────────────────────────────
class AlertRequest(BaseModel):
    symbol: str
    condition: str
    value: float


@app.get("/api/alerts")
def get_alerts():
    return {"alerts": alert_mgr.get_alerts(), "log": alert_mgr.get_log()}


@app.post("/api/alerts")
def create_alert(req: AlertRequest):
    a = alert_mgr.add_alert(req.symbol, req.condition, req.value)
    return {"status": "created", "alert": {"symbol": a.symbol, "condition": a.condition, "value": a.value}}


@app.delete("/api/alerts/{index}")
def delete_alert(index: int):
    alert_mgr.remove_alert(index)
    return {"status": "deleted"}


# ── AI AutoTrader ─────────────────────────────────────────────────────────────
class AutotraderStartRequest(BaseModel):
    symbols: list[str] = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    timeframe: str = "1h"
    interval_seconds: int = 300
    max_position_pct: float = 0.20
    max_daily_loss_pct: float = 0.05
    max_leverage: int = 15
    max_open_positions: int = 3
    max_same_direction: int = 3
    retrain_every_cycles: int = 24
    min_claude_confidence: float = 0.65
    min_ml_conf: float = 0.35
    min_confluence: int = 8
    min_dir_precision: float = 0.30
    max_stale_cycles: int = 12
    stale_conf_floor: float = 0.15


@app.post("/api/autotrader/start")
async def start_autotrader(req: AutotraderStartRequest):
    global autotrader, futures_paper, _autotrader_starting
    # autotrader.running only flips True after the (multi-second) initial training
    # finishes, so two near-simultaneous requests can both pass that check and spin
    # up duplicate instances. _autotrader_starting is set synchronously, before any
    # await, so a second request lands on it before we ever release control.
    if _autotrader_starting or (autotrader and autotrader.running):
        raise HTTPException(400, "AutoTrader already running")
    _autotrader_starting = True
    try:
        risk_cfg = RiskConfig(
            max_position_pct=req.max_position_pct,
            max_daily_loss_pct=req.max_daily_loss_pct,
        )
        autotrader = AutoTrader(
            symbols=req.symbols,
            timeframe=req.timeframe,
            interval_seconds=req.interval_seconds,
            engine=futures_paper,
            risk_config=risk_cfg,
            max_leverage=req.max_leverage,
            max_open_positions=req.max_open_positions,
            max_same_direction=req.max_same_direction,
            retrain_every_cycles=req.retrain_every_cycles,
            min_claude_confidence=req.min_claude_confidence,
            min_ml_conf=req.min_ml_conf,
            min_confluence=req.min_confluence,
            min_dir_precision=req.min_dir_precision,
            max_stale_cycles=req.max_stale_cycles,
            stale_conf_floor=req.stale_conf_floor,
        )
        _add_watch_symbols(req.symbols)
        train_results = await autotrader.startup()
        return {"status": "started", "models": train_results}
    finally:
        _autotrader_starting = False


@app.post("/api/autotrader/stop")
async def stop_autotrader():
    global autotrader
    if not autotrader or not autotrader.running:
        raise HTTPException(400, "AutoTrader not running")
    await autotrader.stop()
    return {"status": "stopped"}


class AutotraderConfigPatch(BaseModel):
    min_ml_conf: float = None
    min_confluence: int = None
    max_leverage: int = None
    max_open_positions: int = None
    max_same_direction: int = None
    min_dir_precision: float = None
    max_stale_cycles: int = None
    stale_conf_floor: float = None

@app.patch("/api/autotrader/config")
def patch_autotrader_config(patch: AutotraderConfigPatch):
    if not autotrader:
        raise HTTPException(400, "AutoTrader not initialised")
    changed = {}
    if patch.min_ml_conf is not None:
        autotrader.min_ml_conf = patch.min_ml_conf
        changed["min_ml_conf"] = patch.min_ml_conf
    if patch.min_confluence is not None:
        autotrader.min_confluence = patch.min_confluence
        changed["min_confluence"] = patch.min_confluence
    if patch.max_leverage is not None:
        autotrader.max_leverage = patch.max_leverage
        changed["max_leverage"] = patch.max_leverage
    if patch.max_open_positions is not None:
        autotrader.max_open_positions = patch.max_open_positions
        changed["max_open_positions"] = patch.max_open_positions
    if patch.max_same_direction is not None:
        autotrader.max_same_direction = patch.max_same_direction
        changed["max_same_direction"] = patch.max_same_direction
    if patch.min_dir_precision is not None:
        autotrader.min_dir_precision = patch.min_dir_precision
        changed["min_dir_precision"] = patch.min_dir_precision
    if patch.max_stale_cycles is not None:
        autotrader.max_stale_cycles = patch.max_stale_cycles
        changed["max_stale_cycles"] = patch.max_stale_cycles
    if patch.stale_conf_floor is not None:
        autotrader.stale_conf_floor = patch.stale_conf_floor
        changed["stale_conf_floor"] = patch.stale_conf_floor
    return {"status": "updated", "changed": changed}


@app.get("/api/autotrader/status")
def get_autotrader_status():
    if not autotrader:
        prices = {sym: d["price"] for sym, d in _price_cache.items()}
        es = futures_paper.status(prices)
        return {
            "running": False, "model_trained": False,
            "engine": es, "symbols": [], "cycle_count": 0,
        }
    return autotrader.status()




@app.post("/api/autotrader/train")
async def retrain_model(limit: int = 1000):
    if not autotrader:
        raise HTTPException(400, "Start AutoTrader first")
    return await autotrader.train_all(limit)


@app.get("/api/autotrader/positions")
def get_positions():
    engine = autotrader.engine if autotrader else futures_paper
    prices = {sym: d["price"] for sym, d in _price_cache.items()}
    result = []
    for sym, pos in engine.positions.items():
        p = prices.get(sym, pos.entry_price)
        d = pos.to_dict(p)
        d["current_price"] = p
        result.append(d)
    return result


@app.get("/api/autotrader/history")
def get_history(limit: int = 100):
    engine = autotrader.engine if autotrader else futures_paper
    return list(reversed(engine.trade_history[-limit:]))


@app.get("/api/autotrader/equity")
def get_equity_curve():
    engine = autotrader.engine if autotrader else futures_paper
    return engine.equity_history


@app.get("/api/autotrader/log")
def get_autotrader_log(limit: int = 50, symbol: str = None):
    if not autotrader:
        return []
    return autotrader.get_log(limit, symbol)


# ── Funding-Rate Harvest (delta-neutral spot+perp, separate from the trend/scalp AutoTrader) ──
class FundingHarvestStartRequest(BaseModel):
    symbols: list[str] = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
    interval_seconds: int = 900
    entry_rate_threshold: float = 0.00025
    exit_rate_threshold: float = 0.0
    min_hold_settlements: int = 13
    max_basis_pct: float = 0.5
    max_position_pct: float = 0.25
    leverage: int = 2


@app.post("/api/funding-harvest/start")
async def start_funding_harvest(req: FundingHarvestStartRequest):
    global funding_harvester, _funding_harvester_starting
    if _funding_harvester_starting or (funding_harvester and funding_harvester.running):
        raise HTTPException(400, "FundingHarvester already running")
    _funding_harvester_starting = True
    try:
        funding_harvester = FundingHarvester(
            symbols=req.symbols,
            engine=funding_harvest_engine,
            interval_seconds=req.interval_seconds,
            entry_rate_threshold=req.entry_rate_threshold,
            exit_rate_threshold=req.exit_rate_threshold,
            min_hold_settlements=req.min_hold_settlements,
            max_basis_pct=req.max_basis_pct,
            max_position_pct=req.max_position_pct,
            leverage=req.leverage,
        )
        funding_harvester.start()
        return {"status": "started"}
    finally:
        _funding_harvester_starting = False


@app.post("/api/funding-harvest/stop")
async def stop_funding_harvest():
    if not funding_harvester or not funding_harvester.running:
        raise HTTPException(400, "FundingHarvester not running")
    await funding_harvester.stop()
    return {"status": "stopped"}


@app.get("/api/funding-harvest/status")
async def funding_harvest_status():
    prices = {}
    if funding_harvest_engine.positions:
        async with BitgetClient() as spot, FuturesClient() as perp:
            for sym in list(funding_harvest_engine.positions):
                try:
                    s, p = await asyncio.gather(spot.fetch_ticker(sym), perp.fetch_ticker(sym))
                    prices[sym] = {"spot": s["last"], "perp": p["last"]}
                except Exception:
                    continue
    base = funding_harvester.status() if funding_harvester else {"running": False}
    base["engine"] = funding_harvest_engine.status(prices)
    return base


@app.get("/api/funding-harvest/history")
def funding_harvest_history(limit: int = 50):
    return list(reversed(funding_harvest_engine.trade_history[-limit:]))


@app.get("/api/funding-harvest/log")
def funding_harvest_log(limit: int = 50):
    if not funding_harvester:
        return []
    return funding_harvester.log[-limit:]


# ── Grid Trading (structural range-oscillation edge, separate from everything else) ──
class GridStartRequest(BaseModel):
    symbols: list[str] = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
    interval_seconds: int = 300
    timeframe: str = "15m"
    n_levels: int = 10
    atr_range_mult: float = 2.0
    capital_per_grid_pct: float = 0.20
    stop_loss_pct: float = 0.03
    adx_ranging_max: float = 20.0
    min_margin_multiple: float = 3.0
    min_capital_per_level: float = 15.0
    dynamic_symbols: bool = True
    max_symbols: int = 8
    anchor_symbols: list[str] = None
    symbol_refresh_cycles: int = 12


@app.post("/api/grid/start")
async def start_grid(req: GridStartRequest):
    global grid_trader, _grid_trader_starting
    if _grid_trader_starting or (grid_trader and grid_trader.running):
        raise HTTPException(400, "GridTrader already running")
    _grid_trader_starting = True
    try:
        grid_trader = GridTrader(
            symbols=req.symbols,
            engine=grid_engine,
            interval_seconds=req.interval_seconds,
            timeframe=req.timeframe,
            n_levels=req.n_levels,
            atr_range_mult=req.atr_range_mult,
            capital_per_grid_pct=req.capital_per_grid_pct,
            stop_loss_pct=req.stop_loss_pct,
            adx_ranging_max=req.adx_ranging_max,
            min_margin_multiple=req.min_margin_multiple,
            min_capital_per_level=req.min_capital_per_level,
            dynamic_symbols=req.dynamic_symbols,
            max_symbols=req.max_symbols,
            anchor_symbols=req.anchor_symbols,
            symbol_refresh_cycles=req.symbol_refresh_cycles,
        )
        grid_trader.start()
        return {"status": "started"}
    finally:
        _grid_trader_starting = False


@app.post("/api/grid/stop")
async def stop_grid():
    if not grid_trader or not grid_trader.running:
        raise HTTPException(400, "GridTrader not running")
    await grid_trader.stop()
    return {"status": "stopped"}


@app.post("/api/grid/close/{symbol:path}")
async def close_grid_manual(symbol: str):
    if symbol not in grid_engine.grids:
        raise HTTPException(400, f"No active grid for {symbol}")
    async with FuturesClient() as client:
        t = await client.fetch_ticker(symbol)
        price = t["last"]
    record = grid_engine.close_grid(symbol, price, "manual")
    return {"status": "closed", "record": record}


@app.get("/api/grid/status")
async def grid_status():
    prices = {}
    if grid_engine.grids:
        async with FuturesClient() as client:
            for sym in list(grid_engine.grids):
                try:
                    t = await client.fetch_ticker(sym)
                    prices[sym] = t["last"]
                except Exception:
                    continue
    base = grid_trader.status() if grid_trader else {"running": False}
    base["engine"] = grid_engine.status(prices)
    return base


@app.get("/api/grid/history")
def grid_history(limit: int = 50):
    return list(reversed(grid_engine.trade_history[-limit:]))


@app.get("/api/strategy-correlation")
def strategy_correlation():
    from trading.correlation import strategy_correlation_report
    return strategy_correlation_report(
        futures_paper.trade_history,
        grid_engine.trade_history,
        funding_harvest_engine.trade_history,
    )


@app.get("/api/grid/log")
def grid_log(limit: int = 50):
    if not grid_trader:
        return []
    return grid_trader.log[-limit:]


@app.get("/api/portfolio/total")
async def portfolio_total():
    """The true combined total across the shared wallet — the individual
    engines' own portfolio_value() only adds back their own positions, so
    summing those directly would double-count the shared balance."""
    futures_prices = {sym: d["price"] for sym, d in _price_cache.items()}
    futures_value = futures_paper.portfolio_value(futures_prices)
    futures_positions_value = futures_value - shared_wallet.balance

    harvest_prices = {}
    if funding_harvest_engine.positions:
        async with BitgetClient() as spot, FuturesClient() as perp:
            for sym in list(funding_harvest_engine.positions):
                try:
                    s, p = await asyncio.gather(spot.fetch_ticker(sym), perp.fetch_ticker(sym))
                    harvest_prices[sym] = {"spot": s["last"], "perp": p["last"]}
                except Exception:
                    continue
    harvest_value = funding_harvest_engine.portfolio_value(harvest_prices)
    harvest_positions_value = harvest_value - shared_wallet.balance

    grid_prices = {}
    if grid_engine.grids:
        async with FuturesClient() as client:
            for sym in list(grid_engine.grids):
                try:
                    t = await client.fetch_ticker(sym)
                    grid_prices[sym] = t["last"]
                except Exception:
                    continue
    grid_value = grid_engine.portfolio_value(grid_prices)
    grid_positions_value = grid_value - shared_wallet.balance

    total = shared_wallet.balance + futures_positions_value + harvest_positions_value + grid_positions_value
    return {
        "shared_balance": round(shared_wallet.balance, 2),
        "initial_balance": shared_wallet.initial_balance,
        "autotrader_positions_value": round(futures_positions_value, 2),
        "funding_harvest_positions_value": round(harvest_positions_value, 2),
        "grid_positions_value": round(grid_positions_value, 2),
        "total_portfolio_value": round(total, 2),
        "total_pnl": round(total - shared_wallet.initial_balance, 2),
    }


@app.get("/api/market/trending")
async def market_trending(top_n: int = 10):
    """Top trending USDT-perp pairs by momentum score."""
    return await get_trending_symbols(top_n=top_n)


@app.get("/api/market/overview")
async def market_overview():
    """Top gainers + losers for UI display."""
    return await get_all_market_overview(top_n=20)


@app.post("/api/autotrader/symbols/refresh")
async def refresh_symbols():
    """Manually trigger a symbol refresh."""
    if not autotrader:
        raise HTTPException(400, "Start AutoTrader first")
    await autotrader._refresh_symbols()
    return {"symbols": autotrader.symbols, "trending": autotrader.trending_data}


# ── UI ────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    with open("web/templates/index.html") as f:
        return f.read()
