#!/usr/bin/env python3
import argparse
import asyncio
import sys
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Bitget Trading Tool")
    sub = parser.add_subparsers(dest="cmd")

    web_p = sub.add_parser("web", help="Start web UI (default: port 8080)")
    web_p.add_argument("--port", type=int, default=8080)
    web_p.add_argument("--host", default="0.0.0.0")

    sub.add_parser("terminal", help="Start terminal dashboard")

    bt_p = sub.add_parser("backtest", help="Quick backtest from CLI")
    bt_p.add_argument("--symbol", default="BTC/USDT")
    bt_p.add_argument("--timeframe", default="1h")
    bt_p.add_argument("--limit", type=int, default=500)
    bt_p.add_argument("--strategy", default="sma_crossover", choices=["sma_crossover", "rsi", "bollinger_bands"])
    bt_p.add_argument("--balance", type=float, default=10000)

    args = parser.parse_args()

    if args.cmd == "web" or args.cmd is None:
        from web.app import app
        port = getattr(args, "port", 8080)
        host = getattr(args, "host", "0.0.0.0")
        print(f"Starting web UI at http://{host}:{port}")
        uvicorn.run(app, host=host, port=port, log_level="warning")

    elif args.cmd == "terminal":
        from terminal.dashboard import start
        start()

    elif args.cmd == "backtest":
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text
        from rich import box
        from exchange.client import BitgetClient
        from trading.backtest import run_backtest
        from strategies.base import STRATEGIES

        console = Console()

        async def run():
            console.print(f"[blue]Fetching {args.limit} candles for {args.symbol} ({args.timeframe})...[/]")
            async with BitgetClient() as client:
                ohlcv = await client.fetch_ohlcv(args.symbol, args.timeframe, args.limit)
            result = run_backtest(ohlcv, STRATEGIES[args.strategy], args.balance)
            s = result.summary()

            table = Table(title=f"Backtest: {args.symbol} / {args.strategy}", box=box.ROUNDED)
            table.add_column("Metric", style="dim")
            table.add_column("Value", justify="right")
            for k, v in s.items():
                style = ""
                if "return" in k or "pnl" in k:
                    style = "green" if (isinstance(v, (int, float)) and v >= 0) else "red"
                table.add_row(k.replace("_", " ").title(), Text(str(v), style=style) if style else str(v))

            console.print(table)

        asyncio.run(run())


if __name__ == "__main__":
    main()
