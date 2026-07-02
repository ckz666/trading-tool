import asyncio
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import box

from exchange.client import BitgetClient
from trading.paper import PaperEngine
from monitoring.alerts import AlertManager


console = Console()


def make_ticker_panel(prices: dict) -> Panel:
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
    table.add_column("Symbol", style="bold")
    table.add_column("Price", justify="right")
    table.add_column("24h Change", justify="right")
    table.add_column("Volume", justify="right")

    for sym, d in prices.items():
        chg = d.get("change_pct", 0) or 0
        chg_str = f"{chg:+.2f}%"
        chg_style = "green" if chg >= 0 else "red"
        table.add_row(
            sym,
            f"${d['price']:,.2f}",
            Text(chg_str, style=chg_style),
            f"${d.get('volume', 0):,.0f}",
        )
    return Panel(table, title="[bold blue]Market Prices[/]", border_style="blue")


def make_portfolio_panel(paper: PaperEngine, prices: dict) -> Panel:
    price_map = {sym: d["price"] for sym, d in prices.items()}
    balance = paper.get_balance()
    positions = paper.get_positions()
    portfolio_val = paper.portfolio_value(price_map)
    pnl = paper.pnl(price_map)

    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("Key", style="dim")
    table.add_column("Value", justify="right")
    table.add_row("USDT Balance", f"${balance.get('USDT', 0):,.2f}")
    table.add_row("Portfolio Value", f"${portfolio_val:,.2f}")
    pnl_style = "green" if pnl >= 0 else "red"
    table.add_row("Unrealized PnL", Text(f"${pnl:+,.2f}", style=pnl_style))
    table.add_row("Trades", str(len(paper.trade_history)))

    if positions:
        table.add_section()
        for asset, qty in positions.items():
            sym = f"{asset}/USDT"
            val = qty * price_map.get(sym, 0)
            table.add_row(f"  {asset}", f"{qty:.6f} (${val:,.2f})")

    return Panel(table, title="[bold green]Paper Portfolio[/]", border_style="green")


def make_alerts_panel(alert_mgr: AlertManager) -> Panel:
    alerts = alert_mgr.get_alerts()
    if not alerts:
        return Panel("[dim]No alerts set[/]", title="[bold yellow]Alerts[/]", border_style="yellow")
    table = Table(box=box.SIMPLE, show_header=True, header_style="dim")
    table.add_column("#")
    table.add_column("Symbol")
    table.add_column("Condition")
    table.add_column("Value", justify="right")
    table.add_column("Status")
    for i, a in enumerate(alerts):
        status = Text("TRIGGERED", style="yellow bold") if a["triggered"] else Text("watching", style="dim")
        table.add_row(str(i), a["symbol"], a["condition"], str(a["value"]), status)
    return Panel(table, title="[bold yellow]Alerts[/]", border_style="yellow")


async def run_dashboard():
    paper = PaperEngine()
    alert_mgr = AlertManager()
    prices: dict = {}
    SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    layout = Layout()
    layout.split_column(
        Layout(name="top", size=8),
        Layout(name="bottom"),
    )
    layout["bottom"].split_row(
        Layout(name="portfolio"),
        Layout(name="alerts"),
    )

    async def fetch_prices(client):
        for sym in SYMBOLS:
            try:
                t = await client.fetch_ticker(sym)
                prices[sym] = {
                    "price": t["last"],
                    "change_pct": t.get("percentage", 0),
                    "volume": t.get("quoteVolume", 0),
                }
                alert_mgr.check(sym, t["last"])
            except Exception:
                pass

    async with BitgetClient() as client:
        with Live(layout, refresh_per_second=1, screen=True):
            while True:
                await fetch_prices(client)
                layout["top"].update(make_ticker_panel(prices))
                layout["portfolio"].update(make_portfolio_panel(paper, prices))
                layout["alerts"].update(make_alerts_panel(alert_mgr))
                await asyncio.sleep(5)


def start():
    try:
        asyncio.run(run_dashboard())
    except KeyboardInterrupt:
        console.print("\n[dim]Dashboard stopped.[/]")
