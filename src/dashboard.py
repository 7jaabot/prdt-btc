"""
dashboard.py — Rich-based live dashboard for BNB Up/Down Trading Bot

Displays a compact, auto-refreshing terminal UI with:
  - Status panel: mode, BNB price, balance, PnL, ROI, win rate
  - Two round cards side-by-side:
      * LIVE card (epoch N-1, locked): lock price, live BNB, position, close countdown
      * NEXT card (epoch N, betting): phase, pool, P(Up), entry info
  - Recent trades table (last 10 trades)

Usage:
    dashboard = Dashboard(trader=trader, binance=binance_feed, mode="paper")
    live = dashboard.start()   # returns a rich.live.Live instance
    # Use as context manager in your run() method:
    async with dashboard.start() as live:
        ...
        live.update(dashboard.render())   # call to refresh display
"""

import time
from collections import deque
from datetime import datetime
from typing import Any, Optional

from rich.columns import Columns
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class Dashboard:
    """
    Rich live dashboard for the BNB Up/Down trading bot.

    Works with both PaperTrader and LiveTrader — reads from trader.metrics
    and trader._trades using duck typing.
    """

    def __init__(self, trader: Any, binance: Any, mode: str):
        """
        Args:
            trader: PaperTrader or LiveTrader instance.
            binance: BinanceFeed instance (for last_price).
            mode: "paper" or "live".
        """
        self.trader = trader
        self.binance = binance
        self.mode = mode  # "paper" or "live"

        # Activity log: kept in buffer for file logging only, not shown on screen
        self._log_buffer: deque[str] = deque(maxlen=20)

        # Current status text (updated by main loop)
        self._status: str = "⏳ Waiting for entry window"

        # Current window info (updated on each tick)
        self._current_window: Optional[Any] = None

        # Cached live BNB balance (avoid blocking web3 calls in render)
        self._cached_bnb_balance: Optional[float] = None
        self._last_balance_update: float = 0.0

        # Chainlink BNB/USD price (same oracle as PancakeSwap, refreshed every ~10s)
        self._chainlink_bnb_price: Optional[float] = None

        # ── NEXT round card data (epoch N, betting open) ──
        self._round_epoch: Optional[int] = None
        self._round_lock_price: Optional[float] = None
        self._round_pool_bnb: Optional[float] = None
        self._round_p_up: Optional[float] = None
        self._round_edge: Optional[float] = None
        self._round_signal_side: Optional[str] = None  # "YES", "NO", or None
        self._round_bet_bnb: Optional[float] = None
        self._round_bet_usdc: Optional[float] = None

        # ── LIVE round card data (epoch N-1, already locked) ──
        self._live_epoch: Optional[int] = None
        self._live_lock_price: Optional[float] = None
        self._live_close_ts: Optional[float] = None
        self._live_signal_side: Optional[str] = None  # "YES", "NO", or None
        self._live_bet_bnb: Optional[float] = None
        self._live_bet_usdc: Optional[float] = None
        self._live_edge: Optional[float] = None

    # ─── Public API ──────────────────────────────────────────────────────────

    def log(self, message: str) -> None:
        """Add a timestamped message to the activity log buffer (not shown on screen)."""
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_buffer.append(f"[{ts}] {message}")

    def update_status(self, status_text: str) -> None:
        """Update the current status text shown in the status panel."""
        self._status = status_text

    def update_window(self, window: Any) -> None:
        """Update the current window info for display."""
        self._current_window = window

    def update_live_round(
        self,
        epoch: Optional[int],
        lock_price: Optional[float],
        close_ts: Optional[float],
        signal_side: Optional[str],
        bet_bnb: Optional[float],
        bet_usdc: Optional[float],
        edge: Optional[float],
    ) -> None:
        """
        Update the LIVE round card (epoch N-1, already locked, waiting for close).

        Called from main.py when a new epoch is detected (the previous epoch just locked).

        Args:
            epoch: The live epoch number (N-1).
            lock_price: BNB price at lock of this round.
            close_ts: Unix timestamp when this round closes.
            signal_side: "YES" or "NO" if we have a position, else None.
            bet_bnb: Bet size in BNB (None if no trade).
            bet_usdc: Bet size in USDC (None if no trade).
            edge: Edge at entry (None if no trade).
        """
        self._live_epoch = epoch
        self._live_lock_price = lock_price
        self._live_close_ts = close_ts
        self._live_signal_side = signal_side
        self._live_bet_bnb = bet_bnb
        self._live_bet_usdc = bet_usdc
        self._live_edge = edge

    def update_round_info(
        self,
        epoch: Optional[int],
        lock_price: Optional[float],
        pool_total_bnb: Optional[float],
        p_up: Optional[float],
        edge: Optional[float],
        signal_side: Optional[str],
        bet_bnb: Optional[float],
        bet_usdc: Optional[float],
    ) -> None:
        """
        Update the NEXT round card data (epoch N, betting open).

        Called from main.py on each tick to pass fresh round/signal info.

        Args:
            epoch: Current round epoch number (betting round).
            lock_price: BNB price at lock (None if not yet locked).
            pool_total_bnb: Total BNB in the pool.
            p_up: Estimated probability of BNB going up.
            edge: Signal edge (None if no signal this round).
            signal_side: "YES" or "NO" if a trade was placed, else None.
            bet_bnb: Bet size in BNB (None if no trade).
            bet_usdc: Bet size in USDC (None if no trade).
        """
        self._round_epoch = epoch
        self._round_lock_price = lock_price
        self._round_pool_bnb = pool_total_bnb
        self._round_p_up = p_up
        self._round_edge = edge
        self._round_signal_side = signal_side
        self._round_bet_bnb = bet_bnb
        self._round_bet_usdc = bet_usdc

    def start(self) -> Live:
        """
        Create and return a Rich Live context manager.

        Usage:
            async with dashboard.start() as live:
                live.update(dashboard.render())
        """
        return Live(
            self.render(),
            refresh_per_second=0.5,  # auto-refresh every 2s as fallback
            auto_refresh=True,
            screen=False,
            transient=False,
        )

    # ─── Panel builders ──────────────────────────────────────────────────────

    def _make_status_panel(self) -> Panel:
        """Build the top status panel with mode, price, balance, PnL, window."""
        metrics = self.trader.metrics
        bnb_price = self.binance.last_price

        # ── Line 1: Mode + wallet ──
        if self.mode == "live":
            mode_str = "[bold red]🔴 LIVE[/bold red]"
            wallet = getattr(self.trader, "_wallet_address", None)
            if wallet and len(wallet) > 10:
                wallet_short = f"{wallet[:6]}...{wallet[-4:]}"
            else:
                wallet_short = wallet or "N/A"
            first_line = f"Mode: {mode_str}  |  Wallet: {wallet_short}"
        else:
            mode_str = "[bold blue]📝 PAPER[/bold blue]"
            first_line = f"Mode: {mode_str}"

        # ── Line 2: BNB price + balance ──
        price_str = f"${bnb_price:.2f}" if bnb_price is not None else "[yellow]N/A[/yellow]"

        if self.mode == "live":
            if self._cached_bnb_balance is not None and bnb_price:
                bnb_usdc = self._cached_bnb_balance * bnb_price
                balance_str = f"{self._cached_bnb_balance:.4f} BNB (${bnb_usdc:.2f})"
            elif self._cached_bnb_balance is not None:
                balance_str = f"{self._cached_bnb_balance:.4f} BNB"
            else:
                balance_str = "[yellow]loading...[/yellow]"
        else:
            balance_str = f"${metrics.bankroll:.2f}"

        price_line = f"BNB Price: {price_str}  |  Balance: {balance_str}"

        # ── Line 4: PnL / ROI / Win Rate ──
        pnl = metrics.total_pnl
        pnl_color = "green" if pnl >= 0 else "red"
        pnl_str = f"[{pnl_color}]${pnl:+.2f}[/{pnl_color}]"

        roi = metrics.roi * 100
        roi_color = "green" if roi >= 0 else "red"
        roi_str = f"[{roi_color}]{roi:+.1f}%[/{roi_color}]"

        wins = metrics.wins
        losses = metrics.losses
        win_rate = metrics.win_rate * 100
        pnl_line = (
            f"PnL: {pnl_str}  |  ROI: {roi_str}  |  "
            f"Win Rate: {win_rate:.1f}% ({wins}W / {losses}L)"
        )

        # ── Line 5: Wagered / Avg Edge ──
        wagered = metrics.total_wagered
        avg_edge = metrics.avg_edge
        wager_line = f"Wagered: ${wagered:.2f}  |  Avg Edge: {avg_edge:.3f}"

        # ── Line 7: Epoch / Lock in / Phase / Status ──
        if self._current_window is not None:
            w = self._current_window
            lock_in = int(w.seconds_remaining)
            entry_secs = getattr(w, 'entry_window_seconds', 60.0)

            if lock_in <= 0:
                phase = "[bold red]LOCKED[/bold red]"
            elif lock_in <= entry_secs:
                phase = "[bold yellow]EVALUATING[/bold yellow]"
            else:
                phase = "[bold green]BET OPEN[/bold green]"

            window_line = (
                f"Epoch: #{w.window_index}  |  Lock in: {lock_in}s  |  "
                f"Phase: {phase}  |  Status: {self._status}"
            )
        else:
            window_line = f"Status: {self._status}"

        content = Text.from_markup(
            f"{first_line}\n"
            f"{price_line}\n"
            f"\n"
            f"{pnl_line}\n"
            f"{wager_line}\n"
            f"\n"
            f"{window_line}"
        )

        return Panel(
            content,
            title="[bold cyan]BNB Up/Down Trading Bot[/bold cyan]",
            border_style="cyan",
        )

    def _make_live_round_card(self) -> Panel:
        """Build the LIVE round card (epoch N-1, already locked, waiting for close)."""
        # Use Chainlink price if available (same oracle as PancakeSwap) — fallback to Binance
        bnb_price = self._chainlink_bnb_price if self._chainlink_bnb_price is not None else self.binance.last_price
        epoch = self._live_epoch
        lock_price = self._live_lock_price
        close_ts = self._live_close_ts
        signal_side = self._live_signal_side
        bet_bnb = self._live_bet_bnb
        bet_usdc = self._live_bet_usdc
        edge = self._live_edge

        # No live round yet (first epoch after startup)
        if epoch is None:
            content = Text.from_markup(
                "\n"
                "  [dim]Waiting for first round lock...[/dim]\n"
                "\n"
            )
            return Panel(
                content,
                title="[bold red]LIVE[/bold red]",
                border_style="red",
            )

        title_str = f"[bold white]Round #{epoch}[/bold white] [bold red]🔴 LIVE[/bold red]"

        # ── Lock price ──
        lock_str = (
            f"Lock: [white]${lock_price:.2f}[/white]"
            if lock_price is not None
            else "Lock: [dim]N/A[/dim]"
        )

        # ── BNB current price (colored relative to lock) ──
        # Prefer Chainlink (same oracle as PancakeSwap) — label it accordingly
        price_source = "CL" if self._chainlink_bnb_price is not None else "Binance"
        if bnb_price is not None:
            if lock_price is not None and bnb_price > lock_price:
                price_color = "green"
            elif lock_price is not None and bnb_price < lock_price:
                price_color = "red"
            else:
                price_color = "dim"
            bnb_str = f"BNB [{price_source}]: [{price_color}]${bnb_price:.2f}[/{price_color}]"
        else:
            bnb_str = "BNB: [dim]N/A[/dim]"

        # ── UP / DOWN indicators ──
        if signal_side == "YES":
            up_str = "[bold green]▲ UP[/bold green]"
            down_str = "[dim]▼ DOWN[/dim]"
        elif signal_side == "NO":
            up_str = "[dim]▲ UP[/dim]"
            down_str = "[bold red]▼ DOWN[/bold red]"
        else:
            up_str = "[dim]▲ UP[/dim]"
            down_str = "[dim]▼ DOWN[/dim]"

        # ── Bet info / No position ──
        if bet_bnb is not None:
            bet_line = f"Bet: [white]{bet_bnb:.3f} BNB"
            if bet_usdc is not None:
                bet_line += f" (${bet_usdc:.2f})"
            bet_line += "[/white]"
            edge_line = (
                f"  Edge: [white]{edge:.2f}[/white]"
                if edge is not None
                else ""
            )
        else:
            bet_line = "[dim]No position[/dim]"
            edge_line = ""

        # ── Close countdown ──
        if close_ts is not None:
            close_in = int(close_ts - time.time())
            if close_in > 0:
                close_str = f"Close in: [white]{close_in}s[/white]"
            else:
                close_str = "[dim]Closed[/dim]"
        else:
            close_str = "[dim]—[/dim]"

        # ── Provisional win/loss ──
        outcome_str = ""
        if signal_side is not None and bnb_price is not None and lock_price is not None:
            won = (signal_side == "YES" and bnb_price > lock_price) or \
                  (signal_side == "NO" and bnb_price < lock_price)
            outcome_str = "[bold green]✅ Winning[/bold green]" if won else "[bold red]❌ Losing[/bold red]"

        content = Text.from_markup(
            f"\n"
            f"  {lock_str}   {bnb_str}\n"
            f"\n"
            f"  {up_str}    {down_str}\n"
            f"\n"
            f"  {bet_line}{edge_line}\n"
            f"  {close_str}   {outcome_str}\n"
        )

        return Panel(
            content,
            title=title_str,
            border_style="red",
        )

    def _make_next_round_card(self) -> Panel:
        """Build the NEXT round card (epoch N, betting open)."""
        epoch = self._round_epoch
        pool_bnb = self._round_pool_bnb
        p_up = self._round_p_up
        signal_side = self._round_signal_side
        bet_bnb = self._round_bet_bnb
        bet_usdc = self._round_bet_usdc

        title_str = (
            f"[bold white]Next #{epoch}[/bold white]"
            if epoch is not None
            else "[bold white]Next[/bold white]"
        )

        # ── Phase + lock countdown ──
        if self._current_window is not None:
            w = self._current_window
            seconds_left = int(w.seconds_remaining)
            entry_secs = getattr(w, 'entry_window_seconds', 60)
            if seconds_left <= 0:
                phase_str = "[bold red]🔒 Locked[/bold red]"
                lock_in_str = ""
            elif seconds_left <= entry_secs:
                phase_str = "[bold yellow]🔍 Evaluating[/bold yellow]"
                lock_in_str = f"Lock in: [white]{seconds_left}s[/white]"
            else:
                phase_str = "[bold green]Betting open[/bold green]"
                lock_in_str = f"Lock in: [white]{seconds_left}s[/white]"
        else:
            phase_str = "[dim]—[/dim]"
            lock_in_str = ""

        # ── Pool ──
        pool_str = (
            f"Pool: [white]{pool_bnb:.1f} BNB[/white]"
            if pool_bnb is not None
            else "Pool: [dim]-[/dim]"
        )

        # ── P(Up) ──
        p_up_str = (
            f"P(Up): [white]{p_up:.2f}[/white]"
            if p_up is not None
            else "P(Up): [dim]-[/dim]"
        )

        # ── Bet placed this round ──
        bet_line = ""
        if signal_side is not None and bet_bnb is not None:
            side_color = "green" if signal_side == "YES" else "red"
            side_label = "▲ UP" if signal_side == "YES" else "▼ DOWN"
            bet_line = f"[{side_color}]{side_label}[/{side_color}] {bet_bnb:.3f} BNB"
            if bet_usdc is not None:
                bet_line += f" (${bet_usdc:.2f})"

        lock_line = f"  {lock_in_str}\n" if lock_in_str else ""
        bet_section = f"  {bet_line}\n" if bet_line else ""

        content = Text.from_markup(
            f"\n"
            f"  {phase_str}\n"
            f"{lock_line}"
            f"\n"
            f"  {pool_str}\n"
            f"  {p_up_str}\n"
            f"{bet_section}"
        )

        return Panel(
            content,
            title=title_str,
            border_style="white",
        )

    def _make_trades_table(self) -> Panel:
        """Build the recent trades table (last 10, most recent first)."""
        table = Table(
            show_header=True,
            header_style="bold magenta",
            box=None,
            padding=(0, 1),
            show_edge=False,
        )
        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("Epoch", width=8, style="dim")
        table.add_column("Time", width=7)
        table.add_column("Side", width=7)
        table.add_column("Edge", width=6)
        table.add_column("PnL", width=9, justify="right")

        all_trades = list(self.trader._trades)
        recent = list(reversed(all_trades[-10:]))  # most recent first

        for i, trade in enumerate(recent):
            trade_num = len(all_trades) - i

            ts = datetime.fromtimestamp(trade.timestamp_entry).strftime("%H:%M")

            side = trade.side
            side_label = "▲ UP" if side == "YES" else "▼ DOWN"
            side_color = "green" if side == "YES" else "red"
            side_str = f"[{side_color}]{side_label}[/{side_color}]"

            edge_str = f"{trade.edge_at_entry:.2f}"

            if trade.pnl_usdc is not None:
                pnl_color = "green" if trade.pnl_usdc >= 0 else "red"
                pnl_str = f"[{pnl_color}]{trade.pnl_usdc:+.1f}[/{pnl_color}]"
            else:
                pnl_str = "[yellow]…[/yellow]"

            epoch_str = str(getattr(trade, 'epoch', '-'))
            table.add_row(str(trade_num), epoch_str, ts, side_str, edge_str, pnl_str)

        if not recent:
            table.add_row("[dim]-[/dim]", "[dim]-[/dim]", "[dim]-[/dim]", "[dim]-[/dim]", "[dim]-[/dim]", "[dim]-[/dim]")

        return Panel(
            table,
            title="[bold magenta]Recent Trades[/bold magenta]",
            border_style="magenta",
        )

    def render(self) -> Group:
        """Return a Rich renderable for the full dashboard."""
        return Group(
            self._make_status_panel(),
            Columns(
                [self._make_live_round_card(), self._make_next_round_card()],
                equal=True,
                expand=True,
            ),
            self._make_trades_table(),
        )
