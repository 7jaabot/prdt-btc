"""
paper_trader.py — Paper trading simulator

Simulates trades without real money or API calls.
Logs all simulated trades to data/paper_trades.json.
Tracks PnL, win rate, and edge metrics.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional

from strategy import Signal, WindowInfo

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """A simulated paper trade."""

    trade_id: str
    timestamp_entry: float
    timestamp_exit: Optional[float]       # When resolution occurred
    side: str                              # "YES" or "NO"
    entry_price: float                     # Price we "bought" at
    position_size_usdc: float             # USDC wagered
    p_up_at_entry: float                  # Our P(Up) estimate at entry
    yes_price_at_entry: float             # Polymarket YES price at entry
    edge_at_entry: float                  # Edge we saw
    kelly_fraction: float
    window_start_ts: float
    window_end_ts: float
    window_index: int
    is_mock: bool                         # Based on mock data?

    # Resolution fields (filled later)
    bnb_open: Optional[float] = None      # BNB price at window start
    bnb_close: Optional[float] = None     # BNB price at window end
    outcome: Optional[str] = None         # "WIN" or "LOSS" or "PENDING"
    pnl_usdc: Optional[float] = None      # Profit/loss in USDC
    payout_per_share: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PaperMetrics:
    """Running metrics for the paper trader."""

    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    pending: int = 0
    total_pnl: float = 0.0
    total_wagered: float = 0.0
    avg_edge: float = 0.0
    bankroll: float = 1000.0

    @property
    def win_rate(self) -> float:
        resolved = self.wins + self.losses
        if resolved == 0:
            return 0.0
        return self.wins / resolved

    @property
    def roi(self) -> float:
        if self.total_wagered == 0:
            return 0.0
        return self.total_pnl / self.total_wagered

    def summary(self) -> str:
        return (
            f"Trades: {self.total_trades} | "
            f"Win rate: {self.win_rate:.1%} ({self.wins}W/{self.losses}L/{self.pending}P) | "
            f"PnL: ${self.total_pnl:+.2f} | "
            f"ROI: {self.roi:.2%} | "
            f"Bankroll: ${self.bankroll:.2f} | "
            f"Avg edge: {self.avg_edge:.3f}"
        )


class PaperTrader:
    """
    Paper trading engine.

    Accepts signals from the strategy and simulates trade execution.
    Resolves trades after each window ends (comparing BNB open vs close).
    Persists all trades to a JSON log file.
    """

    def __init__(self, config: dict):
        cfg_paper = config.get("paper_trading", {})
        cfg_strategy = config.get("strategy", {})

        self.log_file = cfg_paper.get("log_file", "data/paper_trades.json")
        self.simulate_latency_ms = cfg_paper.get("simulate_latency_ms", 200)
        self.starting_bankroll = cfg_strategy.get("starting_bankroll_usdc", 1000.0)

        self.metrics = PaperMetrics(bankroll=self.starting_bankroll)
        self._trades: list[Trade] = []
        self._pending_trades: list[Trade] = []
        self._trade_counter = 0

        # Load existing trades from file
        self._load_trades()

    def _load_trades(self):
        """Load existing trades from the log file."""
        if not os.path.exists(self.log_file):
            logger.info(f"No existing trades file at {self.log_file}. Starting fresh.")
            return

        try:
            with open(self.log_file) as f:
                data = json.load(f)
            trades_data = data.get("trades", [])
            logger.info(f"Loaded {len(trades_data)} existing trades from {self.log_file}")

            for t_dict in trades_data:
                trade = Trade(**t_dict)
                self._trades.append(trade)
                if trade.outcome == "PENDING" or trade.outcome is None:
                    self._pending_trades.append(trade)

            # Recompute metrics from loaded trades
            self._recompute_metrics()

        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning(f"Failed to load trades from {self.log_file}: {e}. Starting fresh.")

    def _recompute_metrics(self):
        """Recompute all metrics from the full trades list."""
        wins = [t for t in self._trades if t.outcome == "WIN"]
        losses = [t for t in self._trades if t.outcome == "LOSS"]
        pending = [t for t in self._trades if t.outcome in ("PENDING", None)]

        self.metrics.total_trades = len(self._trades)
        self.metrics.wins = len(wins)
        self.metrics.losses = len(losses)
        self.metrics.pending = len(pending)
        self.metrics.total_pnl = sum(t.pnl_usdc for t in self._trades if t.pnl_usdc is not None)
        self.metrics.total_wagered = sum(t.position_size_usdc for t in self._trades)
        self.metrics.bankroll = self.starting_bankroll + self.metrics.total_pnl

        edges = [t.edge_at_entry for t in self._trades]
        self.metrics.avg_edge = sum(edges) / len(edges) if edges else 0.0

    def _save_trades(self):
        """Persist all trades to the log file."""
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        data = {
            "metadata": {
                "last_updated": time.time(),
                "last_updated_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "total_trades": len(self._trades),
            },
            "metrics": {
                "bankroll": self.metrics.bankroll,
                "total_pnl": self.metrics.total_pnl,
                "win_rate": self.metrics.win_rate,
                "roi": self.metrics.roi,
                "total_wagered": self.metrics.total_wagered,
                "wins": self.metrics.wins,
                "losses": self.metrics.losses,
                "pending": self.metrics.pending,
                "avg_edge": self.metrics.avg_edge,
            },
            "trades": [t.to_dict() for t in self._trades],
        }

        with open(self.log_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Trades saved to {self.log_file}")

    def enter_trade(self, signal: Signal, window: WindowInfo) -> Optional[Trade]:
        """
        Simulate entering a trade based on a strategy signal.

        Args:
            signal: The trading signal.
            window: Current window information.

        Returns:
            The created Trade object, or None if rejected.
        """
        # Check bankroll
        if signal.position_size_usdc > self.metrics.bankroll:
            logger.warning(
                f"Insufficient bankroll (${self.metrics.bankroll:.2f}) "
                f"for trade size ${signal.position_size_usdc:.2f}"
            )
            return None

        self._trade_counter += 1
        trade_id = f"PT-{int(time.time())}-{self._trade_counter:04d}"

        # Simulate latency impact on entry price (slight slippage).
        # Entry is always based on signal.yes_price (0.50 when use_fair_odds=True),
        # ensuring symmetric PnL (~$50 win / ~$50 loss for $50 staked).
        latency_slippage = (self.simulate_latency_ms / 1000.0) * 0.001  # tiny
        if signal.side == "YES":
            entry_price = min(0.99, signal.yes_price + latency_slippage)
        else:
            # NO share price = 1 - YES price (+ slippage making it slightly worse)
            no_price = 1.0 - signal.yes_price
            entry_price = min(0.99, no_price + latency_slippage)

        trade = Trade(
            trade_id=trade_id,
            timestamp_entry=time.time(),
            timestamp_exit=None,
            side=signal.side,
            entry_price=entry_price,
            position_size_usdc=signal.position_size_usdc,
            p_up_at_entry=signal.p_up,
            yes_price_at_entry=signal.yes_price,
            edge_at_entry=signal.edge,
            kelly_fraction=signal.kelly_fraction,
            window_start_ts=window.window_start_ts,
            window_end_ts=window.window_end_ts,
            window_index=window.window_index,
            is_mock=signal.is_mock,
            outcome="PENDING",
        )

        self._trades.append(trade)
        self._pending_trades.append(trade)
        self.metrics.total_trades += 1
        self.metrics.pending += 1
        self.metrics.total_wagered += trade.position_size_usdc

        # Deduct from bankroll immediately (committed capital)
        self.metrics.bankroll -= trade.position_size_usdc

        self._save_trades()

        mock_tag = " [MOCK]" if trade.is_mock else ""
        logger.info(
            f"📝 Trade entered{mock_tag}: {trade_id} | "
            f"{trade.side} @ {trade.entry_price:.3f} | "
            f"${trade.position_size_usdc:.2f} USDC | "
            f"edge={trade.edge_at_entry:.3f}"
        )

        return trade

    def resolve_trades(self, window_index: int, bnb_open: float, bnb_close: float):
        """
        Resolve all pending trades for a completed window.

        Args:
            window_index: The window that just completed.
            bnb_open: BNB price at window start.
            bnb_close: BNB price at window end.
        """
        resolved = []
        still_pending = []

        bnb_went_up = bnb_close > bnb_open

        for trade in self._pending_trades:
            if trade.window_index != window_index:
                still_pending.append(trade)
                continue

            # Determine outcome
            if trade.side == "YES":
                won = bnb_went_up
            else:  # NO
                won = not bnb_went_up

            trade.bnb_open = bnb_open
            trade.bnb_close = bnb_close
            trade.timestamp_exit = time.time()

            if won:
                # Payout: 1 USDC per share × shares bought
                # Shares = position_size_usdc / entry_price
                shares = trade.position_size_usdc / trade.entry_price
                payout = shares * 1.0  # 1 USDC per winning share
                trade.pnl_usdc = round(payout - trade.position_size_usdc, 4)
                trade.payout_per_share = 1.0
                trade.outcome = "WIN"
                self.metrics.wins += 1
                # Return capital + profit
                self.metrics.bankroll += trade.position_size_usdc + trade.pnl_usdc
            else:
                trade.pnl_usdc = -trade.position_size_usdc  # Lost entire stake
                trade.payout_per_share = 0.0
                trade.outcome = "LOSS"
                self.metrics.losses += 1
                # Capital already deducted; add nothing back

            self.metrics.pending -= 1
            self.metrics.total_pnl = round(
                self.metrics.total_pnl + trade.pnl_usdc, 4
            )

            resolved.append(trade)

            result_emoji = "✅" if won else "❌"
            logger.info(
                f"{result_emoji} Trade resolved: {trade.trade_id} | "
                f"{trade.side} | BNB {bnb_open:.2f}→{bnb_close:.2f} | "
                f"PnL: ${trade.pnl_usdc:+.2f} | "
                f"Bankroll: ${self.metrics.bankroll:.2f}"
            )

        self._pending_trades = still_pending

        if resolved:
            # Recompute avg_edge
            edges = [t.edge_at_entry for t in self._trades]
            self.metrics.avg_edge = sum(edges) / len(edges) if edges else 0.0
            self._save_trades()
            logger.info(f"📊 {self.metrics.summary()}")

    def print_summary(self):
        """Print current paper trading summary."""
        print(f"\n{'='*60}")
        print(f"  PAPER TRADING SUMMARY")
        print(f"{'='*60}")
        print(f"  Bankroll:    ${self.metrics.bankroll:.2f}")
        print(f"  Total PnL:   ${self.metrics.total_pnl:+.2f}")
        print(f"  Total trades: {self.metrics.total_trades}")
        print(f"  Win rate:    {self.metrics.win_rate:.1%}")
        print(f"  Wins:        {self.metrics.wins}")
        print(f"  Losses:      {self.metrics.losses}")
        print(f"  Pending:     {self.metrics.pending}")
        print(f"  ROI:         {self.metrics.roi:.2%}")
        print(f"  Avg edge:    {self.metrics.avg_edge:.3f}")
        print(f"{'='*60}\n")
