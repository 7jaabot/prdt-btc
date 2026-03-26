"""
paper_trader.py — Paper trading simulator

Simulates trades without real money or API calls.
Logs all simulated trades to logs/<strategy>/paper_trades.json.
Tracks PnL, win rate, and edge metrics.

PnL is calculated using on-chain PancakeSwap round data (rewardAmount /
rewardBaseCalAmount) — identical to LiveTrader — so paper results are
directly comparable to live results.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional

from strategy import Signal, WindowInfo

logger = logging.getLogger(__name__)

PANCAKE_FEE = 0.03


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

    # On-chain fields (mirrors LiveTrade)
    epoch: int = 0                        # PancakeSwap epoch number
    bet_bnb: float = 0.0                  # BNB equivalent of the bet
    bnb_price_at_entry: float = 0.0      # BNB/USD at entry time

    # Pool state at entry
    bull_pct: float = 0.0                 # % of pool on bull side at entry
    bear_pct: float = 0.0                 # % of pool on bear side at entry

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
    Resolves trades after each window ends using on-chain PancakeSwap data
    (same PnL calculation as LiveTrader).
    Persists all trades to a JSON log file.
    """

    def __init__(self, config: dict):
        cfg_paper = config.get("paper_trading", {})
        cfg_strategy = config.get("strategy", {})

        self.log_file = cfg_paper.get("log_file", "logs/paper/trades.json")
        self.simulate_latency_ms = cfg_paper.get("simulate_latency_ms", 200)
        self.starting_bankroll = cfg_strategy.get("starting_bankroll_usdc", 1000.0)

        self.metrics = PaperMetrics(bankroll=self.starting_bankroll)
        self._trades: list[Trade] = []
        self._pending_trades: list[Trade] = []
        self._trade_counter = 0

        # Injected externally by main.py (same as LiveTrader)
        self._pancake_client = None
        self._binance_feed = None

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
                # Filter to only known Trade fields so old JSON with extra keys doesn't break
                known_fields = Trade.__dataclass_fields__.keys()
                filtered = {k: v for k, v in t_dict.items() if k in known_fields}
                trade = Trade(**filtered)
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
        self._export_csv()

    def _export_csv(self):
        """Export all trades to a CSV file alongside the JSON log (for Google Sheets analysis)."""
        import csv
        from datetime import datetime

        csv_path = self.log_file.replace(".json", ".csv")
        os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)

        columns = [
            "trade_id", "epoch", "timestamp_entry", "time_entry", "timestamp_exit", "time_exit",
            "side", "side_label", "edge_at_entry", "p_up_at_entry", "kelly_fraction",
            "position_size_usdc", "bet_bnb", "bnb_price_at_entry",
            "bull_pct", "bear_pct", "bnb_open", "bnb_close", "outcome", "pnl_usdc", "payout_per_share", "is_mock",
        ]

        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                for t in self._trades:
                    row = t.to_dict()
                    row["time_entry"] = datetime.fromtimestamp(t.timestamp_entry).strftime("%Y-%m-%d %H:%M:%S") if t.timestamp_entry else ""
                    row["time_exit"] = datetime.fromtimestamp(t.timestamp_exit).strftime("%Y-%m-%d %H:%M:%S") if t.timestamp_exit else ""
                    row["side_label"] = "UP" if t.side == "YES" else "DOWN"
                    writer.writerow(row)
            logger.debug(f"CSV exported to {csv_path}")
        except Exception as e:
            logger.warning(f"CSV export failed: {e}")

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

        # Get current BNB price from the injected Binance feed
        bnb_price = self._binance_feed.last_price if self._binance_feed else None
        if not bnb_price or bnb_price <= 0:
            logger.warning("BNB price unavailable — bet_bnb will be 0.0 for this trade.")
            bnb_price = 0.0

        bet_bnb = signal.position_size_usdc / bnb_price if bnb_price > 0 else 0.0

        # Simulate latency impact on entry price (slight slippage).
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
            bull_pct=signal.bull_pct,
            bear_pct=signal.bear_pct,
            # On-chain fields
            epoch=window.window_index,
            bet_bnb=round(bet_bnb, 8),
            bnb_price_at_entry=bnb_price,
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
            f"📝 Paper trade entered{mock_tag}: {trade_id} | "
            f"{trade.side} @ epoch={trade.epoch} | "
            f"{bet_bnb:.6f} BNB (${trade.position_size_usdc:.2f}) | "
            f"edge={trade.edge_at_entry:.3f}"
        )

        return trade

    def _fetch_round_pnl(self, trade: Trade, bnb_price: float):
        """
        Fetch on-chain round data for a completed epoch and compute real PnL.

        Mirrors LiveTrader._fetch_round_pnl() — reads PancakeSwap reward pool
        to determine the actual payout multiplier.

        PancakeSwap Prediction V2 payout mechanics:
          - rewardBaseCalAmount = total BNB bet on the winning side
          - rewardAmount        = total payout pool (totalAmount * (1 - fee))
          - payout for a winner = bet_bnb * rewardAmount / rewardBaseCalAmount
          - pnl_bnb = payout - bet_bnb
          - pnl_usdc = pnl_bnb * bnb_price

        Returns:
            (pnl_usdc, payout_per_bnb) or (None, None) if data not yet available.
        """
        if self._pancake_client is None:
            logger.debug(
                f"No PancakeClient injected — cannot compute on-chain PnL for epoch {trade.epoch}."
            )
            return None, None

        try:
            round_data = self._pancake_client.get_round_by_epoch(trade.epoch)
        except Exception as e:
            logger.warning(f"Could not fetch on-chain round data for epoch {trade.epoch}: {e}")
            return None, None

        if round_data is None:
            logger.warning(f"No round data returned for epoch {trade.epoch}")
            return None, None

        reward_base = round_data.reward_base_cal_amount
        reward_pool = round_data.reward_amount
        bet_bnb = trade.bet_bnb

        if reward_base <= 0 or reward_pool <= 0:
            logger.warning(
                f"Epoch {trade.epoch}: reward fields not yet populated on-chain "
                f"(rewardBaseCalAmount={reward_base}, rewardAmount={reward_pool}). "
                f"Will retry at next resolution cycle."
            )
            return None, None

        payout_bnb = bet_bnb * reward_pool / reward_base
        pnl_bnb = payout_bnb - bet_bnb
        pnl_usdc = round(pnl_bnb * bnb_price, 4)
        payout_per_bnb = payout_bnb / bet_bnb if bet_bnb > 0 else 0.0

        logger.info(
            f"📡 On-chain PnL for epoch {trade.epoch}: "
            f"bet={bet_bnb:.6f} BNB | "
            f"rewardBase={reward_base:.6f} | rewardPool={reward_pool:.6f} | "
            f"payout={payout_bnb:.6f} BNB | "
            f"pnl_bnb={pnl_bnb:+.6f} | pnl_usdc=${pnl_usdc:+.4f}"
        )
        return pnl_usdc, payout_per_bnb

    def resolve_pending_on_startup(self):
        """
        Called at startup (after _pancake_client is injected by main.py) to resolve
        any trades that were left PENDING from a previous session.

        Mirrors LiveTrader.resolve_pending_on_startup() — without the claim logic.
        """
        if self._pancake_client is None:
            logger.info("resolve_pending_on_startup: no PancakeClient injected — skipping.")
            return

        if not self._pending_trades:
            logger.info("resolve_pending_on_startup: no PENDING trades to resolve.")
            return

        logger.info(
            f"resolve_pending_on_startup: resolving {len(self._pending_trades)} PENDING paper trade(s)..."
        )

        still_pending = []
        resolved_count = 0

        for trade in list(self._pending_trades):
            try:
                round_data = self._pancake_client.get_round_by_epoch(trade.epoch)
            except Exception as e:
                logger.warning(
                    f"resolve_pending_on_startup: could not fetch epoch {trade.epoch}: {e} — keeping PENDING"
                )
                still_pending.append(trade)
                continue

            if round_data is None:
                logger.warning(
                    f"resolve_pending_on_startup: no round data for epoch {trade.epoch} — keeping PENDING"
                )
                still_pending.append(trade)
                continue

            if not round_data.oracle_called:
                logger.info(
                    f"resolve_pending_on_startup: epoch {trade.epoch} not yet settled "
                    f"(oracleCalled=false) — keeping PENDING"
                )
                still_pending.append(trade)
                continue

            lock_price = round_data.lock_price
            close_price = round_data.close_price

            if lock_price is None or close_price is None:
                logger.warning(
                    f"resolve_pending_on_startup: epoch {trade.epoch} missing price data "
                    f"(lock={lock_price}, close={close_price}) — keeping PENDING"
                )
                still_pending.append(trade)
                continue

            # Determine win/loss (same logic as live)
            if trade.side == "YES":   # bull: wins if price went up
                won = close_price > lock_price
            else:                      # bear ("NO"): wins if price went down
                won = close_price < lock_price

            trade.bnb_open = lock_price
            trade.bnb_close = close_price
            trade.timestamp_exit = time.time()

            if won:
                pnl_usdc, payout_per_bnb = self._fetch_round_pnl(trade, trade.bnb_price_at_entry)

                if pnl_usdc is None:
                    logger.warning(
                        f"resolve_pending_on_startup: epoch {trade.epoch} WIN but reward "
                        f"pool not settled — keeping PENDING"
                    )
                    still_pending.append(trade)
                    continue

                trade.pnl_usdc = pnl_usdc
                trade.payout_per_share = round(payout_per_bnb, 6) if payout_per_bnb else 0.0
                trade.outcome = "WIN"
                self.metrics.wins += 1
            else:
                trade.pnl_usdc = round(-trade.bet_bnb * trade.bnb_price_at_entry, 4)
                trade.payout_per_share = 0.0
                trade.outcome = "LOSS"
                self.metrics.losses += 1

            self.metrics.pending -= 1
            self.metrics.total_pnl = round(self.metrics.total_pnl + trade.pnl_usdc, 4)

            result_emoji = "✅" if won else "❌"
            logger.info(
                f"{result_emoji} Startup resolved: {trade.trade_id} | "
                f"{trade.side} | epoch={trade.epoch} | "
                f"lock={lock_price:.2f} close={close_price:.2f} | "
                f"PnL: ${trade.pnl_usdc:+.4f}"
            )
            resolved_count += 1

        self._pending_trades = still_pending

        if resolved_count > 0:
            # Recompute bankroll and avg_edge from scratch for accuracy
            self._recompute_metrics()
            self._save_trades()
            logger.info(
                f"resolve_pending_on_startup: resolved {resolved_count} paper trade(s). "
                f"{self.metrics.summary()}"
            )

    def resolve_trades(self, window_index: int, bnb_open: float, bnb_close: float):
        """
        Resolve all pending trades for a completed window.

        PnL is calculated from on-chain PancakeSwap round data when available
        (same as LiveTrader). Falls back to keeping PENDING if on-chain data
        is not yet settled.

        Args:
            window_index: The window (epoch) that just completed.
            bnb_open: BNB price at window start.
            bnb_close: BNB price at window end.
        """
        resolved = []
        still_pending = []

        bnb_went_up = bnb_close > bnb_open
        bnb_price_at_close = bnb_close if bnb_close > 0 else bnb_open

        for trade in self._pending_trades:
            if trade.window_index != window_index:
                still_pending.append(trade)
                continue

            # Determine outcome (YES=bull wins if price went up)
            if trade.side == "YES":
                won = bnb_went_up
            else:  # NO = bear
                won = not bnb_went_up

            trade.bnb_open = bnb_open
            trade.bnb_close = bnb_close
            trade.timestamp_exit = time.time()

            if won:
                # Use real on-chain payout data
                pnl_usdc, payout_per_bnb = self._fetch_round_pnl(trade, bnb_price_at_close)

                if pnl_usdc is None:
                    # On-chain reward pool not yet settled — retry next epoch
                    logger.warning(
                        f"⏳ Paper trade {trade.trade_id} epoch {trade.epoch}: "
                        f"on-chain reward not settled yet. Keeping PENDING."
                    )
                    still_pending.append(trade)
                    continue

                trade.pnl_usdc = pnl_usdc
                trade.payout_per_share = round(payout_per_bnb, 6) if payout_per_bnb else 0.0
                trade.outcome = "WIN"
                self.metrics.wins += 1
                # Return committed capital + profit
                self.metrics.bankroll += trade.position_size_usdc + trade.pnl_usdc
            else:
                # Loss: cost = bet_bnb at current BNB price
                trade.pnl_usdc = round(-trade.bet_bnb * bnb_price_at_close, 4)
                trade.payout_per_share = 0.0
                trade.outcome = "LOSS"
                self.metrics.losses += 1
                # Stake already deducted at entry; adjust for BNB price drift
                self.metrics.bankroll += trade.position_size_usdc + trade.pnl_usdc

            self.metrics.pending -= 1
            self.metrics.total_pnl = round(
                self.metrics.total_pnl + trade.pnl_usdc, 4
            )

            resolved.append(trade)

            result_emoji = "✅" if won else "❌"
            logger.info(
                f"{result_emoji} Paper trade resolved: {trade.trade_id} | "
                f"{trade.side} | BNB {bnb_open:.2f}→{bnb_close:.2f} | "
                f"bet={trade.bet_bnb:.6f} BNB | PnL: ${trade.pnl_usdc:+.4f} | "
                f"Bankroll: ${self.metrics.bankroll:.2f}"
            )

        self._pending_trades = still_pending

        if resolved:
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
