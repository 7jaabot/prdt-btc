"""
combined.py — Combined Strategy (consensus of multiple strategies)

Runs N strategies in parallel. Only trades when ALL agree on the same direction.
Each sub-strategy can have its own edge range filter.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from .base import BaseStrategy
from strategy import Signal, WindowInfo

logger = logging.getLogger(__name__)


class EdgeFilter:
    """Filters signals by edge range (min/max bounds)."""

    def __init__(self, min_edge: Optional[float] = None, max_edge: Optional[float] = None):
        self.min_edge = min_edge
        self.max_edge = max_edge

    def passes(self, edge: float) -> bool:
        if self.min_edge is not None and edge < self.min_edge:
            return False
        if self.max_edge is not None and edge > self.max_edge:
            return False
        return True

    def __repr__(self):
        parts = []
        if self.min_edge is not None:
            parts.append(f">{self.min_edge}")
        if self.max_edge is not None:
            parts.append(f"<{self.max_edge}")
        return " ".join(parts) if parts else "any"

    @staticmethod
    def parse(text: str) -> 'EdgeFilter':
        """
        Parse edge filter from user input.

        Examples:
            ""          → no filter (any edge)
            ">0.1"      → edge > 0.1
            "<0.3"      → edge < 0.3
            ">0.1 <0.3" → 0.1 < edge < 0.3
        """
        text = text.strip()
        if not text:
            return EdgeFilter()

        min_edge = None
        max_edge = None

        for part in text.split():
            part = part.strip()
            if part.startswith(">"):
                try:
                    min_edge = float(part[1:])
                except ValueError:
                    pass
            elif part.startswith("<"):
                try:
                    max_edge = float(part[1:])
                except ValueError:
                    pass

        return EdgeFilter(min_edge=min_edge, max_edge=max_edge)


class CombinedStrategy(BaseStrategy):
    """
    Consensus strategy: runs multiple sub-strategies and trades only when
    ALL agree on the same direction (UP or DOWN).

    Each sub-strategy has its own optional EdgeFilter.
    """

    def __init__(self, config: dict, strategies: list, edge_filters: Optional[dict] = None):
        """
        Args:
            config: Global config dict.
            strategies: List of (name, BaseStrategy instance) tuples.
            edge_filters: Dict mapping strategy name → EdgeFilter (optional).
        """
        super().__init__(config)
        self.strategies = strategies  # [(name, instance), ...]
        self.edge_filters = edge_filters or {}

    @property
    def name(self) -> str:
        names = [s[0] for s in self.strategies]
        return "🔗 Combined: " + " + ".join(names)

    def update_bankroll(self, new_bankroll: float):
        """Propagate bankroll to all sub-strategies."""
        self.bankroll = new_bankroll
        for _, strat in self.strategies:
            strat.update_bankroll(new_bankroll)

    def prefetch(self, prices: list[float], epoch=None) -> None:
        """
        Pre-fetch data for all sub-strategies in parallel.

        Each sub-strategy's prefetch() is called concurrently via a thread pool.
        This minimises wall-clock time for the pre-load phase.
        """
        super().prefetch(prices, epoch)

        def _do_prefetch(name_strat):
            name, strat = name_strat
            try:
                strat.prefetch(prices, epoch)
            except Exception as e:
                logger.warning(f"Combined prefetch: {name} failed — {e}")

        with ThreadPoolExecutor(max_workers=len(self.strategies), thread_name_prefix="prefetch") as pool:
            futures = {pool.submit(_do_prefetch, (name, strat)): name for name, strat in self.strategies}
            for future in as_completed(futures):
                name = futures[future]
                exc = future.exception()
                if exc:
                    logger.warning(f"Combined prefetch thread error for {name}: {exc}")

        logger.info(f"Combined prefetch complete for {len(self.strategies)} strategies")

    def evaluate(
        self,
        prices: list[float],
        yes_price: float,
        window: WindowInfo,
        is_mock_data: bool = False,
        pool_total_bnb: float = 0.0,
        pool_bull_bnb: float = 0.0,
        pool_bear_bnb: float = 0.0,
    ) -> Optional[Signal]:
        self.last_skip_reason = None

        if not window.is_entry_window and window.seconds_remaining > self.entry_window_seconds:
            return None

        # Evaluate all sub-strategies
        signals = {}
        for name, strat in self.strategies:
            sig = strat.evaluate(
                prices=prices,
                yes_price=yes_price,
                window=window,
                is_mock_data=is_mock_data,
                pool_total_bnb=pool_total_bnb,
                pool_bull_bnb=pool_bull_bnb,
                pool_bear_bnb=pool_bear_bnb,
            )

            if sig is None:
                skip = getattr(strat, 'last_skip_reason', 'no signal')
                self.last_skip_reason = f"⏸ {name}: {skip}"
                logger.info(f"Combined: {name} → no signal ({skip})")
                return None

            # Apply edge filter for this strategy
            edge_filter = self.edge_filters.get(name)
            if edge_filter and not edge_filter.passes(sig.edge):
                self.last_skip_reason = (
                    f"⏸ {name}: edge {sig.edge:.3f} outside filter ({edge_filter})"
                )
                logger.info(
                    f"Combined: {name} → edge {sig.edge:.3f} filtered out ({edge_filter})"
                )
                return None

            signals[name] = sig

        # Check consensus: all must agree on direction
        sides = set(sig.side for sig in signals.values())
        if len(sides) != 1:
            side_summary = ", ".join(f"{n}={s.side}" for n, s in signals.items())
            self.last_skip_reason = f"⏸ No consensus: {side_summary}"
            logger.info(f"Combined: split signals — {side_summary}")
            return None

        # Consensus reached — use average edge and the agreed side
        consensus_side = sides.pop()
        avg_edge = sum(s.edge for s in signals.values()) / len(signals)

        # Use the smallest position size (most conservative)
        min_pos = min(s.position_size_usdc for s in signals.values())

        # Use the first signal's p_up as reference
        first_sig = list(signals.values())[0]

        signal = Signal(
            side=consensus_side,
            edge=avg_edge,
            p_up=first_sig.p_up,
            yes_price=first_sig.yes_price,
            kelly_fraction=first_sig.kelly_fraction,
            position_size_usdc=min_pos,
            timestamp=time.time(),
            is_mock=is_mock_data,
            bull_pct=pool_bull_bnb / pool_total_bnb if pool_total_bnb > 0 else 0.0,
            bear_pct=pool_bear_bnb / pool_total_bnb if pool_total_bnb > 0 else 0.0,
        )

        names = " + ".join(signals.keys())
        logger.info(
            f"🎯 Combined consensus: {consensus_side} | strategies: {names} | "
            f"avg_edge={avg_edge:.3f} | size=${min_pos:.2f}"
        )
        return signal
