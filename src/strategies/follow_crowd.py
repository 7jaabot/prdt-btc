"""
follow_crowd.py — Follow the Crowd Strategy

Bets on the side with the most weight in the PancakeSwap pool.
If bear > bull → bet DOWN. If bull > bear → bet UP.
Only condition: the opposite side must not be empty.
"""

import logging
import time
from typing import Optional

from .base import BaseStrategy
from strategy import Signal, WindowInfo, kelly_fraction

logger = logging.getLogger(__name__)

PANCAKE_FEE = 0.03


class FollowCrowdStrategy(BaseStrategy):
    """Bet with the majority side of the pool."""

    def __init__(self, config: dict):
        super().__init__(config)
        cfg = config.get("strategy", {})
        self.pool_min_bnb = cfg.get("pool_min_bnb", 0.3)
        self.min_majority_pct = cfg.get("follow_min_majority_pct", 0.55)

    @property
    def name(self) -> str:
        return "👥 Follow the Crowd"

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

        if pool_total_bnb < self.pool_min_bnb:
            self.last_skip_reason = f"⏸ Pool too small ({pool_total_bnb:.2f} BNB)"
            return None

        if pool_total_bnb == 0:
            return None

        bull_ratio = pool_bull_bnb / pool_total_bnb
        bear_ratio = pool_bear_bnb / pool_total_bnb

        # Follow the majority
        if bull_ratio > bear_ratio:
            side = "YES"
            majority_pct = bull_ratio
            opposite_bnb = pool_bear_bnb
        else:
            side = "NO"
            majority_pct = bear_ratio
            opposite_bnb = pool_bull_bnb

        # Opposite side must not be empty
        if opposite_bnb < 0.001:
            self.last_skip_reason = f"⏸ Opposite side empty ({opposite_bnb:.4f} BNB)"
            return None

        # NOTE: min_majority_pct check removed — majority_pct is directly used to
        # compute edge below. A weak majority will produce a low edge that naturally
        # fails the edge_threshold check. Let the edge be the sole filter.

        # Edge: how much the crowd agrees (more agreement = higher edge)
        edge = (majority_pct - 0.50) * 0.5  # scale: 60%→0.05, 70%→0.10, 90%→0.20

        # p_win: slight bias toward crowd being right
        p_win = 0.50 + (majority_pct - 0.50) * 0.2  # 60%→0.52, 80%→0.56
        p_win = min(p_win, 0.65)

        odds = 1.0
        raw_k = kelly_fraction(p_win, odds)
        if raw_k <= 0:
            return None

        frac = min(raw_k, self.kelly_fraction_cap)
        pos_size = min(self.bankroll * frac, self.max_position_usdc)
        pos_size = max(0.0, round(pos_size, 2))

        if pos_size <= 0:
            return None

        dir_label = "UP" if side == "YES" else "DOWN"
        logger.info(
            f"FollowCrowd eval: crowd={majority_pct:.0%} {dir_label} | "
            f"edge={edge:.3f} | p_win={p_win:.2f}"
        )

        signal = Signal(
            side=side,
            edge=edge,
            p_up=p_win if side == "YES" else 1.0 - p_win,
            yes_price=yes_price,
            kelly_fraction=raw_k,
            position_size_usdc=pos_size,
            timestamp=time.time(),
            is_mock=is_mock_data,
            bull_pct=bull_ratio,
            bear_pct=bear_ratio,
        )
        logger.info(f"🎯 Signal generated: {signal}")
        return signal
