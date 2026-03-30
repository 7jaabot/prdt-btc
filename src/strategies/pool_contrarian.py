"""
pool_contrarian.py — Pool Contrarian Strategy

Bets AGAINST the crowd when the PancakeSwap pool is heavily imbalanced.
If 90% bet bull, the bear payout is ~×10 — and P(bear) ≈ 50% on 5min crypto.
Pure pool exploitation: no price prediction, no technical analysis.
"""

import logging
import time
from typing import Optional

from .base import BaseStrategy
from strategy import Signal, WindowInfo

logger = logging.getLogger(__name__)

PANCAKE_FEE = 0.03


class PoolContrarianStrategy(BaseStrategy):
    """Bet against the crowd when pool is heavily one-sided."""

    def __init__(self, config: dict):
        super().__init__(config)
        cfg = config.get("strategy", {})
        self.imbalance_threshold = cfg.get("pool_imbalance_threshold", 0.75)
        self.min_payout = cfg.get("pool_min_payout", 2.5)
        self.pool_min_bnb = cfg.get("pool_min_bnb", 0.3)

    @property
    def name(self) -> str:
        return "🔄 Pool Contrarian"

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

        # Determine if pool is imbalanced enough
        if bull_ratio >= self.imbalance_threshold:
            # Crowd is bullish → bet bear (contrarian)
            side = "NO"
            our_side_bnb = pool_bear_bnb
            payout = (pool_total_bnb * (1 - PANCAKE_FEE)) / pool_bear_bnb if pool_bear_bnb > 0 else 0
            crowd_pct = bull_ratio
        elif bear_ratio >= self.imbalance_threshold:
            # Crowd is bearish → bet bull (contrarian)
            side = "YES"
            our_side_bnb = pool_bull_bnb
            payout = (pool_total_bnb * (1 - PANCAKE_FEE)) / pool_bull_bnb if pool_bull_bnb > 0 else 0
            crowd_pct = bear_ratio
        else:
            self.last_skip_reason = (
                f"⏸ Pool balanced (bull={bull_ratio:.0%} bear={bear_ratio:.0%}, "
                f"need >{self.imbalance_threshold:.0%})"
            )
            return None

        # NOTE: min_payout check removed — payout directly computes edge below.
        # A low payout → low edge → naturally filtered by edge_threshold.
        # Safety guard: payout must be > 0 to avoid division issues (guaranteed
        # by the bear_bnb/bull_bnb > 0 condition above, but be explicit).
        if payout <= 0:
            self.last_skip_reason = "⏸ Invalid payout (division by zero guard)"
            return None

        # Edge = payout attractiveness. P(win)=0.50 assumed (no directional prediction).
        # Formula: Kelly-derived edge = 0.5 - 0.5/payout → spreads in [0, 0.50]
        #   payout=2x → 0.00 (fair), payout=3x → 0.17, payout=4x → 0.25,
        #   payout=10x → 0.40, payout=∞ → 0.50
        edge = 0.5 - (0.5 / payout)  # always in [0, 0.50]

        # Apply edge threshold guard
        if edge <= self.edge_threshold:
            self.last_skip_reason = (
                f"⏸ PoolContrarian: edge too low ({edge:.3f} ≤ {self.edge_threshold}) | "
                f"payout=×{payout:.1f}"
            )
            return None

        # Flat position sizing with guards
        pool_total_usdc = pool_total_bnb * prices[-1] if prices else 0.0
        caps = [self.bankroll * self.max_bankroll_pct]
        if pool_total_usdc > 0:
            caps.append(pool_total_usdc * self.max_pool_pct)
        pos_size = min(self.position_size_usdc, *caps)
        if pos_size < self.min_position_usdc:
            return None
        pos_size = max(0.0, round(pos_size, 2))

        if pos_size <= 0:
            return None

        logger.info(
            f"PoolContrarian eval: crowd={crowd_pct:.0%} on {'BULL' if side == 'NO' else 'BEAR'} | "
            f"payout=×{payout:.1f} | side={side} | edge={edge:.3f}"
        )

        signal = Signal(
            side=side,
            edge=edge,
            p_up=0.50,
            yes_price=yes_price,
            kelly_fraction=0.0,
            position_size_usdc=pos_size,
            timestamp=time.time(),
            is_mock=is_mock_data,
            bull_pct=pool_bull_bnb / pool_total_bnb if pool_total_bnb > 0 else 0.0,
            bear_pct=pool_bear_bnb / pool_total_bnb if pool_total_bnb > 0 else 0.0,
        )
        logger.info(f"🎯 Signal generated: {signal}")
        return signal
