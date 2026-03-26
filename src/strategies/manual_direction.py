"""
manual_direction.py — Manual Direction Strategy

User picks UP or DOWN at launch. Bot bets that side every round,
but ONLY when the pool ratio makes it profitable (our side < 50% of pool).
"""

import logging
import time
from typing import Optional

from .base import BaseStrategy
from strategy import Signal, WindowInfo, kelly_fraction

logger = logging.getLogger(__name__)

PANCAKE_FEE = 0.03


class ManualDirectionStrategy(BaseStrategy):
    """Fixed direction (UP or DOWN), filtered by favorable pool ratio."""

    def __init__(self, config: dict):
        super().__init__(config)
        cfg = config.get("strategy", {})
        self.min_payout = cfg.get("manual_min_payout", 1.5)
        self.pool_min_bnb = cfg.get("pool_min_bnb", 0.3)
        self.direction: Optional[str] = None  # "YES" or "NO", set by prompt

    @property
    def name(self) -> str:
        dir_label = "UP" if self.direction == "YES" else ("DOWN" if self.direction == "NO" else "?")
        return f"🎯 Manual Direction ({dir_label})"

    def prompt_direction(self):
        """Ask the user to pick UP or DOWN. Called before bot starts."""
        print("\n" + "=" * 40)
        print("  Manual Direction Strategy")
        print("=" * 40)
        print("  [U] Always bet UP (bull)")
        print("  [D] Always bet DOWN (bear)")
        print("=" * 40)

        while True:
            choice = input("Your direction: ").strip().upper()
            if choice in ("U", "UP"):
                self.direction = "YES"
                break
            elif choice in ("D", "DOWN"):
                self.direction = "NO"
                break
            print("Invalid. Enter U or D.")

        label = "UP 🟢" if self.direction == "YES" else "DOWN 🔴"
        print(f"  → Locked in: {label}\n")

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

        if self.direction is None:
            self.last_skip_reason = "⏸ Direction not set"
            return None

        if not window.is_entry_window and window.seconds_remaining > self.entry_window_seconds:
            return None

        if pool_total_bnb < self.pool_min_bnb:
            self.last_skip_reason = f"⏸ Pool too small ({pool_total_bnb:.2f} BNB)"
            return None

        if pool_total_bnb == 0:
            return None

        # Check if our side is < 50% of pool (favorable payout)
        if self.direction == "YES":
            our_side_bnb = pool_bull_bnb
            our_ratio = pool_bull_bnb / pool_total_bnb
        else:
            our_side_bnb = pool_bear_bnb
            our_ratio = pool_bear_bnb / pool_total_bnb

        if our_ratio >= 0.50:
            self.last_skip_reason = (
                f"⏸ Our side too crowded ({our_ratio:.0%} ≥ 50%)"
            )
            return None

        # Calculate payout
        payout = (pool_total_bnb * (1 - PANCAKE_FEE)) / our_side_bnb if our_side_bnb > 0 else 0

        if payout < self.min_payout:
            self.last_skip_reason = (
                f"⏸ Payout too low (×{payout:.1f}, min ×{self.min_payout})"
            )
            return None

        edge = (payout - 1.0) / payout  # normalized edge from payout
        p_win = 0.50  # no directional prediction
        odds = payout - 1.0

        raw_k = kelly_fraction(p_win, odds)
        if raw_k <= 0:
            return None

        frac = min(raw_k, self.kelly_fraction_cap)
        pos_size = min(self.bankroll * frac, self.max_position_usdc)
        pos_size = max(0.0, round(pos_size, 2))

        if pos_size <= 0:
            return None

        dir_label = "UP" if self.direction == "YES" else "DOWN"
        logger.info(
            f"ManualDirection eval: {dir_label} | our_ratio={our_ratio:.0%} | "
            f"payout=×{payout:.1f} | edge={edge:.3f}"
        )

        signal = Signal(
            side=self.direction,
            edge=edge,
            p_up=0.50,
            yes_price=yes_price,
            kelly_fraction=raw_k,
            position_size_usdc=pos_size,
            timestamp=time.time(),
            is_mock=is_mock_data,
            bull_pct=pool_bull_bnb / pool_total_bnb if pool_total_bnb > 0 else 0.0,
            bear_pct=pool_bear_bnb / pool_total_bnb if pool_total_bnb > 0 else 0.0,
        )
        logger.info(f"🎯 Signal generated: {signal}")
        return signal
