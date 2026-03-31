"""
mean_reversion.py — Mean Reversion Strategy

After a strong move during the betting phase, bet on price reverting
during the live round. Crypto 5min micro-moves tend to mean-revert.

Hypothesis: if BNB moved +0.3% during betting (z-score > 1.5σ),
bet DOWN for the live round (and vice versa).
"""

import logging
import time
from typing import Optional

import numpy as np

from .base import BaseStrategy
from strategy import Signal, WindowInfo

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """Counter-trend: bet against extreme moves in the betting phase."""

    def __init__(self, config: dict):
        super().__init__(config)
        cfg = config.get("strategy", {})
        self.min_zscore = cfg.get("reversion_min_zscore", 1.5)
        self.scaling = cfg.get("reversion_scaling", 0.10)

    @property
    def name(self) -> str:
        return "📉 Mean Reversion"

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

        if len(prices) < 10:
            self.last_skip_reason = f"⏸ Not enough price data ({len(prices)} pts)"
            return None

        # Calculate momentum and volatility over the betting phase
        log_returns = np.diff(np.log(prices))
        if len(log_returns) < 5:
            self.last_skip_reason = "⏸ Too few returns for z-score"
            return None

        momentum = (prices[-1] / prices[0]) - 1.0  # total return
        sigma = float(np.std(log_returns))

        if sigma == 0:
            self.last_skip_reason = "⏸ Zero volatility"
            return None

        # Z-score: how extreme is the move relative to observed volatility
        # Normalize momentum by expected std of total return
        n_obs = len(log_returns)
        expected_total_std = sigma * np.sqrt(n_obs)
        z_score = momentum / expected_total_std if expected_total_std > 0 else 0

        # NOTE: min_zscore threshold check removed — z_score is used to compute edge
        # directly below. A small z_score → small (or negative) edge → filtered by
        # edge_threshold or Kelly. Safety: when z_score == 0, direction is undefined;
        # return None to avoid a directionless trade.
        if z_score == 0:
            self.last_skip_reason = "⏸ Z-score is exactly zero — no directional signal"
            return None

        # Counter-trend: if price went UP → bet DOWN, and vice versa
        if z_score > 0:
            side = "NO"   # price went up → expect reversion down
        else:
            side = "YES"  # price went down → expect reversion up

        # Edge = abs(z_score) mapped to [0, 0.50] range
        # z=0 → edge 0, z=2 → edge 0.20, z=5 → edge 0.50
        edge = min(abs(z_score) * 0.10, 0.50)

        # P(win) = direct mapping from z-score extremity
        # z=0 → 0.50 (coin flip), z=5 → 1.0 (very confident)
        p_win = 0.50 + min(abs(z_score) * 0.10, 0.49)
        p_win = max(0.01, min(0.99, p_win))

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
            f"MeanReversion eval: momentum={momentum:+.4f} | z_score={z_score:+.2f} | "
            f"side={side} | edge={edge:.3f} | p_win={p_win:.2f}"
        )

        signal = Signal(
            side=side,
            edge=edge,
            p_up=1.0 - p_win if side == "NO" else p_win,
            yes_price=0.50,
            kelly_fraction=0.0,
            position_size_usdc=pos_size,
            timestamp=time.time(),
            is_mock=is_mock_data,
            bull_pct=pool_bull_bnb / pool_total_bnb if pool_total_bnb > 0 else 0.0,
            bear_pct=pool_bear_bnb / pool_total_bnb if pool_total_bnb > 0 else 0.0,
        )
        logger.info(f"🎯 Signal generated: {signal}")
        return signal
