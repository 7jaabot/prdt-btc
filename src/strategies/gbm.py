"""
gbm.py — GBM Momentum Strategy

Estimates P(close > lock) via Geometric Brownian Motion Monte Carlo,
then computes edge vs market price and sizes via Kelly criterion.
"""

import logging
import time
from typing import Optional

from .base import BaseStrategy
from strategy import (
    Signal, WindowInfo,
    estimate_p_up_momentum, compute_edge, compute_position_size,
)

logger = logging.getLogger(__name__)


class GBMStrategy(BaseStrategy):
    """
    Last-Second Momentum Strategy using GBM Monte Carlo.

    Estimates P(close_price > lock_price) from observed price drift/volatility,
    then trades if edge exceeds threshold.
    """

    FAIR_ODDS_PRICE = 0.50

    def __init__(self, config: dict):
        super().__init__(config)
        cfg = config.get("strategy", {})
        self.use_fair_odds = cfg.get("use_fair_odds", True)
        self.min_prob_diff = cfg.get("min_prob_diff", 0.03)

    @property
    def name(self) -> str:
        return "📈 GBM Momentum"

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

        if len(prices) < 5:
            self.last_skip_reason = f"⏸ Not enough price data ({len(prices)} pts)"
            return None

        # Estimate P(close > lock) via GBM Monte Carlo
        p_up = estimate_p_up_momentum(
            prices=prices,
            seconds_to_lock=window.seconds_remaining,
            round_duration=300.0,
        )

        # Pool quality filters (only when trading against real pool prices)
        if not self.use_fair_odds:
            if pool_total_bnb < self.min_pool_bnb:
                self.last_skip_reason = (
                    f"⏸ Pool too small ({pool_total_bnb:.2f} BNB < {self.min_pool_bnb:.1f} min)"
                )
                logger.info(
                    f"Pool too small: {pool_total_bnb:.3f} BNB "
                    f"(min={self.min_pool_bnb:.1f}) — skipping"
                )
                return None

        # Fair-odds mode: use 0.50 as reference price
        if self.use_fair_odds:
            effective_yes_price = self.FAIR_ODDS_PRICE
        else:
            effective_yes_price = yes_price

        edge, side = compute_edge(p_up, effective_yes_price)

        logger.info(
            f"Strategy eval: P(Up)={p_up:.3f} | "
            f"effective_price={effective_yes_price:.3f} | "
            f"edge={edge:.3f} | side={side} | remaining={window.seconds_remaining:.1f}s"
        )

        if edge <= self.edge_threshold:
            self.last_skip_reason = (
                f"⏸ Edge too low ({edge:.3f} ≤ {self.edge_threshold:.2f}) | "
                f"P(Up)={p_up:.2f} | {side}"
            )
            return None

        pool_total_usdc = pool_total_bnb * prices[-1] if prices else 0.0
        raw_k, pos_size = compute_position_size(
            edge=edge,
            p_up=p_up,
            side=side,
            yes_price=effective_yes_price,
            bankroll=self.bankroll,
            position_size_usdc=self.position_size_usdc,
        
            min_usdc=self.min_position_usdc,
            max_bankroll_pct=self.max_bankroll_pct,
            max_pool_pct=self.max_pool_pct,
            pool_total_usdc=pool_total_usdc,
        )

        if pos_size <= 0:
            return None

        # Pool-based filters (when not using fair odds)
        if not self.use_fair_odds and prices:
            opposite_bnb = pool_bear_bnb if side == "YES" else pool_bull_bnb
            if opposite_bnb < 0.01:
                self.last_skip_reason = (
                    f"⏸ No counterparty "
                    f"({'BEAR' if side == 'YES' else 'BULL'} side empty)"
                )
                return None

            bet_bnb = pos_size / prices[-1]
            side_bnb = pool_bull_bnb if side == "YES" else pool_bear_bnb
            if side_bnb > 0 and bet_bnb / side_bnb > self.max_bet_share_of_side:
                self.last_skip_reason = (
                    f"⏸ Bet too large ({bet_bnb:.3f} BNB = "
                    f"{bet_bnb/side_bnb:.0%} of {side} side)"
                )
                return None

        signal = Signal(
            side=side,
            edge=edge,
            p_up=p_up,
            yes_price=effective_yes_price,
            kelly_fraction=0.0,
            position_size_usdc=pos_size,
            timestamp=time.time(),
            is_mock=is_mock_data,
            bull_pct=pool_bull_bnb / pool_total_bnb if pool_total_bnb > 0 else 0.0,
            bear_pct=pool_bear_bnb / pool_total_bnb if pool_total_bnb > 0 else 0.0,
        )

        logger.info(f"🎯 Signal generated: {signal}")
        return signal
