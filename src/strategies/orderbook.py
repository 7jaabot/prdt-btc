"""
orderbook.py — Order Book Imbalance Strategy

Fetches Binance BNB/USDT depth and measures bid/ask volume imbalance.
More bids than asks → buying pressure → bet UP.
More asks than bids → selling pressure → bet DOWN.
"""

import logging
import time
from typing import Optional

import requests

from .base import BaseStrategy
from strategy import Signal, WindowInfo, compute_position_size, kelly_fraction

logger = logging.getLogger(__name__)


class OrderBookStrategy(BaseStrategy):
    """Bid/ask imbalance from Binance order book."""

    def __init__(self, config: dict):
        super().__init__(config)
        cfg = config.get("strategy", {})
        self.imbalance_threshold = cfg.get("orderbook_imbalance_threshold", 0.15)
        self.depth_levels = cfg.get("orderbook_depth_levels", 20)

    @property
    def name(self) -> str:
        return "📊 Order Book Imbalance"

    def _fetch_depth(self) -> Optional[tuple[float, float]]:
        """Fetch bid/ask volumes from Binance. Returns (bid_vol, ask_vol) or None."""
        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/depth",
                params={"symbol": "BNBUSDT", "limit": self.depth_levels},
                timeout=5,
            )
            data = resp.json()
            bid_vol = sum(float(b[1]) for b in data.get("bids", []))
            ask_vol = sum(float(a[1]) for a in data.get("asks", []))
            return bid_vol, ask_vol
        except Exception as e:
            logger.warning(f"Order book fetch failed: {e}")
            return None

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

        result = self._fetch_depth()
        if result is None:
            self.last_skip_reason = "⏸ Could not fetch order book"
            return None

        bid_vol, ask_vol = result
        total = bid_vol + ask_vol
        if total == 0:
            self.last_skip_reason = "⏸ Empty order book"
            return None

        imbalance = (bid_vol - ask_vol) / total  # [-1, 1]

        if abs(imbalance) <= self.imbalance_threshold:
            self.last_skip_reason = (
                f"⏸ Imbalance too weak ({imbalance:+.3f}, threshold={self.imbalance_threshold})"
            )
            return None

        side = "YES" if imbalance > 0 else "NO"
        edge = abs(imbalance) - self.imbalance_threshold
        p_win = 0.5 + abs(imbalance) * 0.3  # rough: stronger imbalance → higher p_win, cap ~0.8

        # Kelly sizing
        odds = 1.0  # even money approximation
        raw_k = kelly_fraction(p_win, odds)
        if raw_k <= 0:
            self.last_skip_reason = f"⏸ Kelly negative (p_win={p_win:.2f})"
            return None

        frac = min(raw_k, self.kelly_fraction_cap)
        pos_size = min(self.bankroll * frac, self.max_position_usdc)
        pos_size = max(0.0, round(pos_size, 2))

        if pos_size <= 0:
            return None

        logger.info(
            f"OrderBook eval: imbalance={imbalance:+.3f} | bid={bid_vol:.1f} ask={ask_vol:.1f} | "
            f"side={side} | edge={edge:.3f}"
        )

        signal = Signal(
            side=side,
            edge=edge,
            p_up=0.5 + imbalance * 0.3,
            yes_price=0.50,
            kelly_fraction=raw_k,
            position_size_usdc=pos_size,
            timestamp=time.time(),
            is_mock=is_mock_data,
            bull_pct=pool_bull_bnb / pool_total_bnb if pool_total_bnb > 0 else 0.0,
            bear_pct=pool_bear_bnb / pool_total_bnb if pool_total_bnb > 0 else 0.0,
        )
        logger.info(f"🎯 Signal generated: {signal}")
        return signal
