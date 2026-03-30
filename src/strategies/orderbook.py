"""
orderbook.py — Order Book Imbalance Strategy (v2)

Fetches Binance BNB/USDT depth (100 levels) and measures bid/ask volume imbalance,
weighted by proximity to the mid price (closer orders = more weight).

v2 changes vs v1 (commit 430dd43):
  - Depth increased from 20 to 100 levels
  - Distance-weighted imbalance (orders near mid count more)
  - p_win capped at 0.60
  - Edge capped at 0.20
  - 2-snapshot averaging (5s apart) for noise reduction
"""

import logging
import time
from typing import Optional

import requests

from .base import BaseStrategy
from strategy import Signal, WindowInfo

logger = logging.getLogger(__name__)


class OrderBookStrategy(BaseStrategy):
    """Bid/ask imbalance from Binance order book (distance-weighted, multi-snapshot)."""

    def __init__(self, config: dict):
        super().__init__(config)
        cfg = config.get("strategy", {})
        self.imbalance_threshold = cfg.get("orderbook_imbalance_threshold", 0.15)
        self.depth_levels = cfg.get("orderbook_depth_levels", 100)
        self._last_imbalance: Optional[float] = None
        self._last_fetch_ts: float = 0.0

    @property
    def name(self) -> str:
        return "📊 Order Book Imbalance"

    def _fetch_weighted_imbalance(self) -> Optional[float]:
        """
        Fetch order book and compute distance-weighted imbalance.

        Orders closer to mid price get exponentially more weight:
          weight = 1 / (1 + distance_pct * 100)

        Returns imbalance in [-1, 1] or None on failure.
        """
        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/depth",
                params={"symbol": "BNBUSDT", "limit": self.depth_levels},
                timeout=5,
            )
            data = resp.json()
            bids = data.get("bids", [])
            asks = data.get("asks", [])

            if not bids or not asks:
                return None

            # Mid price
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            mid = (best_bid + best_ask) / 2.0

            if mid <= 0:
                return None

            # Distance-weighted volume
            weighted_bid = 0.0
            for price_str, qty_str in bids:
                price, qty = float(price_str), float(qty_str)
                distance_pct = abs(mid - price) / mid
                weight = 1.0 / (1.0 + distance_pct * 100)  # near = ~1.0, far = ~0.01
                weighted_bid += qty * weight

            weighted_ask = 0.0
            for price_str, qty_str in asks:
                price, qty = float(price_str), float(qty_str)
                distance_pct = abs(price - mid) / mid
                weight = 1.0 / (1.0 + distance_pct * 100)
                weighted_ask += qty * weight

            total = weighted_bid + weighted_ask
            if total == 0:
                return None

            return (weighted_bid - weighted_ask) / total

        except Exception as e:
            logger.warning(f"Order book fetch failed: {e}")
            return None

    def _get_averaged_imbalance(self) -> Optional[float]:
        """
        Average two snapshots ~5s apart for noise reduction.

        On the first call in a window, fetch snapshot 1 and cache it.
        On the second call (~5s later from the next tick), fetch snapshot 2 and average.
        """
        now = time.time()
        current = self._fetch_weighted_imbalance()

        if current is None:
            return self._last_imbalance  # fallback to cached

        if self._last_imbalance is not None and (now - self._last_fetch_ts) < 15:
            # Average with previous snapshot
            averaged = (self._last_imbalance + current) / 2.0
            self._last_imbalance = current
            self._last_fetch_ts = now
            return averaged
        else:
            # First snapshot or too stale — just use current
            self._last_imbalance = current
            self._last_fetch_ts = now
            return current

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

        imbalance = self._get_averaged_imbalance()
        if imbalance is None:
            self.last_skip_reason = "⏸ Could not fetch order book"
            return None

        # NOTE: imbalance_threshold pre-filter removed — imbalance_threshold is used
        # directly in the edge formula below. A weak imbalance → low/negative edge →
        # filtered naturally by edge_threshold. Safety: imbalance == 0 → no direction.
        if imbalance == 0:
            self.last_skip_reason = "⏸ Zero imbalance — no directional signal"
            return None

        side = "YES" if imbalance > 0 else "NO"

        # Edge: raw value (no cap — p_win cap handles risk via Kelly)
        # Note: edge may be negative when |imbalance| < threshold; Kelly will be ≤ 0.
        edge = abs(imbalance) - self.imbalance_threshold

        # p_win: capped at 0.60 (conservative for a noisy signal)
        p_win = min(0.5 + abs(imbalance) * 0.2, 0.60)

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
            f"OrderBook eval: imbalance={imbalance:+.3f} (weighted, averaged) | "
            f"side={side} | edge={edge:.3f} | p_win={p_win:.3f}"
        )

        signal = Signal(
            side=side,
            edge=edge,
            p_up=0.5 + imbalance * 0.2 if side == "YES" else 0.5 + imbalance * 0.2,
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
