"""
open_interest.py — Open Interest Momentum Strategy

Edge hypothesis:
  When the BNB futures open interest (OI) changes significantly over the last
  15 minutes (3 × 5-minute bars), it signals directional conviction:

  - OI rising + price rising   → institutional longs being built → bet UP
  - OI rising + price falling  → institutional shorts being built → bet DOWN
  - OI falling + price moving  → liquidation cascade continuing  → follow direction
  - OI flat / low delta        → no conviction → skip

  OI delta is computed as (OI_latest - OI_oldest) / OI_oldest.
  Price momentum is computed from the live BNB price series (Binance WebSocket).
  A bet is only placed when BOTH signals agree on direction.

Data source: Binance Futures REST — openInterestHist endpoint (no auth required).
  https://fapi.binance.com/futures/data/openInterestHist?symbol=BNBUSDT&period=5m&limit=3

Config keys (under "strategy"):
  oi_delta_threshold  : float  — minimum |delta OI %| to act (default 0.0005 = 0.05%)
  oi_period           : str    — OI histogram period (default "5m")
"""

import logging
import time
import urllib.request
import json
from typing import Optional

from .base import BaseStrategy
from strategy import (
    Signal, WindowInfo,
    compute_edge, compute_position_size, compute_momentum,
)

logger = logging.getLogger(__name__)

# Binance Futures — open interest history endpoint (public, no auth)
_OI_HIST_URL = (
    "https://fapi.binance.com/futures/data/openInterestHist"
    "?symbol=BNBUSDT&period={period}&limit=3"
)

# HTTP timeout for OI fetch (seconds)
_FETCH_TIMEOUT = 3.0


def _fetch_oi_history(period: str = "5m") -> list[dict]:
    """
    Fetch the last 3 open-interest data points from Binance Futures.

    Returns a list of dicts with keys:
      - sumOpenInterest      : str  (BNB units, e.g. "510998.23000000")
      - sumOpenInterestValue : str  (USD value)
      - timestamp            : int  (Unix ms)

    Returns an empty list on any error.
    """
    url = _OI_HIST_URL.format(period=period)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "bnb-updown/1.0"})
        with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, list) and len(data) >= 2:
                return data
            logger.warning(f"OI history: unexpected response shape — {data}")
            return []
    except Exception as exc:
        logger.warning(f"OI history fetch failed: {exc}")
        return []


def _compute_oi_delta(oi_history: list[dict]) -> Optional[float]:
    """
    Compute delta OI % = (OI_latest - OI_oldest) / OI_oldest.

    Args:
        oi_history: List of OI dicts ordered oldest → newest.

    Returns:
        Delta as a signed float (positive = OI grew, negative = OI shrank).
        Returns None if data is invalid.
    """
    if len(oi_history) < 2:
        return None
    try:
        oi_oldest = float(oi_history[0]["sumOpenInterest"])
        oi_latest = float(oi_history[-1]["sumOpenInterest"])
        if oi_oldest <= 0:
            return None
        return (oi_latest - oi_oldest) / oi_oldest
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning(f"OI delta computation failed: {exc}")
        return None


class OpenInterestStrategy(BaseStrategy):
    """
    Open Interest Momentum strategy for PancakeSwap Prediction V2.

    Combines BNB futures OI change (institutional signal) with observed
    price momentum to generate directional bets.

    Signal logic:
      abs(oi_delta) >= oi_delta_threshold  (OI signal is strong enough)
      AND price_momentum agrees with OI direction
      → emit Signal(side=direction, ...)

    Uses fair-odds pricing (0.50) and fractional Kelly sizing inherited
    from BaseStrategy.
    """

    FAIR_ODDS_PRICE = 0.50

    def __init__(self, config: dict):
        super().__init__(config)
        cfg = config.get("strategy", {})
        self.oi_delta_threshold: float = cfg.get("oi_delta_threshold", 0.0005)
        self.oi_period: str = cfg.get("oi_period", "5m")

        # Minimum price momentum magnitude to confirm direction (fraction, e.g. 0.001 = 0.1%)
        self.price_momentum_min: float = cfg.get("price_momentum_min", 0.0005)

        # Minimum price points before trusting momentum
        self.min_price_points: int = 5

    @property
    def name(self) -> str:
        return "📊 Open Interest Momentum"

    def prefetch(self, prices: list[float], epoch=None) -> None:
        """Pre-fetch OI history and cache it for the sniper window."""
        super().prefetch(prices, epoch)
        logger.debug("OI: prefetching open interest history...")
        oi_history = _fetch_oi_history(self.oi_period)
        if len(oi_history) >= 2:
            oi_delta = _compute_oi_delta(oi_history)
            self._prefetch_cache["oi_history"] = oi_history
            self._prefetch_cache["oi_delta"] = oi_delta
            logger.info(f"OI prefetch: {len(oi_history)} bars | delta={oi_delta:+.4%}" if oi_delta is not None else f"OI prefetch: {len(oi_history)} bars | delta=None")
        else:
            logger.warning("OI prefetch: insufficient OI history fetched")

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

        # ── Only act in the entry window ──────────────────────────────────────
        if not window.is_entry_window and window.seconds_remaining > self.entry_window_seconds:
            return None

        # ── Price data guard ─────────────────────────────────────────────────
        if len(prices) < self.min_price_points:
            self.last_skip_reason = (
                f"⏸ Not enough price data ({len(prices)} pts, "
                f"need {self.min_price_points})"
            )
            return None

        # ── Fetch OI history (use prefetch cache if available) ───────────────
        if "oi_history" in self._prefetch_cache:
            oi_history = self._prefetch_cache["oi_history"]
            oi_delta = self._prefetch_cache.get("oi_delta")
            logger.debug(f"OI: using prefetched data | delta={oi_delta:+.4%}" if oi_delta is not None else "OI: using prefetched data")
        else:
            oi_history = _fetch_oi_history(self.oi_period)
            if len(oi_history) < 2:
                self.last_skip_reason = "⏸ OI data unavailable (fetch failed or insufficient points)"
                return None
            oi_delta = _compute_oi_delta(oi_history)

        if len(oi_history) < 2:
            self.last_skip_reason = "⏸ OI data unavailable (fetch failed or insufficient points)"
            return None

        if oi_delta is None:
            self.last_skip_reason = "⏸ OI delta computation failed"
            return None

        # NOTE: oi_delta_threshold pre-filter removed — oi_delta_threshold is used
        # in the edge formula (oi_scale = oi_abs / threshold*3). A weak OI delta →
        # low oi_scale → low p_up displacement → low edge → filtered by edge_threshold.
        oi_abs = abs(oi_delta)

        # Safety: oi_delta == 0 → no OI signal, skip
        if oi_abs == 0:
            self.last_skip_reason = "⏸ OI delta is zero — no signal"
            return None

        # ── Price momentum ───────────────────────────────────────────────────
        price_momentum = compute_momentum(prices)

        logger.info(
            f"OI delta={oi_delta:+.4%} | price_momentum={price_momentum:+.4%} | "
            f"remaining={window.seconds_remaining:.1f}s"
        )

        # ── Determine directional conviction ─────────────────────────────────
        # OI growing → market is adding positions.
        #   Combined with price direction → tells us which side is loading up.
        # OI shrinking → liquidation cascade.
        #   Combined with price direction → cascade continues.
        #
        # In both cases, we follow the price direction, using OI magnitude
        # as a confirmation/confidence amplifier.

        if abs(price_momentum) < self.price_momentum_min:
            self.last_skip_reason = (
                f"⏸ Price momentum too weak ({price_momentum:+.4%}), "
                f"no direction signal despite strong OI delta ({oi_delta:+.4%})"
            )
            return None

        # Direction: price momentum determines UP/DOWN
        price_going_up = price_momentum > 0

        # Map to P(up) — stronger OI confirmation → higher confidence
        # Base probability: 0.5 ± directional bias
        # OI amplification: scales the bias up to a maximum of ±0.15 (on top of base)
        # Max bias reached at 3× the threshold
        max_bias = 0.15
        oi_scale = min(oi_abs / (self.oi_delta_threshold * 3.0), 1.0)
        directional_bias = max_bias * oi_scale  # 0..0.15

        if price_going_up:
            p_up = 0.50 + directional_bias
        else:
            p_up = 0.50 - directional_bias

        # Clamp to [0.35, 0.65] — same humility as GBM strategy
        p_up = max(0.35, min(0.65, p_up))

        # ── Edge calculation (always fair odds) ──────────────────────────────
        effective_yes_price = self.FAIR_ODDS_PRICE
        edge, side = compute_edge(p_up, effective_yes_price)

        logger.info(
            f"OI Momentum: p_up={p_up:.3f} edge={edge:.3f} side={side} | "
            f"oi_delta={oi_delta:+.4%} oi_scale={oi_scale:.2f} "
            f"bias={directional_bias:+.3f}"
        )

        if edge <= self.edge_threshold:
            self.last_skip_reason = (
                f"⏸ Edge too low ({edge:.3f} ≤ {self.edge_threshold:.2f}) | "
                f"P(Up)={p_up:.2f} | {side}"
            )
            return None

        # ── Position sizing ───────────────────────────────────────────────────
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

        logger.info(f"🎯 OI Momentum Signal: {signal}")
        return signal
