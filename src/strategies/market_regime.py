"""
market_regime.py — Market Regime Adaptive Strategy

Detects the current market regime (trending vs mean-reverting) using the Hurst
Exponent (R/S analysis) on recent BNB prices, then adapts direction accordingly:

  H > 0.6  → Trending regime  → follow momentum
  H < 0.4  → Ranging regime   → fade momentum (mean-reversion)
  0.4–0.6  → Ambiguous        → skip

ATR (approximated from tick data) is used as a secondary confirmation:
expanding volatility supports trending; contracting volatility supports ranging.

Edge = abs(H - 0.5) * 2  (distance from random walk, normalised to [0, 1])
"""

import logging
import math
import time
from typing import Optional

import numpy as np

from .base import BaseStrategy
from strategy import (
    Signal, WindowInfo,
    compute_edge, compute_position_size,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Pure maths helpers (unit-testable)
# ─────────────────────────────────────────────────────────────────────────────

def compute_hurst_rs(prices: list[float]) -> Optional[float]:
    """
    Estimate the Hurst Exponent via simplified R/S analysis.

    Works reliably on as few as 20–30 data points.  We use multiple sub-series
    lengths and fit log(R/S) ~ H * log(n) via least squares.

    H ≈ 0.5  → random walk (no memory)
    H > 0.5  → persistent / trending
    H < 0.5  → anti-persistent / mean-reverting

    Args:
        prices: Price series (≥ 4 points required).

    Returns:
        Hurst exponent in [0, 1], or None if insufficient data.
    """
    if len(prices) < 4:
        return None

    log_returns = np.diff(np.log(prices))
    n_total = len(log_returns)

    if n_total < 3:
        return None

    # Build a set of sub-series lengths evenly spaced between 4 and n_total
    min_len = max(4, n_total // 4)
    lengths = list(set(
        int(round(x))
        for x in np.linspace(min_len, n_total, num=min(8, n_total - min_len + 1))
    ))
    lengths = sorted(l for l in lengths if 4 <= l <= n_total)

    if len(lengths) < 2:
        # Fallback: single R/S estimate
        return _rs_single(log_returns)

    log_n_vals: list[float] = []
    log_rs_vals: list[float] = []

    for length in lengths:
        # Use all non-overlapping sub-series of this length and average R/S
        rs_values = []
        for start in range(0, n_total - length + 1, length):
            sub = log_returns[start: start + length]
            rs = _rs_for_subseries(sub)
            if rs is not None:
                rs_values.append(rs)
        if rs_values:
            mean_rs = float(np.mean(rs_values))
            if mean_rs > 0:
                log_n_vals.append(math.log(length))
                log_rs_vals.append(math.log(mean_rs))

    if len(log_n_vals) < 2:
        return _rs_single(log_returns)

    # Ordinary least squares: H = slope of log(R/S) vs log(n)
    x = np.array(log_n_vals)
    y = np.array(log_rs_vals)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = float(np.sum((x - x_mean) ** 2))
    if denom == 0:
        return _rs_single(log_returns)
    H = float(np.sum((x - x_mean) * (y - y_mean)) / denom)

    return max(0.0, min(1.0, H))


def _rs_for_subseries(returns: np.ndarray) -> Optional[float]:
    """R/S statistic for a single sub-series of log returns."""
    n = len(returns)
    if n < 2:
        return None
    mean_r = returns.mean()
    deviations = returns - mean_r
    cumdev = np.cumsum(deviations)
    R = float(cumdev.max() - cumdev.min())
    S = float(returns.std(ddof=1))
    if S <= 0:
        return None
    return R / S


def _rs_single(log_returns: np.ndarray) -> Optional[float]:
    """Fallback single R/S estimate → H ≈ log(R/S) / log(n)."""
    rs = _rs_for_subseries(log_returns)
    if rs is None or rs <= 0:
        return None
    n = len(log_returns)
    if n < 2:
        return None
    H = math.log(rs) / math.log(n)
    return max(0.0, min(1.0, H))


def compute_atr_pct(prices: list[float], period: int = 10) -> tuple[float, float]:
    """
    Approximate ATR as mean absolute tick-to-tick changes (as % of price).

    Since we have tick data (not OHLC), True Range ≈ |price[i] - price[i-1]|.

    Args:
        prices: Tick price series.
        period: Window for 'recent' ATR (short-term).

    Returns:
        (recent_atr_pct, long_atr_pct) — both normalised by current price.
        Returns (0.0, 0.0) if insufficient data.
    """
    if len(prices) < 3:
        return 0.0, 0.0

    abs_changes = np.abs(np.diff(prices))
    current_price = prices[-1]

    long_atr = float(np.mean(abs_changes))
    recent_changes = abs_changes[-min(period, len(abs_changes)):]
    recent_atr = float(np.mean(recent_changes))

    if current_price <= 0:
        return 0.0, 0.0

    return recent_atr / current_price, long_atr / current_price


def atr_regime_multiplier(
    H: float,
    recent_atr_pct: float,
    long_atr_pct: float,
    hurst_trending: float = 0.6,
    hurst_ranging: float = 0.4,
) -> float:
    """
    ATR confirmation factor in [0.5, 1.0].

    - Trending regime (H > hurst_trending): high recent ATR confirms trending.
    - Ranging regime  (H < hurst_ranging):  low  recent ATR confirms ranging.

    Uses ratio recent_atr / long_atr as a self-normalising measure so no
    hardcoded price-level thresholds are needed.

    Returns:
        Multiplier to apply to hurst_edge (1.0 = fully confirmed, 0.5 = weak).
    """
    if long_atr_pct <= 0:
        return 1.0

    ratio = recent_atr_pct / long_atr_pct  # >1 = expanding vol, <1 = contracting

    if H > hurst_trending:
        # Trending: expanding ATR is good (ratio > 1).  Contracting ATR is a warning.
        if ratio >= 1.2:
            return 1.0       # Strong confirmation
        elif ratio >= 0.8:
            return 0.8       # Moderate confirmation
        else:
            return 0.5       # Weak — ATR says ranging, Hurst says trending

    elif H < hurst_ranging:
        # Ranging: contracting ATR is good (ratio < 1).  Expanding ATR is a warning.
        inv_ratio = 1.0 / max(ratio, 1e-9)
        if inv_ratio >= 1.2:
            return 1.0
        elif inv_ratio >= 0.8:
            return 0.8
        else:
            return 0.5

    return 1.0  # Ambiguous regime — no adjustment needed (caller will skip)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy class
# ─────────────────────────────────────────────────────────────────────────────

class MarketRegimeStrategy(BaseStrategy):
    """
    Market Regime Adaptive Strategy for BNB Up/Down 5mn.

    Uses the Hurst Exponent to identify whether BNB is trending or
    mean-reverting, then aligns (trending) or fades (ranging) momentum.
    ATR expansion/contraction provides secondary confirmation.

    Config keys (under 'strategy'):
        regime_hurst_trending   (float, default 0.60): H above this → trending
        regime_hurst_ranging    (float, default 0.40): H below this → ranging
        regime_min_prices       (int,   default 20):   min price points needed
    """

    FAIR_ODDS_PRICE = 0.50
    # Maximum p_up displacement from 0.5 allowed (keeps humility like GBM)
    MAX_P_DISPLACEMENT = 0.15

    def __init__(self, config: dict):
        super().__init__(config)
        cfg = config.get("strategy", {})
        self.hurst_trending: float = cfg.get("regime_hurst_trending", 0.60)
        self.hurst_ranging: float  = cfg.get("regime_hurst_ranging",  0.40)
        self.min_prices: int       = cfg.get("regime_min_prices",     20)

    @property
    def name(self) -> str:
        return "🎯 Market Regime Adaptive"

    # ─────────────────────────────────────────────────────────────────────────
    # Core evaluate()
    # ─────────────────────────────────────────────────────────────────────────

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

        # Only act in the entry window
        if not window.is_entry_window and window.seconds_remaining > self.entry_window_seconds:
            return None

        # ── 1. Data sufficiency check ──────────────────────────────────────
        if len(prices) < self.min_prices:
            self.last_skip_reason = (
                f"⏸ Not enough prices ({len(prices)} < {self.min_prices} required)"
            )
            logger.info(self.last_skip_reason)
            return None

        # ── 2. Hurst Exponent via R/S analysis ────────────────────────────
        H = compute_hurst_rs(prices)
        if H is None:
            self.last_skip_reason = "⏸ Hurst computation failed (degenerate series)"
            logger.info(self.last_skip_reason)
            return None

        # ── 3. Regime classification ───────────────────────────────────────
        # NOTE: ambiguous zone pre-filter removed — Hurst exponent is used directly
        # in edge formula (hurst_edge_raw = abs(H - 0.5) * 2). H near 0.5 →
        # low edge → filtered naturally by edge_threshold.
        is_trending = H > self.hurst_trending   # True  → trend-following
        is_ranging  = H < self.hurst_ranging    # True  → mean-reversion
        # When H is between ranging and trending thresholds, we still use the
        # closest regime interpretation based on which side of 0.5 H falls on.

        # ── 4. Momentum direction ─────────────────────────────────────────
        if prices[-1] <= 0 or prices[0] <= 0:
            self.last_skip_reason = "⏸ Invalid prices (zero or negative)"
            return None

        momentum = (prices[-1] / prices[0]) - 1.0

        if momentum == 0.0:
            self.last_skip_reason = "⏸ Zero momentum — no directional signal"
            return None

        # ── 5. Direction: follow or fade ──────────────────────────────────
        if is_trending:
            direction_up = momentum > 0     # Follow trend
        else:  # ranging
            direction_up = momentum < 0     # Fade extreme

        # ── 6. Hurst edge (distance from random walk) ─────────────────────
        hurst_edge_raw = abs(H - 0.5) * 2.0   # [0, 1]; 0 at H=0.5, 1 at H=0 or H=1

        # ── 7. ATR confirmation ───────────────────────────────────────────
        recent_atr_pct, long_atr_pct = compute_atr_pct(prices)
        atr_mult = atr_regime_multiplier(
            H=H,
            recent_atr_pct=recent_atr_pct,
            long_atr_pct=long_atr_pct,
            hurst_trending=self.hurst_trending,
            hurst_ranging=self.hurst_ranging,
        )
        hurst_edge = hurst_edge_raw * atr_mult

        # ── 8. Map edge → p_up ────────────────────────────────────────────
        # Scale hurst_edge into a probability displacement.
        # hurst_edge ∈ [0, 1] → displacement ∈ [0, MAX_P_DISPLACEMENT]
        displacement = hurst_edge * self.MAX_P_DISPLACEMENT
        p_up_raw = (0.5 + displacement) if direction_up else (0.5 - displacement)
        p_up = max(0.35, min(0.65, p_up_raw))

        # ── 9. Edge vs fair odds ──────────────────────────────────────────
        effective_yes_price = self.FAIR_ODDS_PRICE
        edge, side = compute_edge(p_up, effective_yes_price)

        logger.info(
            f"[MarketRegime] H={H:.3f} | regime={'TREND' if is_trending else 'RANGE'} | "
            f"momentum={momentum:+.4f} | dir={'UP' if direction_up else 'DOWN'} | "
            f"hurst_edge_raw={hurst_edge_raw:.3f} | atr_mult={atr_mult:.2f} | "
            f"hurst_edge={hurst_edge:.3f} | p_up={p_up:.3f} | edge={edge:.3f} | side={side}"
        )

        # ── 10. Edge threshold filter ─────────────────────────────────────
        if edge <= self.edge_threshold:
            self.last_skip_reason = (
                f"⏸ Edge too low ({edge:.3f} ≤ {self.edge_threshold:.2f}) | "
                f"H={H:.3f} p_up={p_up:.3f} {side}"
            )
            return None

        # ── 11. Position sizing (fractional Kelly) ────────────────────────
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
            self.last_skip_reason = "⏸ Position size is zero (below min or cap)"
            return None

        # ── 12. Emit signal ───────────────────────────────────────────────
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

        logger.info(
            f"🎯 [MarketRegime] Signal: {signal} | "
            f"regime={'TREND' if is_trending else 'RANGE'} H={H:.3f}"
        )
        return signal
