"""
volume_breakout.py — Volume Profile Breakout Strategy

Hypothesis: High-volume price nodes (HVN) act as strong support/resistance.
When price breaks above/below the Point of Control (POC) with volume confirmation,
the move tends to continue in the breakout direction.

Volume Profile (VPVR) method:
- Fetch 30x 1m klines from Binance
- Distribute each candle's volume proportionally across its high-low price range
- POC = price bucket with the highest cumulative volume
- If price > POC + buffer AND last candle volume is elevated → bet UP
- If price < POC - buffer AND last candle volume is elevated → bet DOWN
- Edge = normalized distance from POC relative to recent volatility
"""

import logging
import time
from typing import Optional

import numpy as np
import requests

from .base import BaseStrategy
from strategy import Signal, WindowInfo, compute_edge, compute_position_size

logger = logging.getLogger(__name__)

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


class VolumeBreakoutStrategy(BaseStrategy):
    """
    Volume Profile Breakout Strategy for PancakeSwap Prediction V2.

    Builds a real-time Volume Profile from Binance 1m klines to identify
    the Point of Control (POC). Signals a trade when price breaks through
    the POC zone with volume confirmation (spike above average).

    Config keys (under strategy.{}):
        volume_breakout_buffer_pct (float): % buffer above/below POC to define
            breakout zone. Default: 0.001 (0.1%).
        volume_breakout_confirm_multiplier (float): Last candle volume must be
            this many times the average. Default: 1.5 (1.5x avg).
        volume_breakout_klines (int): Number of 1m klines to fetch. Default: 30.
        volume_breakout_buckets (int): Number of price buckets for the profile.
            Default: 50.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        cfg = config.get("strategy", {})
        self.buffer_pct = cfg.get("volume_breakout_buffer_pct", 0.001)
        self.confirm_multiplier = cfg.get("volume_breakout_confirm_multiplier", 1.5)
        self.n_klines = cfg.get("volume_breakout_klines", 30)
        self.n_buckets = cfg.get("volume_breakout_buckets", 50)

    @property
    def name(self) -> str:
        return "📈 Volume Profile Breakout"

    # ─────────────────────────────────────────────────────────────────────────
    # Prefetch (sniper pre-load phase)
    # ─────────────────────────────────────────────────────────────────────────

    def prefetch(self, prices: list[float], epoch=None) -> None:
        """Pre-fetch Binance klines for volume profile analysis."""
        super().prefetch(prices, epoch)
        logger.debug("[VolumeBreakout] prefetch: fetching klines...")
        klines = self._fetch_klines()
        if len(klines) >= 5:
            self._prefetch_cache["klines"] = klines
            logger.info(f"[VolumeBreakout] prefetch: {len(klines)} klines cached")
        else:
            logger.warning(f"[VolumeBreakout] prefetch: insufficient klines ({len(klines)})")

    # ─────────────────────────────────────────────────────────────────────────
    # Data fetching
    # ─────────────────────────────────────────────────────────────────────────

    def _fetch_klines(self, symbol: str = "BNBUSDT") -> list[dict]:
        """
        Fetch 1m klines from Binance REST API.

        Returns a list of dicts with keys: open, high, low, close, volume.
        Returns empty list on error.
        """
        try:
            resp = requests.get(
                BINANCE_KLINES_URL,
                params={"symbol": symbol, "interval": "1m", "limit": self.n_klines},
                timeout=5.0,
            )
            resp.raise_for_status()
            raw = resp.json()
            # Kline format: [open_time, open, high, low, close, volume, ...]
            return [
                {
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                }
                for k in raw
                if isinstance(k, (list, tuple)) and len(k) >= 6
            ]
        except requests.RequestException as e:
            logger.warning(f"[VolumeBreakout] Binance klines fetch failed: {e}")
            return []
        except Exception as e:
            logger.warning(f"[VolumeBreakout] Unexpected error fetching klines: {e}")
            return []

    # ─────────────────────────────────────────────────────────────────────────
    # Volume Profile construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_volume_profile(
        self, klines: list[dict]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build a simplified Volume Profile from klines.

        For each candle, distributes its volume proportionally across the
        price buckets that fall within its [low, high] range. This mirrors
        the classic VPVR methodology used in professional charting.

        Returns:
            (bucket_prices, bucket_volumes) — center price and cumulative
            volume for each bucket. Both arrays of length n_buckets.
            Returns (empty, empty) on failure.
        """
        if not klines:
            return np.array([]), np.array([])

        price_min = min(k["low"] for k in klines)
        price_max = max(k["high"] for k in klines)

        if price_max <= price_min:
            return np.array([]), np.array([])

        n = self.n_buckets
        bucket_size = (price_max - price_min) / n
        volumes = np.zeros(n)

        for k in klines:
            candle_range = k["high"] - k["low"]
            if candle_range <= 0:
                # Doji candle: assign all volume to the single close bucket
                idx = int((k["close"] - price_min) / bucket_size)
                idx = max(0, min(idx, n - 1))
                volumes[idx] += k["volume"]
                continue

            # Map candle range to bucket indices
            low_idx = int((k["low"] - price_min) / bucket_size)
            high_idx = int((k["high"] - price_min) / bucket_size)
            low_idx = max(0, min(low_idx, n - 1))
            high_idx = max(0, min(high_idx, n - 1))

            span = high_idx - low_idx + 1
            vol_per_bucket = k["volume"] / span
            for b in range(low_idx, high_idx + 1):
                volumes[b] += vol_per_bucket

        # Bucket center prices
        bucket_prices = np.array(
            [price_min + (i + 0.5) * bucket_size for i in range(n)]
        )
        return bucket_prices, volumes

    def _compute_poc(
        self, bucket_prices: np.ndarray, bucket_volumes: np.ndarray
    ) -> float:
        """
        Point of Control: center price of the bucket with maximum volume.

        In Volume Profile theory, the POC represents the price level where
        the most trading activity occurred — a strong magnetic level.
        """
        if len(bucket_volumes) == 0:
            return 0.0
        poc_idx = int(np.argmax(bucket_volumes))
        return float(bucket_prices[poc_idx])

    # ─────────────────────────────────────────────────────────────────────────
    # Indicators
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_recent_volatility(self, klines: list[dict]) -> float:
        """
        Realized volatility from 1m close prices (std dev of log returns).

        Used to normalize the edge: a breakout that covers 2× the typical
        per-candle range is a stronger signal than one that covers 0.5×.

        Returns a small positive floor (0.0001) to avoid division by zero.
        """
        closes = [k["close"] for k in klines]
        if len(closes) < 3:
            return 0.0001
        log_returns = np.diff(np.log(closes))
        vol = float(np.std(log_returns))
        return max(vol, 1e-6)

    # ─────────────────────────────────────────────────────────────────────────
    # Strategy core
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
        """
        Evaluate a Volume Profile Breakout signal.

        Flow:
        1. Only evaluate within the entry window (last N seconds before lock).
        2. Fetch Binance 1m klines and build a Volume Profile.
        3. Identify the POC (Point of Control).
        4. Check price is outside the POC buffer zone.
        5. Confirm with a volume spike on the last candle.
        6. Compute edge = distance_to_POC / (poc × volatility × scale).
        7. Size position via Kelly criterion.

        Args:
            prices: Real-time BNB price series (tick data from WebSocket).
            yes_price: PancakeSwap pool YES price (ignored — uses fair odds 0.5).
            window: Current round/window info (seconds to lock, etc.).
            is_mock_data: True when running in simulation mode.
            pool_total_bnb, pool_bull_bnb, pool_bear_bnb: Pool composition.

        Returns:
            Signal if conditions are met, None otherwise.
            Sets self.last_skip_reason when returning None.
        """
        self.last_skip_reason = None

        # Gate: only act in entry window
        if not window.is_entry_window and window.seconds_remaining > self.entry_window_seconds:
            return None

        if len(prices) < 2:
            self.last_skip_reason = "⏸ Not enough real-time price data"
            return None

        current_price = prices[-1]

        # ── Fetch klines (use prefetch cache if available) ────────────────────
        if "klines" in self._prefetch_cache:
            klines = self._prefetch_cache["klines"]
            logger.debug(f"[VolumeBreakout] using prefetched klines ({len(klines)})")
        else:
            klines = self._fetch_klines()
        if len(klines) < 5:
            self.last_skip_reason = (
                f"⏸ Insufficient klines ({len(klines)} / {self.n_klines})"
            )
            logger.warning(f"[VolumeBreakout] Not enough klines: {len(klines)}")
            return None

        # ── Volume Profile ────────────────────────────────────────────────────
        bucket_prices, bucket_volumes = self._build_volume_profile(klines)
        if len(bucket_prices) == 0:
            self.last_skip_reason = "⏸ Volume profile construction failed"
            return None

        poc = self._compute_poc(bucket_prices, bucket_volumes)
        if poc <= 0:
            self.last_skip_reason = "⏸ Invalid POC computed"
            return None

        # ── Volume confirmation ───────────────────────────────────────────────
        all_volumes = [k["volume"] for k in klines]
        # Compare last candle against average of prior candles
        prior_avg = float(np.mean(all_volumes[:-1])) if len(all_volumes) > 1 else all_volumes[0]
        last_volume = all_volumes[-1]
        required_volume = prior_avg * self.confirm_multiplier

        if last_volume < required_volume:
            self.last_skip_reason = (
                f"⏸ Volume not confirmed "
                f"(last={last_volume:.2f} < {required_volume:.2f} = "
                f"{self.confirm_multiplier}× avg {prior_avg:.2f})"
            )
            logger.debug(
                f"[VolumeBreakout] Volume spike absent: "
                f"last={last_volume:.2f} avg={prior_avg:.2f} "
                f"threshold={required_volume:.2f}"
            )
            return None

        # ── Breakout detection ────────────────────────────────────────────────
        buffer = poc * self.buffer_pct
        distance_to_poc = current_price - poc  # signed
        abs_distance = abs(distance_to_poc)

        volatility = self._compute_recent_volatility(klines)

        # Edge = normalized distance from POC (how many volatility units away).
        # Scale factor 10 is empirical: vol of 1m returns is ~0.001–0.003,
        # so poc*vol*10 ≈ 0.01–0.03 price, roughly one typical 1m candle range.
        edge_raw = abs_distance / (poc * volatility * 10.0)
        edge_raw = max(0.0, min(edge_raw, 0.99))

        logger.info(
            f"[VolumeBreakout] POC={poc:.4f} | current={current_price:.4f} | "
            f"distance={distance_to_poc:+.4f} | buffer={buffer:.4f} | "
            f"vol_confirmed={last_volume:.2f}/{required_volume:.2f} | "
            f"edge_raw={edge_raw:.4f} | volatility={volatility:.6f}"
        )

        # NOTE: POC buffer pre-filter removed — distance_to_poc (which includes the
        # buffer zone) is used directly in edge_raw. A price inside the buffer →
        # small abs_distance → low edge_raw → low edge → filtered by edge_threshold.
        # Safety: price == poc → zero distance → edge_raw = 0 → no signal.
        if distance_to_poc == 0:
            self.last_skip_reason = "⏸ Price exactly at POC — no directional signal"
            return None

        # Direction: sign of distance from POC
        if current_price > poc:
            # Price above POC → momentum UP
            # edge_raw ∈ [0, 0.99] → p_up ∈ [0.50, 0.99], edge ∈ [0, 0.49]
            p_up = min(0.99, 0.5 + edge_raw * 0.5)
            direction = "UP"
        else:
            # Price below POC → momentum DOWN
            p_up = max(0.01, 0.5 - edge_raw * 0.5)
            direction = "DOWN"

        # ── Edge vs threshold ─────────────────────────────────────────────────
        # Use fair odds (0.5) — same as GBMStrategy fair_odds mode
        effective_yes_price = 0.5
        computed_edge, side = compute_edge(p_up, effective_yes_price)

        logger.info(
            f"[VolumeBreakout] direction={direction} | p_up={p_up:.3f} | "
            f"edge={computed_edge:.3f} | side={side} | "
            f"remaining={window.seconds_remaining:.1f}s"
        )

        if computed_edge <= self.edge_threshold:
            self.last_skip_reason = (
                f"⏸ Edge too low ({computed_edge:.3f} ≤ {self.edge_threshold:.2f}) | "
                f"POC={poc:.4f} | {direction}"
            )
            return None

        # ── Position sizing ───────────────────────────────────────────────────
        pool_total_usdc = pool_total_bnb * prices[-1] if prices else 0.0
        raw_k, pos_size = compute_position_size(
            edge=computed_edge,
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
            self.last_skip_reason = "⏸ Position size is zero"
            return None

        signal = Signal(
            side=side,
            edge=computed_edge,
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
            f"🎯 [VolumeBreakout] Signal: {signal} | "
            f"POC={poc:.4f} | breakout={direction}"
        )
        return signal
