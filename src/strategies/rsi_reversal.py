"""
rsi_reversal.py — RSI Extreme Reversal Strategy

Hypothesis: Extreme RSI readings on short timeframes (1m) on BNB predict
mean-reversion moves over the following 5 minutes with ~55-60% win rate.

Logic:
- Compute RSI(5) and RSI(14) on 1m Binance klines (BNBUSDT)
- RSI(5) < 25 AND RSI(14) < 40  → extreme oversold  → bet UP
- RSI(5) > 75 AND RSI(14) > 60  → extreme overbought → bet DOWN
- Edge is proportional to how far RSI is beyond the threshold
- Volume spike confirmation boosts edge slightly

Klines are fetched directly from Binance REST API and cached for 45s to
avoid hammering the endpoint on every evaluate() tick.
"""

import logging
import time
import json
from typing import Optional
from urllib.request import urlopen
from urllib.error import URLError

import numpy as np

from .base import BaseStrategy
from strategy import (
    Signal, WindowInfo,
    compute_edge, compute_position_size,
)

logger = logging.getLogger(__name__)

# Binance REST endpoint for 1m klines
_BINANCE_KLINES_URL = (
    "https://api.binance.com/api/v3/klines"
    "?symbol=BNBUSDT&interval=1m&limit={limit}"
)

# Kline field indices (Binance format)
_K_CLOSE = 4    # close price
_K_VOLUME = 5   # base asset volume


class RSIReversalStrategy(BaseStrategy):
    """
    RSI Extreme Reversal Strategy for PancakeSwap Prediction V2.

    Trades mean-reversion signals when RSI on 1m BNB klines reaches extreme
    oversold or overbought territory.  Both a fast RSI and a slow RSI must
    confirm the signal to reduce false positives.
    """

    # Fair-odds reference (pool price ignored — we trade at 50/50)
    FAIR_ODDS_PRICE = 0.50

    # Max p_up deviation from 0.50 we'll assign
    # 0.49 → p_up in [0.01, 0.99], giving full edge range [0, 0.49]
    _MAX_P_DEVIATION = 0.49   # → p_up in [0.01, 0.99]

    # Volume spike: if current candle volume > N × median, boost edge
    _VOLUME_SPIKE_MULTIPLIER = 1.5
    _VOLUME_SPIKE_BOOST = 0.02   # +2% p_up boost when volume spike confirmed

    def __init__(self, config: dict):
        super().__init__(config)

        cfg = config.get("strategy", {})

        # RSI periods
        self.rsi_period_fast: int = cfg.get("rsi_period_fast", 5)
        self.rsi_period_slow: int = cfg.get("rsi_period_slow", 14)

        # Oversold thresholds (bet UP when both below)
        # Calibrated from observed RSI distributions on 1m BNB klines:
        #   RSI(5) p5=18, RSI(14) p5=33 — old thresholds (15/35) never triggered
        self.rsi_oversold_fast: float = cfg.get("rsi_oversold_fast", 25.0)
        self.rsi_oversold_slow: float = cfg.get("rsi_oversold_slow", 40.0)

        # Overbought thresholds (bet DOWN when both above)
        # RSI(5) p95=81, RSI(14) p95=70 — old thresholds (85/65) nearly impossible
        self.rsi_overbought_fast: float = cfg.get("rsi_overbought_fast", 75.0)
        self.rsi_overbought_slow: float = cfg.get("rsi_overbought_slow", 60.0)

        # How many 1m klines to fetch (need at least slow_period + a few buffer)
        self._kline_limit: int = max(40, self.rsi_period_slow * 3)

        # Internal kline cache — avoid spamming Binance on every tick
        self._kline_cache: Optional[list] = None
        self._kline_cache_ts: float = 0.0
        self._kline_cache_ttl: float = 45.0   # refresh every 45 seconds

    # ──────────────────────────────────────────────────────────────────────────
    # BaseStrategy interface
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "🔄 RSI Extreme Reversal"

    def prefetch(self, prices: list[float], epoch=None) -> None:
        """Pre-fetch Binance klines for RSI computation."""
        super().prefetch(prices, epoch)
        logger.debug("RSI: prefetch: fetching klines...")
        klines = self._fetch_klines()
        if klines is not None:
            self._prefetch_cache["klines"] = klines
            logger.info(f"RSI prefetch: {len(klines)} klines cached")
        else:
            logger.warning("RSI prefetch: failed to fetch klines")

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

        # Only act inside the entry window (last N seconds before lock)
        if not window.is_entry_window and window.seconds_remaining > self.entry_window_seconds:
            return None

        # ── Fetch 1m klines (use prefetch cache if available) ────────────────
        if "klines" in self._prefetch_cache:
            klines = self._prefetch_cache["klines"]
            logger.debug(f"RSI: using prefetched klines ({len(klines)})")
        else:
            klines = self._get_klines()
        if klines is None:
            self.last_skip_reason = "⏸ RSI: failed to fetch klines from Binance"
            logger.warning("RSI strategy: could not fetch klines — skipping")
            return None

        # Minimum data check: need slow_period + 1 deltas
        min_required = self.rsi_period_slow + 2
        if len(klines) < min_required:
            self.last_skip_reason = (
                f"⏸ RSI: not enough klines ({len(klines)} < {min_required})"
            )
            return None

        # ── Extract closes and volumes ────────────────────────────────────────
        closes = np.array([float(k[_K_CLOSE]) for k in klines], dtype=np.float64)
        volumes = np.array([float(k[_K_VOLUME]) for k in klines], dtype=np.float64)

        # ── Compute RSI values ────────────────────────────────────────────────
        rsi_fast = self._compute_rsi(closes, self.rsi_period_fast)
        rsi_slow = self._compute_rsi(closes, self.rsi_period_slow)

        logger.info(
            f"RSI eval: RSI({self.rsi_period_fast})={rsi_fast:.2f}  "
            f"RSI({self.rsi_period_slow})={rsi_slow:.2f}  "
            f"remaining={window.seconds_remaining:.1f}s"
        )

        # NOTE: RSI threshold pre-filters removed — the RSI thresholds are used
        # directly in the p_up / norm formulas below. A neutral RSI → norm ≈ 0 →
        # p_up ≈ 0.50 → near-zero edge → filtered by edge_threshold.
        # We must still determine direction (oversold → UP, overbought → DOWN).
        # Use the midpoint (50) as the dividing line when no threshold is breached.
        is_oversold = (
            rsi_fast < self.rsi_oversold_fast
            and rsi_slow < self.rsi_oversold_slow
        )
        is_overbought = (
            rsi_fast > self.rsi_overbought_fast
            and rsi_slow > self.rsi_overbought_slow
        )

        # ── Compute p_up from RSI extremity ───────────────────────────────────
        if is_oversold or rsi_fast < 50.0:
            # How far below the oversold threshold is the fast RSI?
            # Positive norm when below threshold, negative when above (→ small/negative p_up deviation)
            norm = (self.rsi_oversold_fast - rsi_fast) / max(self.rsi_oversold_fast, 1.0)
            norm = max(-1.0, min(1.0, norm))
            p_up = 0.5 + self._MAX_P_DEVIATION * norm
            direction = "oversold → UP" if is_oversold else "mild low → UP"
        else:
            # How far above the overbought threshold is the fast RSI?
            norm = (rsi_fast - self.rsi_overbought_fast) / max(100.0 - self.rsi_overbought_fast, 1.0)
            norm = max(-1.0, min(1.0, norm))
            p_up = 0.5 - self._MAX_P_DEVIATION * norm
            direction = "overbought → DOWN" if is_overbought else "mild high → DOWN"

        # ── Volume spike confirmation ─────────────────────────────────────────
        volume_boosted = False
        if len(volumes) >= 5:
            median_vol = float(np.median(volumes[:-1]))   # exclude current candle
            current_vol = volumes[-1]
            if median_vol > 0 and current_vol >= median_vol * self._VOLUME_SPIKE_MULTIPLIER:
                # Volume spike: add a small boost in the predicted direction
                if is_oversold:
                    p_up = min(0.99, p_up + self._VOLUME_SPIKE_BOOST)
                else:
                    p_up = max(0.01, p_up - self._VOLUME_SPIKE_BOOST)
                volume_boosted = True
                logger.info(
                    f"📊 Volume spike confirmed: {current_vol:.2f} vs "
                    f"median {median_vol:.2f} (×{current_vol/median_vol:.1f})"
                )

        # ── Edge vs fair odds ─────────────────────────────────────────────────
        edge, side = compute_edge(p_up, self.FAIR_ODDS_PRICE)

        logger.info(
            f"RSI signal: {direction} | "
            f"RSI_fast={rsi_fast:.2f} RSI_slow={rsi_slow:.2f} | "
            f"p_up={p_up:.3f} edge={edge:.3f} side={side} | "
            f"vol_boost={volume_boosted}"
        )

        if edge <= self.edge_threshold:
            self.last_skip_reason = (
                f"⏸ RSI edge too low ({edge:.3f} ≤ {self.edge_threshold:.2f}) | "
                f"{direction} | RSI_fast={rsi_fast:.1f}"
            )
            return None

        # ── Position sizing (Kelly) ────────────────────────────────────────────
        pool_total_usdc = pool_total_bnb * prices[-1] if prices else 0.0
        raw_k, pos_size = compute_position_size(
            edge=edge,
            p_up=p_up,
            side=side,
            yes_price=self.FAIR_ODDS_PRICE,
            bankroll=self.bankroll,
            position_size_usdc=self.position_size_usdc,
        
            min_usdc=self.min_position_usdc,
            max_bankroll_pct=self.max_bankroll_pct,
            max_pool_pct=self.max_pool_pct,
            pool_total_usdc=pool_total_usdc,
        )

        if pos_size <= 0:
            self.last_skip_reason = f"⏸ RSI: position size is 0 (below min)"
            return None

        signal = Signal(
            side=side,
            edge=edge,
            p_up=p_up,
            yes_price=self.FAIR_ODDS_PRICE,
            kelly_fraction=0.0,
            position_size_usdc=pos_size,
            timestamp=time.time(),
            is_mock=is_mock_data,
            bull_pct=pool_bull_bnb / pool_total_bnb if pool_total_bnb > 0 else 0.0,
            bear_pct=pool_bear_bnb / pool_total_bnb if pool_total_bnb > 0 else 0.0,
        )

        logger.info(f"🎯 RSI signal generated: {signal}")
        return signal

    # ──────────────────────────────────────────────────────────────────────────
    # RSI calculation (numpy-only, no external libraries)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_rsi(closes: np.ndarray, period: int) -> float:
        """
        Compute RSI for the most recent close using Wilder's smoothing method.

        Args:
            closes: 1-D array of close prices in chronological order.
            period: RSI lookback period (e.g. 5 or 14).

        Returns:
            RSI value in [0, 100].  Returns 50.0 if insufficient data.
        """
        if len(closes) < period + 1:
            logger.debug(
                f"RSI({period}): insufficient data "
                f"({len(closes)} closes, need {period + 1})"
            )
            return 50.0

        deltas = np.diff(closes)
        gains = np.where(deltas > 0.0, deltas, 0.0)
        losses = np.where(deltas < 0.0, -deltas, 0.0)

        # Seed with simple average over the first `period` deltas
        avg_gain: float = float(np.mean(gains[:period]))
        avg_loss: float = float(np.mean(losses[:period]))

        # Wilder's exponential smoothing for subsequent deltas
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0.0:
            # No losses at all → either perfectly flat (neutral) or pure uptrend
            return 50.0 if avg_gain == 0.0 else 100.0

        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    # ──────────────────────────────────────────────────────────────────────────
    # Binance kline fetching with cache
    # ──────────────────────────────────────────────────────────────────────────

    def _get_klines(self) -> Optional[list]:
        """
        Return 1m BNBUSDT klines, using an in-memory cache to avoid hammering
        Binance on every evaluate() call.

        Returns:
            List of kline arrays, or None on fetch failure.
        """
        now = time.time()
        if (
            self._kline_cache is not None
            and (now - self._kline_cache_ts) < self._kline_cache_ttl
        ):
            return self._kline_cache

        klines = self._fetch_klines()
        if klines is not None:
            self._kline_cache = klines
            self._kline_cache_ts = now
        return klines

    def _fetch_klines(self) -> Optional[list]:
        """
        Fetch 1m BNBUSDT klines from Binance REST API.

        Returns:
            Parsed list of kline arrays, or None on error.
        """
        url = _BINANCE_KLINES_URL.format(limit=self._kline_limit)
        try:
            with urlopen(url, timeout=5) as resp:
                raw = resp.read().decode("utf-8")
            klines = json.loads(raw)
            logger.debug(f"Fetched {len(klines)} klines from Binance")
            return klines
        except URLError as exc:
            logger.error(f"RSI: Binance kline fetch failed (network): {exc}")
            return None
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error(f"RSI: Binance kline parse error: {exc}")
            return None
        except Exception as exc:
            logger.error(f"RSI: unexpected error fetching klines: {exc}")
            return None
