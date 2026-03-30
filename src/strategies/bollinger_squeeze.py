"""
bollinger_squeeze.py — Bollinger Band Squeeze Strategy

Detects volatility compressions (BB squeeze) and trades the breakout direction.

Academic basis:
  - Bollinger Band Width (Bollinger 2001)
  - TTM Squeeze (John Carter, "Mastering the Trade")
  - Keltner Channel Squeeze with documented 55-65% win rate on confirmed breakouts

Signal logic:
  1. Fetch 1m klines from Binance (bb_klines=25 candles)
  2. Compute BB(bb_period=20, bb_std=2.0) on close prices (numpy only)
  3. BB width = (upper - lower) / middle  (normalized band width)
  4. Squeeze = current BB width < percentile(bb_squeeze_percentile=20) of recent widths
  5. Breakout UP  = squeeze AND last close > upper band
  6. Breakout DOWN = squeeze AND last close < lower band
  7. Confirm with 3-candle momentum (closes trending in breakout direction)
  8. Edge score = breakout_distance / band_range  →  mapped to p_up ∈ [0.40, 0.65]

Config keys (under "strategy" section):
  bb_period              : int   = 20    BB lookback (SMA + std window)
  bb_std                 : float = 2.0   Number of std devs for bands
  bb_squeeze_percentile  : int   = 20    Percentile threshold to declare squeeze
  bb_klines              : int   = 25    Number of 1m candles to fetch
  symbol                 : str   = "BNBUSDT"
  use_fair_odds          : bool  = True  Trade vs 0.50 (avoids pool imbalance bias)
"""

import json
import logging
import time
import urllib.request
from typing import Optional

import numpy as np

from .base import BaseStrategy
from strategy import Signal, WindowInfo, compute_edge, compute_position_size

logger = logging.getLogger(__name__)

_BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


class BollingerSqueezeStrategy(BaseStrategy):
    """
    Bollinger Band Squeeze Strategy for PancakeSwap Prediction V2.

    Trades breakouts from BB squeezes (volatility compression events).
    Uses only numpy (no ta-lib, no pandas).
    """

    def __init__(self, config: dict):
        super().__init__(config)
        cfg = config.get("strategy", {})
        self.bb_period: int = int(cfg.get("bb_period", 20))
        self.bb_std: float = float(cfg.get("bb_std", 2.0))
        self.bb_squeeze_percentile: int = int(cfg.get("bb_squeeze_percentile", 20))
        self.bb_klines: int = int(cfg.get("bb_klines", 25))
        self.symbol: str = cfg.get("symbol", "BNBUSDT")
        self.use_fair_odds: bool = bool(cfg.get("use_fair_odds", True))

        # TTL cache — refresh klines at most once per 30s to avoid Binance rate limits
        self._klines_cache: Optional[list] = None
        self._klines_cache_ts: float = 0.0
        self._klines_cache_ttl: float = 30.0

    # ──────────────────────────────────────────────────────────────────────
    # Interface
    # ──────────────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "🎯 Bollinger Band Squeeze"

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

        # Only act in the entry window (last N seconds before lock)
        if not window.is_entry_window and window.seconds_remaining > self.entry_window_seconds:
            return None

        # ── 1. Fetch klines ──────────────────────────────────────────────
        klines = self._fetch_klines()
        if not klines:
            self.last_skip_reason = "⏸ BB: Could not fetch Binance klines"
            return None

        # ── 2. Analyse squeeze / breakout ────────────────────────────────
        analysis = self._analyse_squeeze(klines)
        if analysis is None:
            self.last_skip_reason = f"⏸ BB: Insufficient kline data (need {self.bb_period})"
            return None

        if not analysis["squeeze"]:
            self.last_skip_reason = (
                f"⏸ BB: No squeeze (width={analysis['bb_width']:.6f} "
                f"≥ threshold={analysis['squeeze_threshold']:.6f})"
            )
            return None

        direction = analysis["direction"]
        if direction is None:
            self.last_skip_reason = "⏸ BB: Squeeze but no band breakout yet"
            return None

        # ── 3. Map breakout strength → p_up ─────────────────────────────
        edge_score = analysis["edge_score"]          # breakout_dist / band_range
        momentum_ok = analysis["momentum_confirms"]

        # base_confidence: how much above/below 0.50 we set p_up
        #   edge_score=0.20 → +0.10  (min to beat default 0.10 threshold with momentum)
        #   edge_score=0.30 → +0.15  (max)
        base_confidence = min(0.15, edge_score * 0.50)
        if not momentum_ok:
            base_confidence *= 0.60   # discount when momentum doesn't confirm

        p_up = (0.50 + base_confidence) if direction == "UP" else (0.50 - base_confidence)
        p_up = float(np.clip(p_up, 0.40, 0.65))

        # ── 4. Fair-odds / pool filters ──────────────────────────────────
        if self.use_fair_odds:
            effective_yes_price = 0.50
        else:
            if pool_total_bnb < self.min_pool_bnb:
                self.last_skip_reason = (
                    f"⏸ BB: Pool too small "
                    f"({pool_total_bnb:.2f} BNB < {self.min_pool_bnb:.1f})"
                )
                return None
            effective_yes_price = yes_price

        # ── 5. Edge check ────────────────────────────────────────────────
        edge, side = compute_edge(p_up, effective_yes_price)

        logger.info(
            f"BB Squeeze eval: direction={direction} "
            f"edge_score={edge_score:.4f} momentum={'✓' if momentum_ok else '✗'} "
            f"P(Up)={p_up:.3f} edge={edge:.3f} side={side} "
            f"remaining={window.seconds_remaining:.1f}s"
        )

        if edge <= self.edge_threshold:
            self.last_skip_reason = (
                f"⏸ BB: Edge too low ({edge:.3f} ≤ {self.edge_threshold:.2f}) | "
                f"{direction} breakout | momentum={'✓' if momentum_ok else '✗'} | "
                f"score={edge_score:.3f}"
            )
            return None

        # ── 6. Position sizing ───────────────────────────────────────────
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
            self.last_skip_reason = "⏸ BB: Position size computed to 0"
            return None

        # Optional pool liquidity check when not using fair odds
        if not self.use_fair_odds and prices:
            opposite_bnb = pool_bear_bnb if side == "YES" else pool_bull_bnb
            if opposite_bnb < 0.01:
                self.last_skip_reason = (
                    f"⏸ BB: No counterparty "
                    f"({'BEAR' if side == 'YES' else 'BULL'} side empty)"
                )
                return None
            bet_bnb = pos_size / prices[-1]
            side_bnb = pool_bull_bnb if side == "YES" else pool_bear_bnb
            if side_bnb > 0 and bet_bnb / side_bnb > self.max_bet_share_of_side:
                self.last_skip_reason = (
                    f"⏸ BB: Bet too large "
                    f"({bet_bnb:.3f} BNB = {bet_bnb / side_bnb:.0%} of {side} side)"
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

        logger.info(
            f"🎯 BB Squeeze Signal: {signal} | "
            f"direction={direction} momentum={'✓' if momentum_ok else '✗'} "
            f"bb_width={analysis['bb_width']:.6f} "
            f"edge_score={edge_score:.4f}"
        )
        return signal

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _fetch_klines(self) -> Optional[list]:
        """
        Fetch 1m klines from Binance REST API with a simple TTL cache.

        Returns a list of dicts: [{open, high, low, close}, ...] or None on failure.
        Falls back to stale cache on network error.
        """
        now = time.time()
        if (
            self._klines_cache is not None
            and (now - self._klines_cache_ts) < self._klines_cache_ttl
        ):
            return self._klines_cache

        url = (
            f"{_BINANCE_KLINES_URL}"
            f"?symbol={self.symbol}&interval=1m&limit={self.bb_klines}"
        )
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "bnb-updown-bb-squeeze/1.0"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                raw = json.loads(resp.read())

            # Binance kline format:
            # [open_time, open, high, low, close, volume, close_time, ...]
            klines = [
                {
                    "open":  float(k[1]),
                    "high":  float(k[2]),
                    "low":   float(k[3]),
                    "close": float(k[4]),
                }
                for k in raw
            ]
            self._klines_cache = klines
            self._klines_cache_ts = now
            logger.debug(f"BB: fetched {len(klines)} klines from Binance")
            return klines

        except Exception as exc:
            logger.warning(f"BB: failed to fetch klines ({exc})")
            return self._klines_cache  # return stale cache if available

    @staticmethod
    def _bollinger_bands(
        closes: np.ndarray, period: int, n_std: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Bollinger Bands with numpy only (no pandas / ta-lib).

        Uses a simple rolling window over the closes array.
        Population std (ddof=0) to match the original Bollinger definition.

        Returns:
            (upper, middle, lower) — arrays of same length as closes.
            First (period - 1) entries are NaN.
        """
        n = len(closes)
        upper  = np.full(n, np.nan)
        middle = np.full(n, np.nan)
        lower  = np.full(n, np.nan)

        for i in range(period - 1, n):
            window = closes[i - period + 1 : i + 1]
            mid = float(np.mean(window))
            std = float(np.std(window, ddof=0))
            middle[i] = mid
            upper[i]  = mid + n_std * std
            lower[i]  = mid - n_std * std

        return upper, middle, lower

    def _analyse_squeeze(self, klines: list) -> Optional[dict]:
        """
        Perform the full Bollinger Squeeze analysis on the kline series.

        Design: squeeze is detected on the PENULTIMATE bar (closes[:-1]); the
        LAST close tells us the breakout direction.  This avoids the paradox
        of a candle being both "squeezed" and "breaking out" in the same bar,
        since a strong outlier close would widen the bands it's included in.

        Returns a dict with:
          squeeze            : bool   — was the penultimate candle in a squeeze?
          squeeze_threshold  : float  — the percentile threshold used
          direction          : "UP" | "DOWN" | None
          edge_score         : float  — normalised breakout strength (0 → ~1+)
          bb_width           : float  — penultimate BB width (upper-lower)/middle
          momentum_confirms  : bool   — last 3 closes trend matches direction

        Returns None if there is insufficient data.
        """
        # Need at least bb_period + 1 candles (bb_period for the squeeze bar,
        # +1 for the breakout candle to evaluate)
        if len(klines) < self.bb_period + 1:
            return None

        closes = np.array([k["close"] for k in klines], dtype=np.float64)

        # ── Compute Bollinger Bands on all-but-last closes ───────────────
        # The bands ending at closes[-2] define the "squeeze state" before
        # the latest price tick arrives.
        prev_closes = closes[:-1]
        upper, middle, lower = self._bollinger_bands(prev_closes, self.bb_period, self.bb_std)

        # Keep only the valid (non-NaN) band values
        valid = ~np.isnan(upper)
        if np.sum(valid) < 2:
            return None

        bb_widths = (upper[valid] - lower[valid]) / middle[valid]

        # Penultimate candle (last bar of the squeeze window) band values
        prev_upper  = upper[-1]
        prev_lower  = lower[-1]
        prev_middle = middle[-1]

        if np.isnan(prev_upper):
            return None

        prev_bb_width = float((prev_upper - prev_lower) / prev_middle)

        # ── Squeeze detection on the penultimate bar ─────────────────────
        window_widths = bb_widths[-20:] if len(bb_widths) >= 20 else bb_widths
        squeeze_threshold = float(np.percentile(window_widths, self.bb_squeeze_percentile))
        is_squeeze = prev_bb_width < squeeze_threshold

        logger.debug(
            f"BB squeeze check (prev bar): width={prev_bb_width:.6f} "
            f"threshold={squeeze_threshold:.6f} squeeze={is_squeeze} "
            f"prev_upper={prev_upper:.4f} prev_lower={prev_lower:.4f}"
        )

        if not is_squeeze:
            return {
                "squeeze": False,
                "squeeze_threshold": squeeze_threshold,
                "direction": None,
                "edge_score": 0.0,
                "bb_width": prev_bb_width,
                "momentum_confirms": False,
            }

        # ── Breakout detection on the LAST close ─────────────────────────
        cur_close  = closes[-1]
        band_range = prev_upper - prev_lower   # width of the squeezed bands
        direction: Optional[str] = None
        edge_score = 0.0

        if cur_close > prev_upper and band_range > 0:
            direction = "UP"
            edge_score = float((cur_close - prev_upper) / band_range)
        elif cur_close < prev_lower and band_range > 0:
            direction = "DOWN"
            edge_score = float((prev_lower - cur_close) / band_range)

        logger.debug(
            f"BB breakout: cur_close={cur_close:.4f} direction={direction} "
            f"edge_score={edge_score:.4f}"
        )

        # ── 3-candle momentum confirmation ───────────────────────────────
        momentum_confirms = False
        if direction is not None and len(closes) >= 3:
            c0, c1, c2 = closes[-3], closes[-2], closes[-1]
            if direction == "UP":
                momentum_confirms = bool(c2 > c1 > c0)
            else:
                momentum_confirms = bool(c2 < c1 < c0)

        return {
            "squeeze": True,
            "squeeze_threshold": squeeze_threshold,
            "direction": direction,
            "edge_score": edge_score,
            "bb_width": prev_bb_width,
            "momentum_confirms": momentum_confirms,
        }
