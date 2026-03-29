"""
funding_rate.py — Funding Rate Contrarian Strategy

Hypothesis: Binance perpetual futures funding rate is a contrarian signal.
- Strongly positive FR → too many longs → overloaded long side → price likely to drop → bet DOWN
- Strongly negative FR → too many shorts → overloaded short side → price likely to rise → bet UP
- Neutral FR → no signal, skip

Funding rate settlement happens every 8h (00:00, 08:00, 16:00 UTC).
Between settlements, the *accumulating* rate (via premiumIndex) can also be used.

Edge formula:
    edge = abs(funding_rate) / max_expected_rate  (normalized to [0, 1])
    p_win = 0.5 + edge * 0.15                    (maps to [0.5, 0.65] range)

This maps to a realistic win-rate range consistent with empirical studies on
funding rate as a contrarian signal at 5-minute horizons.

API endpoint: https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BNBUSDT
Returns markPrice, indexPrice, lastFundingRate, etc.
We use the real-time mark-index premium: (markPrice - indexPrice) / indexPrice
This reflects live funding pressure, unlike lastFundingRate which only updates at settlement.
"""

import logging
import time
from typing import Optional

import urllib.request
import json

from .base import BaseStrategy
from strategy import Signal, WindowInfo, compute_position_size

logger = logging.getLogger(__name__)

# Typical max mark-index premium for BNB.
# Real-time premium values range ~0.00005 to 0.002 in volatile conditions.
# We normalize relative to 0.0005 — a moderate-strong signal for BNB.
DEFAULT_MAX_EXPECTED_RATE = 0.0005

# p_win boost per unit of normalized edge (max boost = 0.15 → p_win up to 0.65)
P_WIN_EDGE_SCALE = 0.15

# How long (seconds) to cache the funding rate before re-fetching
CACHE_TTL_SECONDS = 60.0


class FundingRateStrategy(BaseStrategy):
    """
    Funding Rate Contrarian Strategy for PancakeSwap Prediction V2.

    Uses Binance perpetual futures funding rate (BNBUSDT) as a contrarian
    directional signal. Sizes bets via fractional Kelly criterion.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        cfg = config.get("strategy", {})

        # Thresholds for mark-index premium (e.g. 0.00005 = 0.005%)
        # Premium is more volatile than settlement funding rate, so we can
        # keep meaningful thresholds while still triggering frequently
        self.positive_threshold: float = cfg.get(
            "funding_rate_positive_threshold", 0.00005
        )
        self.negative_threshold: float = cfg.get(
            "funding_rate_negative_threshold", -0.00005
        )

        # API endpoint — premiumIndex gives real-time rate, not just settlement snapshots
        self.funding_rate_url: str = cfg.get(
            "funding_rate_url",
            "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BNBUSDT",
        )

        # Max expected absolute rate for normalization
        self.max_expected_rate: float = cfg.get(
            "funding_rate_max_expected", DEFAULT_MAX_EXPECTED_RATE
        )

        # Internal cache
        self._cached_rate: Optional[float] = None
        self._cache_ts: float = 0.0

        # HTTP timeout for API call
        self._http_timeout: float = cfg.get("funding_rate_timeout", 3.0)

    @property
    def name(self) -> str:
        return "💰 Funding Rate Contrarian"

    # ─────────────────────────────────────────────────────────────────────
    # Prefetch (sniper pre-load phase)
    # ─────────────────────────────────────────────────────────────────────

    def prefetch(self, prices: list[float], epoch=None) -> None:
        """Pre-fetch funding rate and cache it for the sniper window."""
        super().prefetch(prices, epoch)
        logger.debug("FundingRate: prefetching premium index...")
        rate = self._fetch_funding_rate()
        if rate is not None:
            self._prefetch_cache["funding_rate"] = rate
            logger.info(f"FundingRate prefetch: premium={rate:.8f}")
        else:
            logger.warning("FundingRate prefetch: failed to fetch premium index")

    # ─────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    def _fetch_funding_rate(self) -> Optional[float]:
        """
        Fetch the current BNBUSDT funding rate from Binance Futures API.

        Returns the funding rate as a float, or None on error.
        Uses an in-memory cache with TTL to avoid spamming the API.
        """
        now = time.time()
        if self._cached_rate is not None and (now - self._cache_ts) < CACHE_TTL_SECONDS:
            logger.debug(
                f"Funding premium cache hit: {self._cached_rate:.6f} "
                f"(age={now - self._cache_ts:.0f}s)"
            )
            return self._cached_rate

        try:
            req = urllib.request.Request(
                self.funding_rate_url,
                headers={"User-Agent": "bnb-updown-bot/1.0"},
            )
            with urllib.request.urlopen(req, timeout=self._http_timeout) as resp:
                raw = resp.read().decode("utf-8")
                data = json.loads(raw)

            if not isinstance(data, dict):
                logger.warning(f"Unexpected premiumIndex response type: {type(data)}")
                return None

            mark = float(data.get("markPrice", 0))
            index = float(data.get("indexPrice", 0))

            if index <= 0:
                logger.warning(f"Invalid indexPrice: {index}")
                return None

            # Real-time premium: positive = longs dominate, negative = shorts dominate
            premium = (mark - index) / index
            self._cached_rate = premium
            self._cache_ts = now

            logger.info(
                f"Funding premium fetched: {premium:.8f} "
                f"({premium * 100:.5f}%) — mark={mark:.2f} index={index:.2f}"
            )
            return premium

        except urllib.error.URLError as exc:
            logger.error(f"Funding rate API network error: {exc}")
        except json.JSONDecodeError as exc:
            logger.error(f"Funding rate API JSON parse error: {exc}")
        except (KeyError, ValueError, TypeError) as exc:
            logger.error(f"Funding rate API data parse error: {exc}")
        except Exception as exc:
            logger.error(f"Funding rate API unexpected error: {exc}")

        # On error, return stale cache if available (better than nothing)
        if self._cached_rate is not None:
            age = now - self._cache_ts
            logger.warning(
                f"Returning stale funding premium cache: {self._cached_rate:.6f} "
                f"(age={age:.0f}s)"
            )
            return self._cached_rate

        return None

    def _compute_p_win(self, funding_rate: float, side: str) -> float:
        """
        Convert a funding rate magnitude into a win probability estimate.

        Maps abs(funding_rate) / max_expected_rate → [0, 1] edge signal,
        then applies to base probability of 0.5.

        Args:
            funding_rate: Raw funding rate (e.g., 0.0003).
            side: "YES" (UP) or "NO" (DOWN) — used for logging only.

        Returns:
            p_win in [0.50, 0.65].
        """
        abs_rate = abs(funding_rate)
        normalized_edge = min(1.0, abs_rate / max(self.max_expected_rate, 1e-9))
        p_win = 0.5 + normalized_edge * P_WIN_EDGE_SCALE
        p_win = max(0.50, min(0.65, p_win))

        logger.debug(
            f"p_win: abs_rate={abs_rate:.6f} "
            f"normalized={normalized_edge:.3f} p_win={p_win:.3f} side={side}"
        )
        return p_win

    # ─────────────────────────────────────────────────────────────────────
    # BaseStrategy interface
    # ─────────────────────────────────────────────────────────────────────

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
        Evaluate and return a Signal based on funding rate contrarian logic.

        Signal conditions:
        - funding_rate > positive_threshold → DOWN bet (longs overloaded)
        - funding_rate < negative_threshold → UP bet (shorts overloaded)
        - Otherwise → skip (neutral market sentiment)
        """
        self.last_skip_reason = None

        # Only evaluate within entry window
        if not window.is_entry_window and window.seconds_remaining > self.entry_window_seconds:
            return None

        # Fetch mark-index premium (use prefetch cache if available)
        if "funding_rate" in self._prefetch_cache:
            funding_rate = self._prefetch_cache["funding_rate"]
            logger.debug(f"FundingRate: using prefetched premium={funding_rate:.8f}")
        else:
            funding_rate = self._fetch_funding_rate()

        if funding_rate is None:
            self.last_skip_reason = "⏸ Funding premium unavailable (API error)"
            logger.warning("Skipping: could not fetch funding premium")
            return None

        # Determine directional signal (contrarian logic)
        if funding_rate > self.positive_threshold:
            # Positive premium = longs overloaded → expect down move → bet DOWN (NO)
            side = "NO"
            logger.info(
                f"Premium={funding_rate:.8f} > threshold={self.positive_threshold:.8f} "
                f"→ LONGS overloaded → bet DOWN"
            )

        elif funding_rate < self.negative_threshold:
            # Negative premium = shorts overloaded → expect up move → bet UP (YES)
            side = "YES"
            logger.info(
                f"Premium={funding_rate:.8f} < threshold={self.negative_threshold:.8f} "
                f"→ SHORTS overloaded → bet UP"
            )

        else:
            self.last_skip_reason = (
                f"⏸ Premium neutral ({funding_rate:.8f}), "
                f"thresholds=[{self.negative_threshold:.8f}, {self.positive_threshold:.8f}]"
            )
            logger.info(f"Skipping: {self.last_skip_reason}")
            return None

        # Compute edge and p_win from signal magnitude
        abs_rate = abs(funding_rate)
        edge = min(1.0, abs_rate / max(self.max_expected_rate, 1e-9))

        if edge <= self.edge_threshold:
            self.last_skip_reason = (
                f"⏸ Edge too low ({edge:.3f} ≤ {self.edge_threshold:.2f}) "
                f"| FR={funding_rate:.6f} | side={side}"
            )
            logger.info(f"Skipping: {self.last_skip_reason}")
            return None

        # Derive p_win from the funding rate magnitude
        p_win = self._compute_p_win(funding_rate, side)

        # For Kelly sizing: p_up is needed
        # YES = UP bet: p_up = p_win
        # NO  = DOWN bet: p_up = 1 - p_win
        p_up = p_win if side == "YES" else (1.0 - p_win)

        # Use fair odds (0.5) as effective price — consistent with GBM strategy default
        effective_yes_price = 0.50

        # Compute position size via Kelly
        raw_k, pos_size = compute_position_size(
            edge=edge,
            p_up=p_up,
            side=side,
            yes_price=effective_yes_price,
            bankroll=self.bankroll,
            kelly_fraction_cap=self.kelly_fraction_cap,
            max_usdc=self.max_position_usdc,
        )

        if pos_size <= 0:
            self.last_skip_reason = (
                f"⏸ Position size is 0 (Kelly={raw_k:.4f}) "
                f"| FR={funding_rate:.6f} | side={side}"
            )
            return None

        signal = Signal(
            side=side,
            edge=edge,
            p_up=p_up,
            yes_price=effective_yes_price,
            kelly_fraction=raw_k,
            position_size_usdc=pos_size,
            timestamp=time.time(),
            is_mock=is_mock_data,
            bull_pct=pool_bull_bnb / pool_total_bnb if pool_total_bnb > 0 else 0.0,
            bear_pct=pool_bear_bnb / pool_total_bnb if pool_total_bnb > 0 else 0.0,
        )

        logger.info(
            f"🎯 Signal: {signal} | premium={funding_rate:.8f} "
            f"({funding_rate * 100:.5f}%)"
        )
        return signal
