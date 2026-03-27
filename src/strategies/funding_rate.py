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

API endpoint: https://fapi.binance.com/fapi/v1/fundingRate?symbol=BNBUSDT&limit=1
Response: [{"symbol": "BNBUSDT", "fundingTime": <ms>, "fundingRate": "0.00010000", ...}]

Alternatively: https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BNBUSDT
Returns lastFundingRate + real-time accumulating rate info.
"""

import logging
import time
from typing import Optional

import urllib.request
import json

from .base import BaseStrategy
from strategy import Signal, WindowInfo, compute_position_size

logger = logging.getLogger(__name__)

# Typical max funding rate for BNB in normal/stressed markets.
# Empirically: 0.01% (0.0001) is common, 0.1% (0.001) is extreme.
# We normalize relative to 0.075% (0.00075) — a moderate-strong signal.
DEFAULT_MAX_EXPECTED_RATE = 0.00075

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

        # Thresholds (as fractions, e.g. 0.0001 = 0.01%)
        self.positive_threshold: float = cfg.get(
            "funding_rate_positive_threshold", 0.0001
        )
        self.negative_threshold: float = cfg.get(
            "funding_rate_negative_threshold", -0.0001
        )

        # API endpoint (configurable)
        self.funding_rate_url: str = cfg.get(
            "funding_rate_url",
            "https://fapi.binance.com/fapi/v1/fundingRate?symbol=BNBUSDT&limit=1",
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
                f"Funding rate cache hit: {self._cached_rate:.6f} "
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

            # API returns a list: [{"symbol": ..., "fundingRate": "0.00010000", ...}]
            if isinstance(data, list):
                if not data:
                    logger.warning("Binance funding rate API returned empty list")
                    return None
                entry = data[0]
            elif isinstance(data, dict):
                # premiumIndex endpoint returns a single dict with lastFundingRate
                entry = data
            else:
                logger.warning(f"Unexpected funding rate API response type: {type(data)}")
                return None

            # Support both endpoint schemas
            rate_key = "fundingRate" if "fundingRate" in entry else "lastFundingRate"
            if rate_key not in entry:
                logger.warning(f"No funding rate key in response: {entry}")
                return None

            rate = float(entry[rate_key])
            self._cached_rate = rate
            self._cache_ts = now

            logger.info(
                f"Funding rate fetched: {rate:.6f} "
                f"({rate * 100:.4f}%) — symbol={entry.get('symbol', '?')}"
            )
            return rate

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
                f"Returning stale funding rate cache: {self._cached_rate:.6f} "
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

        # Fetch funding rate
        funding_rate = self._fetch_funding_rate()

        if funding_rate is None:
            self.last_skip_reason = "⏸ Funding rate unavailable (API error)"
            logger.warning("Skipping: could not fetch funding rate")
            return None

        # Determine directional signal (contrarian logic)
        if funding_rate > self.positive_threshold:
            # Longs overloaded → expect down move → bet DOWN (NO)
            side = "NO"
            logger.info(
                f"FR={funding_rate:.6f} > threshold={self.positive_threshold:.6f} "
                f"→ LONGS overloaded → bet DOWN"
            )

        elif funding_rate < self.negative_threshold:
            # Shorts overloaded → expect up move → bet UP (YES)
            side = "YES"
            logger.info(
                f"FR={funding_rate:.6f} < threshold={self.negative_threshold:.6f} "
                f"→ SHORTS overloaded → bet UP"
            )

        else:
            self.last_skip_reason = (
                f"⏸ Funding rate neutral ({funding_rate:.6f}), "
                f"thresholds=[{self.negative_threshold:.6f}, {self.positive_threshold:.6f}]"
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
            f"🎯 Signal: {signal} | funding_rate={funding_rate:.6f} "
            f"({funding_rate * 100:.4f}%)"
        )
        return signal
