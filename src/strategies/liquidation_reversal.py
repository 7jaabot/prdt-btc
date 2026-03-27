"""
liquidation_reversal.py — Liquidation Reversal Strategy

Hypothesis: Large forced liquidations create abrupt price moves that temporarily
exhaust the dominant side.
  - If many LONG positions just got liquidated (price dropped hard), selling
    pressure is now depleted → mean-reversion UP is more likely.
  - If many SHORT positions just got liquidated (price spiked hard), buying
    pressure is exhausted → mean-reversion DOWN is more likely.

Data source: Binance Futures public REST endpoint (no auth required)
  GET https://fapi.binance.com/fapi/v1/allForceOrders?symbol=BNBUSDT

Side convention in Binance liquidation data:
  - side == "SELL"  → the system sold (liquidated) a LONG position
  - side == "BUY"   → the system bought back (liquidated) a SHORT position

Edge = z-score of recent liq imbalance vs rolling baseline.
"""

import logging
import time
from typing import Optional

import numpy as np
import requests

from .base import BaseStrategy
from strategy import Signal, WindowInfo, compute_edge, compute_position_size

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

BINANCE_FUTURES_REST = "https://fapi.binance.com"
FORCE_ORDERS_PATH = "/fapi/v1/allForceOrders"

# How many historical buckets to keep for z-score baseline
# Each bucket = 1 lookback window (default 5 min). We keep 12 → 1h of history.
_HISTORY_MAX_BUCKETS = 12

# Clamp for P(Up) from liquidation signal
_P_UP_MIN = 0.35
_P_UP_MAX = 0.65


# ─────────────────────────────────────────────────────────────────────────────
# Strategy
# ─────────────────────────────────────────────────────────────────────────────

class LiquidationReversalStrategy(BaseStrategy):
    """
    ⚡ Liquidation Reversal Strategy for BNB Up/Down 5mn.

    Monitors recent forced liquidations on BNB perpetual futures.
    When a significant imbalance (z-score) is detected, bets on mean-reversion:
      - Large LONG liq volume → UP bet
      - Large SHORT liq volume → DOWN bet

    Config keys (under "liquidation_reversal" in strategy config):
      liquidation_min_volume_usdt (float): minimum net liq volume to trigger (default 100_000)
      liquidation_lookback_minutes (float): look-back window in minutes (default 5)
    """

    @property
    def name(self) -> str:
        return "⚡ Liquidation Reversal"

    def __init__(self, config: dict):
        super().__init__(config)

        # Strategy-specific config
        liq_cfg = config.get("liquidation_reversal", {})
        self.min_volume_usdt: float = liq_cfg.get("liquidation_min_volume_usdt", 100_000.0)
        self.lookback_minutes: float = liq_cfg.get("liquidation_lookback_minutes", 5.0)

        # Rolling history of (long_vol_usdt, short_vol_usdt) per window
        # Used to compute z-score baseline
        self._history: list[tuple[float, float]] = []  # [(long_vol, short_vol), ...]

        # Response cache: (timestamp_fetched, raw_data)
        self._cache_ts: float = 0.0
        self._cache_data: list[dict] = []
        self._cache_ttl: float = 10.0  # seconds — avoid hammering API

        # Use fair-odds or pool-derived price
        cfg_s = config.get("strategy", {})
        self.use_fair_odds: bool = cfg_s.get("use_fair_odds", True)
        self.FAIR_ODDS_PRICE: float = 0.50

    # ─────────────────────────────────────────────────────────────────────────
    # Data fetching
    # ─────────────────────────────────────────────────────────────────────────

    def _fetch_liquidations(self, lookback_minutes: float) -> list[dict]:
        """
        Fetch recent BNBUSDT liquidation orders from Binance Futures REST API.

        Public endpoint — no API key required.

        Returns:
            List of liquidation order dicts with at minimum:
              - side: "BUY" (short liq) or "SELL" (long liq)
              - executedQty: filled quantity in BNB
              - averagePrice: fill price in USDT
              - time: Unix timestamp in ms
        """
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - int(lookback_minutes * 60 * 1000)

        # Cache hit?
        if time.time() - self._cache_ts < self._cache_ttl and self._cache_data:
            # Filter cached data to the lookback window
            return [r for r in self._cache_data if r.get("time", 0) >= start_ms]

        try:
            resp = requests.get(
                BINANCE_FUTURES_REST + FORCE_ORDERS_PATH,
                params={
                    "symbol": "BNBUSDT",
                    "startTime": start_ms,
                    "limit": 1000,  # max allowed
                },
                timeout=5.0,
            )
            resp.raise_for_status()
            data: list[dict] = resp.json()
            self._cache_ts = time.time()
            self._cache_data = data
            logger.debug(f"Fetched {len(data)} BNBUSDT liquidation records from Binance")
            return data
        except Exception as exc:
            logger.warning(f"⚡ Liquidation fetch failed: {exc}")
            return []

    # ─────────────────────────────────────────────────────────────────────────
    # Volume aggregation
    # ─────────────────────────────────────────────────────────────────────────

    def _aggregate_volumes(
        self, liquidations: list[dict], lookback_minutes: float
    ) -> tuple[float, float]:
        """
        Sum liquidation volumes by side within the lookback window.

        Returns:
            (long_vol_usdt, short_vol_usdt)
              long_vol_usdt : USDT value of liquidated LONG positions (side == "SELL")
              short_vol_usdt: USDT value of liquidated SHORT positions (side == "BUY")
        """
        cutoff_ms = int((time.time() - lookback_minutes * 60) * 1000)
        long_vol = 0.0
        short_vol = 0.0

        for order in liquidations:
            ts = order.get("time", 0)
            if ts < cutoff_ms:
                continue

            side = order.get("side", "")
            try:
                qty = float(order.get("executedQty", 0) or 0)
                price = float(order.get("averagePrice", 0) or 0)
            except (TypeError, ValueError):
                continue

            vol_usdt = qty * price

            if side == "SELL":
                # System sold (liquidated) a LONG position → long liq
                long_vol += vol_usdt
            elif side == "BUY":
                # System bought back (liquidated) a SHORT position → short liq
                short_vol += vol_usdt

        return long_vol, short_vol

    # ─────────────────────────────────────────────────────────────────────────
    # Z-score calculation
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_zscore(self, net_imbalance: float) -> float:
        """
        Compute z-score of current net imbalance vs rolling history.

        net_imbalance > 0 → more LONG liq (oversold, UP signal)
        net_imbalance < 0 → more SHORT liq (overbought, DOWN signal)

        Returns:
            Z-score (signed). Returns 0.0 if not enough history.
        """
        if len(self._history) < 2:
            return 0.0

        historical_nets = [lv - sv for lv, sv in self._history]
        mu = float(np.mean(historical_nets))
        sigma = float(np.std(historical_nets))

        if sigma < 1.0:  # Avoid division by near-zero
            return 0.0

        return (net_imbalance - mu) / sigma

    def _update_history(self, long_vol: float, short_vol: float) -> None:
        """Append current window volumes to rolling history, pruning oldest."""
        self._history.append((long_vol, short_vol))
        if len(self._history) > _HISTORY_MAX_BUCKETS:
            self._history.pop(0)

    # ─────────────────────────────────────────────────────────────────────────
    # P(Up) estimation from liquidation signal
    # ─────────────────────────────────────────────────────────────────────────

    def _estimate_p_up(self, zscore: float) -> float:
        """
        Convert signed z-score to P(Up) estimate.

        Positive z-score (long liq dominates) → P(Up) > 0.5
        Negative z-score (short liq dominates) → P(Up) < 0.5

        Uses a logistic-like mapping clamped to [_P_UP_MIN, _P_UP_MAX].
        Each 1-sigma of imbalance adds ~5% to our P(Up) estimate.
        """
        # Base: logistic mapping with scale factor 0.05 per z-unit
        # z=2 → +0.10, z=-2 → -0.10
        delta = 0.05 * zscore
        p_up = 0.5 + delta
        return max(_P_UP_MIN, min(_P_UP_MAX, p_up))

    # ─────────────────────────────────────────────────────────────────────────
    # Main evaluate
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

        # Only act during entry window
        if not window.is_entry_window and window.seconds_remaining > self.entry_window_seconds:
            return None

        # ── 1. Fetch liquidation data ─────────────────────────────────────────
        raw_liqs = self._fetch_liquidations(self.lookback_minutes)

        # ── 2. Aggregate volumes ──────────────────────────────────────────────
        long_vol, short_vol = self._aggregate_volumes(raw_liqs, self.lookback_minutes)
        net_imbalance = long_vol - short_vol  # >0 → long liq dominant → UP signal

        total_vol = long_vol + short_vol

        logger.info(
            f"⚡ Liq volumes ({self.lookback_minutes:.0f}m): "
            f"LONG={long_vol/1000:.1f}k SHORT={short_vol/1000:.1f}k "
            f"TOTAL={total_vol/1000:.1f}k USDT"
        )

        # ── 3. Minimum volume threshold ───────────────────────────────────────
        if total_vol < self.min_volume_usdt:
            self.last_skip_reason = (
                f"⏸ Total liq volume too low "
                f"({total_vol/1000:.1f}k < {self.min_volume_usdt/1000:.0f}k USDT)"
            )
            # Still update history so baseline grows
            self._update_history(long_vol, short_vol)
            return None

        # ── 4. Z-score computation ────────────────────────────────────────────
        zscore = self._compute_zscore(net_imbalance)
        self._update_history(long_vol, short_vol)

        logger.info(
            f"⚡ Liq imbalance: net={net_imbalance/1000:.1f}k USDT "
            f"z={zscore:+.2f} (history={len(self._history)} buckets)"
        )

        # Need a meaningful z-score to have an edge
        if abs(zscore) < 1.0:
            self.last_skip_reason = (
                f"⏸ Liq z-score too low ({zscore:+.2f} < ±1.0) — no edge"
            )
            return None

        # ── 5. P(Up) from liquidation signal ─────────────────────────────────
        p_up = self._estimate_p_up(zscore)

        # ── 6. Compare vs pool/fair price ────────────────────────────────────
        effective_yes_price = (
            self.FAIR_ODDS_PRICE if self.use_fair_odds else yes_price
        )
        edge, side = compute_edge(p_up, effective_yes_price)

        logger.info(
            f"⚡ Liq signal: z={zscore:+.2f} P(Up)={p_up:.3f} | "
            f"price={effective_yes_price:.3f} edge={edge:.3f} side={side}"
        )

        # ── 7. Edge threshold ─────────────────────────────────────────────────
        if edge <= self.edge_threshold:
            self.last_skip_reason = (
                f"⏸ Edge too low ({edge:.3f} ≤ {self.edge_threshold:.2f}) | "
                f"P(Up)={p_up:.2f} z={zscore:+.2f}"
            )
            return None

        # ── 8. Position sizing ────────────────────────────────────────────────
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
            self.last_skip_reason = "⏸ Kelly sizing → 0 position"
            return None

        # Pool-based filters (when not using fair odds)
        if not self.use_fair_odds and prices:
            if pool_total_bnb < self.min_pool_bnb:
                self.last_skip_reason = (
                    f"⏸ Pool too small ({pool_total_bnb:.2f} BNB "
                    f"< {self.min_pool_bnb:.1f} min)"
                )
                return None

            opposite_bnb = pool_bear_bnb if side == "YES" else pool_bull_bnb
            if opposite_bnb < 0.01:
                self.last_skip_reason = (
                    f"⏸ No counterparty "
                    f"({'BEAR' if side == 'YES' else 'BULL'} side empty)"
                )
                return None

            if prices:
                bet_bnb = pos_size / prices[-1]
                side_bnb = pool_bull_bnb if side == "YES" else pool_bear_bnb
                if side_bnb > 0 and bet_bnb / side_bnb > self.max_bet_share_of_side:
                    self.last_skip_reason = (
                        f"⏸ Bet too large ({bet_bnb:.3f} BNB = "
                        f"{bet_bnb/side_bnb:.0%} of {side} side)"
                    )
                    return None

        # ── 9. Build signal ───────────────────────────────────────────────────
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
            f"⚡ Liquidation Reversal signal: {signal} | "
            f"liq_long={long_vol/1000:.1f}k liq_short={short_vol/1000:.1f}k z={zscore:+.2f}"
        )
        return signal
