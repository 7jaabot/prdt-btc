"""
correlation_arbitrage.py — BTC/ETH Correlation Arbitrage Strategy

Hypothesis: BNB is strongly correlated with BTC and ETH but often lags them
by 30–120 seconds on 1m timeframes. When BTC and ETH both move strongly in
one direction but BNB has not yet caught up (< 50% of the BTC move), BNB
is likely to follow — giving a predictable edge.

Research basis:
- Lead-lag relationships between large-caps (BTC, ETH) and altcoins are
  well-documented in crypto markets (Brandvold et al. 2015; Ciaian et al.
  2018; Będowska-Sójka & Kliber 2021).
- BNB typically has a 1–3 minute lag behind BTC impulses during trending
  sessions, especially during high-volume moves (> 0.3% in 5 min).
- Weighted signal: BTC carries more predictive weight (60%) than ETH (40%)
  because BNB/BTC correlation is historically tighter.

Logic:
  - Fetch 1m klines for BTCUSDT and ETHUSDT (last 6 candles → 5-min window)
  - Compute 5-min return for BTC, ETH, and BNB (from prices[] feed)
  - Weighted combined move = BTC_ret * btc_weight + ETH_ret * eth_weight
  - If weighted move > threshold AND BNB has done < lag_threshold * weighted_move
      → BNB lagging upward  → bet UP
  - If weighted move < -threshold AND BNB has done > lag_threshold * weighted_move
      → BNB lagging downward → bet DOWN
  - Edge = |weighted_move - bnb_return| normalised to [0, 1]

Config keys (under "strategy" namespace):
  corr_btc_threshold   : float  = 0.003  (min 0.3% weighted move to trigger)
  corr_lag_threshold   : float  = 0.5    (BNB must have done < 50% of BTC/ETH move)
  corr_btc_weight      : float  = 0.6    (weight for BTC; ETH = 1 - weight)
"""

import logging
import time
from typing import Optional
from urllib.request import urlopen
from urllib.error import URLError
import json

import numpy as np

from .base import BaseStrategy
from strategy import (
    Signal, WindowInfo,
    compute_edge, compute_position_size,
)

logger = logging.getLogger(__name__)

# Binance REST endpoint for klines
_BINANCE_KLINES_URL = (
    "https://api.binance.com/api/v3/klines"
    "?symbol={symbol}&interval=1m&limit={limit}"
)

# Binance kline field indices
_K_OPEN = 1
_K_CLOSE = 4

# Cache TTL in seconds — refresh at most once per 45s to avoid rate limits
_CACHE_TTL = 45.0

# How many 1m candles to fetch (6 gives us the last 5 complete 1m returns)
_KLINE_LIMIT = 6

# Max p_up deviation from 0.5 (keeps predictions humble)
_MAX_P_DEVIATION = 0.15  # p_up in [0.35, 0.65]

# Fair odds reference price (pool price ignored)
_FAIR_ODDS_PRICE = 0.50


class CorrelationArbitrageStrategy(BaseStrategy):
    """
    BTC/ETH Correlation Arbitrage Strategy for PancakeSwap Prediction V2.

    Exploits the lag between large-cap crypto moves (BTC, ETH) and BNB's
    delayed response.  Bets UP/DOWN when BNB has not yet caught up with a
    strong directional impulse in BTC and ETH.

    Klines are fetched from the Binance REST API and cached for 45 seconds
    to avoid excess API calls across evaluate() ticks.
    """

    @property
    def name(self) -> str:
        return "🔗 BTC/ETH Correlation Arbitrage"

    def __init__(self, config: dict):
        super().__init__(config)

        cfg = config.get("strategy", {})

        # Minimum weighted BTC/ETH move required to trigger a signal (0.3%)
        # Lowered from 0.003 — observed weighted moves: median=0.0008, p90=0.003
        self.corr_btc_threshold: float = cfg.get("corr_btc_threshold", 0.0012)

        # BNB must have moved less than this fraction of the BTC/ETH impulse
        self.corr_lag_threshold: float = cfg.get("corr_lag_threshold", 0.5)

        # Weight given to BTC in the combined signal (ETH = 1 - weight)
        self.corr_btc_weight: float = cfg.get("corr_btc_weight", 0.6)

        # Kline cache: keyed by symbol → (timestamp, returns_list)
        self._kline_cache: dict[str, tuple[float, list[float]]] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface
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
        Evaluate the correlation arbitrage signal.

        Returns a Signal if BNB is lagging a strong BTC/ETH move, else None.
        Sets self.last_skip_reason when returning None with a human-readable cause.
        """
        self.last_skip_reason = None

        # Only act inside the entry window (last N seconds before lock)
        if not window.is_entry_window and window.seconds_remaining > self.entry_window_seconds:
            return None

        # Need enough BNB prices to compute a 5-min return
        if len(prices) < 5:
            self.last_skip_reason = f"⏸ Not enough BNB price data ({len(prices)} pts)"
            return None

        # ── 1. Fetch BTC and ETH 5-min returns ───────────────────────────────
        btc_return = self._get_5min_return("BTCUSDT")
        eth_return = self._get_5min_return("ETHUSDT")

        if btc_return is None or eth_return is None:
            self.last_skip_reason = "⏸ Could not fetch BTC/ETH klines"
            return None

        # ── 2. Compute BNB 5-min return from the live price feed ─────────────
        bnb_return = self._compute_bnb_return(prices)

        # ── 3. Compute the weighted BTC/ETH impulse ──────────────────────────
        eth_weight = 1.0 - self.corr_btc_weight
        weighted_move = (
            btc_return * self.corr_btc_weight
            + eth_return * eth_weight
        )

        logger.info(
            f"[CorrArb] BTC={btc_return:+.4f}  ETH={eth_return:+.4f}  "
            f"BNB={bnb_return:+.4f}  weighted={weighted_move:+.4f}  "
            f"threshold={self.corr_btc_threshold:.4f}"
        )

        # ── 4. Determine signal direction ─────────────────────────────────────
        side = self._classify_signal(weighted_move, bnb_return)

        if side is None:
            self.last_skip_reason = (
                f"⏸ No lag signal | weighted={weighted_move:+.4f} "
                f"BNB={bnb_return:+.4f} threshold={self.corr_btc_threshold:.4f}"
            )
            return None

        # ── 5. Compute p_up and edge ──────────────────────────────────────────
        p_up = self._compute_p_up(weighted_move, bnb_return, side)
        edge, computed_side = compute_edge(p_up, _FAIR_ODDS_PRICE)

        # Sanity check: computed side should match our directional signal
        if computed_side != side:
            self.last_skip_reason = (
                f"⏸ Side mismatch (signal={side} vs edge={computed_side}) — skipping"
            )
            return None

        logger.info(
            f"[CorrArb] side={side}  p_up={p_up:.3f}  "
            f"edge={edge:.3f}  threshold={self.edge_threshold:.3f}"
        )

        if edge <= self.edge_threshold:
            self.last_skip_reason = (
                f"⏸ Edge too low ({edge:.3f} ≤ {self.edge_threshold:.2f}) | "
                f"P(Up)={p_up:.3f} | {side}"
            )
            return None

        # ── 6. Position sizing ────────────────────────────────────────────────
        raw_k, pos_size = compute_position_size(
            edge=edge,
            p_up=p_up,
            side=side,
            yes_price=_FAIR_ODDS_PRICE,
            bankroll=self.bankroll,
            kelly_fraction_cap=self.kelly_fraction_cap,
            max_usdc=self.max_position_usdc,
        )

        if pos_size <= 0:
            return None

        signal = Signal(
            side=side,
            edge=edge,
            p_up=p_up,
            yes_price=_FAIR_ODDS_PRICE,
            kelly_fraction=raw_k,
            position_size_usdc=pos_size,
            timestamp=time.time(),
            is_mock=is_mock_data,
            bull_pct=pool_bull_bnb / pool_total_bnb if pool_total_bnb > 0 else 0.0,
            bear_pct=pool_bear_bnb / pool_total_bnb if pool_total_bnb > 0 else 0.0,
        )

        logger.info(f"🎯 [CorrArb] Signal generated: {signal}")
        return signal

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_5min_return(self, symbol: str) -> Optional[float]:
        """
        Return the 5-minute return for `symbol` using the last 5 closed 1m
        candles from Binance.  Results are cached for _CACHE_TTL seconds.

        Return value = (close[-1] / open[-5]) - 1   (log-linear approx)
        Returns None on network error or malformed data.
        """
        now = time.time()
        cached = self._kline_cache.get(symbol)
        if cached is not None:
            cached_ts, cached_return = cached
            if now - cached_ts < _CACHE_TTL:
                return cached_return

        url = _BINANCE_KLINES_URL.format(symbol=symbol, limit=_KLINE_LIMIT)
        try:
            with urlopen(url, timeout=5) as resp:
                raw = json.loads(resp.read().decode())
        except (URLError, OSError, ValueError) as exc:
            logger.warning(f"[CorrArb] Failed to fetch klines for {symbol}: {exc}")
            return None

        if not raw or len(raw) < 2:
            logger.warning(f"[CorrArb] Too few klines returned for {symbol}: {len(raw)}")
            return None

        try:
            # Use open of oldest candle and close of the most recent candle
            # to get a clean 5-min window return
            open_price = float(raw[0][_K_OPEN])
            close_price = float(raw[-1][_K_CLOSE])
        except (IndexError, ValueError, TypeError) as exc:
            logger.warning(f"[CorrArb] Malformed kline data for {symbol}: {exc}")
            return None

        if open_price <= 0:
            return None

        ret = (close_price / open_price) - 1.0

        self._kline_cache[symbol] = (now, ret)

        logger.debug(
            f"[CorrArb] {symbol} 5-min return: {ret:+.4f} "
            f"(open={open_price:.4f} close={close_price:.4f})"
        )
        return ret

    def _compute_bnb_return(self, prices: list[float]) -> float:
        """
        Compute BNB's return over the available price window.

        Uses the first and last price from the BinanceFeed price series.
        Clamps to [-0.05, +0.05] to avoid distortion from outliers.
        """
        if len(prices) < 2:
            return 0.0
        ret = (prices[-1] / prices[0]) - 1.0
        return max(-0.05, min(0.05, ret))

    def _classify_signal(
        self,
        weighted_move: float,
        bnb_return: float,
    ) -> Optional[str]:
        """
        Determine if there is a lag-based trading signal.

        Returns "YES" (bet UP), "NO" (bet DOWN), or None (no signal).

        Conditions:
          UP  : weighted_move > threshold  AND  bnb_return < lag_threshold * weighted_move
          DOWN: weighted_move < -threshold AND  bnb_return > lag_threshold * weighted_move
                (i.e. BNB is positive or less negative than expected)
        """
        thr = self.corr_btc_threshold
        lag = self.corr_lag_threshold

        if weighted_move > thr:
            # BTC/ETH moved up strongly
            # BNB must have done less than `lag` fraction of that move
            if bnb_return < lag * weighted_move:
                return "YES"

        elif weighted_move < -thr:
            # BTC/ETH moved down strongly
            # BNB must still be higher than `lag` fraction of that (negative) move
            # i.e. bnb_return > lag * weighted_move  (less negative than expected)
            if bnb_return > lag * weighted_move:
                return "NO"

        return None

    def _compute_p_up(
        self,
        weighted_move: float,
        bnb_return: float,
        side: str,
    ) -> float:
        """
        Map the correlation lag into a P(up) estimate.

        Edge principle: the larger the gap between weighted_move and bnb_return,
        the higher our confidence that BNB will catch up.

        Formula:
          lag_gap       = abs(weighted_move - bnb_return)
          normalised    = lag_gap / (abs(weighted_move) + 1e-9)  ∈ [0, ∞)
          clamped       = min(normalised, 1.0)                    ∈ [0, 1]
          p_deviation   = clamped * MAX_P_DEVIATION               ∈ [0, 0.15]
          p_up          = 0.5 + p_deviation  (if side == YES)
                        = 0.5 - p_deviation  (if side == NO)
        """
        lag_gap = abs(weighted_move - bnb_return)
        normalised = lag_gap / (abs(weighted_move) + 1e-9)
        clamped = min(normalised, 1.0)
        p_deviation = clamped * _MAX_P_DEVIATION

        if side == "YES":
            p_up = 0.5 + p_deviation
        else:
            p_up = 0.5 - p_deviation

        # Hard clamp to [0.35, 0.65] — never be overconfident
        return max(0.35, min(0.65, p_up))
