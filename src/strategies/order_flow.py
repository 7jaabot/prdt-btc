"""
order_flow.py — Order Flow Imbalance (OFI) Strategy

Hypothesis: Aggressive taker order flow imbalance predicts short-term price
direction. When institutional/smart-money participants need to move size
quickly, they hit the book as takers (market orders). A sustained excess of
taker buys over taker sells precedes upward price moves, and vice-versa.

Method:
  1. Fetch the last N aggTrades from Binance for BNBUSDT.
  2. Separate taker-buy volume (isBuyerMaker=false) from taker-sell volume
     (isBuyerMaker=true).
  3. Weight each trade by its notional size (price × qty) to up-weight large
     "smart money" prints relative to retail noise.
  4. Compute:
       OFI = (weighted_buy - weighted_sell) / (weighted_buy + weighted_sell)
     OFI ∈ [-1, 1].  Positive → buy pressure, Negative → sell pressure.
  5. Signal UP if OFI > threshold, DOWN if OFI < -threshold.
  6. Edge = abs(OFI) - threshold.

References:
  - Cont, Kukanov & Stoikov (2014): "The Price Impact of Order Book Events"
  - Gould & Bonart (2016): "Queue Imbalance as a One-Tick-Ahead Price
    Predictor in a Limit Order Book" — shows aggressive-order imbalance as
    a strong short-horizon signal.
  - Binance aggTrades field `m` (isBuyerMaker): false = taker buy,
    true = taker sell.
"""

import logging
import time
from typing import Optional

import urllib.request
import json

from .base import BaseStrategy
from strategy import (
    Signal, WindowInfo,
    compute_edge, compute_position_size,
)

logger = logging.getLogger(__name__)

BINANCE_AGG_TRADES_URL = (
    "https://api.binance.com/api/v3/aggTrades?symbol=BNBUSDT&limit={limit}"
)


class OrderFlowStrategy(BaseStrategy):
    """
    Order Flow Imbalance strategy for PancakeSwap Prediction V2.

    Fetches recent Binance aggTrades and computes a weighted taker-buy/sell
    imbalance ratio as a directional signal.

    Config keys (under 'strategy'):
      ofi_threshold      float  Minimum |OFI| to generate a signal (default 0.15)
      ofi_min_volume     float  Minimum total weighted BNB volume required (default 100)
      ofi_lookback_trades int   Number of aggTrades to fetch (default 500)
    """

    FAIR_ODDS_PRICE = 0.50

    def __init__(self, config: dict):
        super().__init__(config)
        cfg = config.get("strategy", {})
        self.ofi_threshold: float = cfg.get("ofi_threshold", 0.15)
        self.ofi_min_volume: float = cfg.get("ofi_min_volume", 100.0)
        self.ofi_lookback_trades: int = int(cfg.get("ofi_lookback_trades", 500))

    @property
    def name(self) -> str:
        return "🏦 Order Flow Imbalance"

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _fetch_agg_trades(self) -> list[dict]:
        """
        Fetch aggTrades from Binance REST API.

        Returns list of dicts with at least:
          p  (str) price
          q  (str) quantity
          m  (bool) isBuyerMaker
        Returns empty list on any error.
        """
        url = BINANCE_AGG_TRADES_URL.format(limit=self.ofi_lookback_trades)
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "bnb-updown/1.0"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                raw = resp.read().decode("utf-8")
            return json.loads(raw)
        except Exception as exc:
            logger.warning(f"OFI: aggTrades fetch failed — {exc}")
            return []

    def _compute_ofi(self, trades: list[dict]) -> tuple[float, float, float]:
        """
        Compute weighted Order Flow Imbalance from a list of aggTrades.

        Each trade is weighted by its notional value (price × qty) to give
        larger prints more influence, approximating smart-money flow.

        Args:
            trades: List of aggTrade dicts from Binance.

        Returns:
            (ofi, total_weighted_volume, buy_ratio)
            ofi ∈ [-1, 1]; total_weighted_volume in USDT notional;
            buy_ratio ∈ [0, 1] (fraction of volume that is taker buy).
        """
        weighted_buy = 0.0
        weighted_sell = 0.0

        for trade in trades:
            try:
                price = float(trade["p"])
                qty = float(trade["q"])
                notional = price * qty          # USDT notional → proxy for size
                is_buyer_maker: bool = trade["m"]

                if is_buyer_maker:
                    # Buyer is the maker → seller hit the book → TAKER SELL
                    weighted_sell += notional
                else:
                    # Buyer is the taker → buyer hit the book → TAKER BUY
                    weighted_buy += notional
            except (KeyError, ValueError, TypeError):
                continue

        total = weighted_buy + weighted_sell
        if total == 0.0:
            return 0.0, 0.0, 0.5

        ofi = (weighted_buy - weighted_sell) / total
        buy_ratio = weighted_buy / total
        return ofi, total, buy_ratio

    # ─────────────────────────────────────────────────────────────────────────
    # BaseStrategy interface
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

        # Only act within the entry window
        if not window.is_entry_window and window.seconds_remaining > self.entry_window_seconds:
            return None

        # ── Fetch order flow data ───────────────────────────────────────────
        trades = self._fetch_agg_trades()

        if not trades:
            self.last_skip_reason = "⏸ OFI: no aggTrades data available"
            logger.warning("OFI: skipping — no trades fetched")
            return None

        ofi, total_volume, buy_ratio = self._compute_ofi(trades)

        # Convert USDT notional volume to approximate BNB for min_volume check
        current_price = prices[-1] if prices else None
        if current_price and current_price > 0:
            total_bnb = total_volume / current_price
        else:
            total_bnb = 0.0

        logger.info(
            f"OFI: {len(trades)} trades | OFI={ofi:+.4f} | "
            f"buy_ratio={buy_ratio:.3f} | total_notional={total_volume:.1f} USDT "
            f"(≈{total_bnb:.1f} BNB) | threshold={self.ofi_threshold}"
        )

        # ── Volume guard ────────────────────────────────────────────────────
        if total_bnb < self.ofi_min_volume:
            self.last_skip_reason = (
                f"⏸ OFI: volume too low "
                f"({total_bnb:.1f} BNB < {self.ofi_min_volume:.0f} min)"
            )
            logger.info(f"OFI: skipping — insufficient volume ({total_bnb:.1f} BNB)")
            return None

        # ── Signal threshold ────────────────────────────────────────────────
        abs_ofi = abs(ofi)
        if abs_ofi <= self.ofi_threshold:
            self.last_skip_reason = (
                f"⏸ OFI too weak ({ofi:+.4f} | "
                f"|OFI|={abs_ofi:.4f} ≤ {self.ofi_threshold})"
            )
            return None

        # ── Direction and p_up ──────────────────────────────────────────────
        # Map OFI ∈ [-1, 1] to P(Up) ∈ [0.35, 0.65].
        # P(Up) = 0.5 + OFI * 0.30  (max shift ±0.30 at |OFI|=1)
        # Old factor 0.15 produced edges too low to ever pass edge_threshold
        p_up_raw = 0.5 + ofi * 0.30
        p_up = max(0.35, min(0.65, p_up_raw))

        # ── Edge vs fair odds ───────────────────────────────────────────────
        # Always trade against fair odds (0.50) — pool price may be distorted
        effective_yes_price = self.FAIR_ODDS_PRICE
        edge, side = compute_edge(p_up, effective_yes_price)

        # Override edge with OFI-derived edge for cleaner signal
        ofi_edge = abs_ofi - self.ofi_threshold  # raw OFI edge above threshold

        logger.info(
            f"OFI signal: side={side} | p_up={p_up:.4f} | "
            f"ofi_edge={ofi_edge:.4f} | edge={edge:.4f} | "
            f"threshold={self.edge_threshold} | remaining={window.seconds_remaining:.1f}s"
        )

        # Apply the base edge_threshold as a second guard
        if edge <= self.edge_threshold:
            self.last_skip_reason = (
                f"⏸ OFI: edge too low ({edge:.4f} ≤ {self.edge_threshold}) | "
                f"OFI={ofi:+.4f}"
            )
            return None

        # ── Position size ───────────────────────────────────────────────────
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
            self.last_skip_reason = f"⏸ OFI: position size is 0 (Kelly={raw_k:.4f})"
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

        logger.info(f"🎯 OFI Signal: {signal} | OFI={ofi:+.4f} | buy_ratio={buy_ratio:.3f}")
        return signal
