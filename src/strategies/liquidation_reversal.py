"""
liquidation_reversal.py — Liquidation Reversal Strategy

Hypothesis: Large forced liquidations create abrupt price moves that temporarily
exhaust the dominant side.
  - If many LONG positions just got liquidated (price dropped hard), selling
    pressure is now depleted → mean-reversion UP is more likely.
  - If many SHORT positions just got liquidated (price spiked hard), buying
    pressure is exhausted → mean-reversion DOWN is more likely.

Data source: Binance Futures WebSocket stream (no auth required)
  wss://fstream.binance.com/ws/bnbusdt@forceOrder

Side convention in Binance liquidation data:
  - side == "SELL"  → the system sold (liquidated) a LONG position
  - side == "BUY"   → the system bought back (liquidated) a SHORT position

Edge = z-score of recent liq imbalance vs rolling baseline.
"""

import asyncio
import json
import logging
import threading
import time
from collections import deque
from typing import Optional

import numpy as np

try:
    import websockets
except ImportError:
    websockets = None

from .base import BaseStrategy
from strategy import Signal, WindowInfo, compute_edge, compute_position_size

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

WS_URL = "wss://fstream.binance.com/ws/bnbusdt@forceOrder"

# Max age of liquidation events to keep in buffer (seconds)
_BUFFER_MAX_AGE_SECONDS = 3600  # 1 hour

# How many historical buckets to keep for z-score baseline
_HISTORY_MAX_BUCKETS = 12

# Clamp for P(Up) from liquidation signal
_P_UP_MIN = 0.35
_P_UP_MAX = 0.65


# ─────────────────────────────────────────────────────────────────────────────
# Liquidation event
# ─────────────────────────────────────────────────────────────────────────────

class LiqEvent:
    """A single liquidation event from the WebSocket stream."""
    __slots__ = ("timestamp", "side", "qty", "price", "vol_usdt")

    def __init__(self, timestamp: float, side: str, qty: float, price: float):
        self.timestamp = timestamp
        self.side = side  # "SELL" (long liq) or "BUY" (short liq)
        self.qty = qty
        self.price = price
        self.vol_usdt = qty * price


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket listener (background thread)
# ─────────────────────────────────────────────────────────────────────────────

class LiquidationListener:
    """
    Background WebSocket listener for BNBUSDT forced liquidation events.
    
    Runs in a daemon thread, accumulates events in a thread-safe deque.
    Auto-reconnects on disconnect.
    """

    def __init__(self):
        self._events: deque[LiqEvent] = deque()
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._connected = False
        self._total_received = 0

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def total_received(self) -> int:
        return self._total_received

    def start(self):
        """Start the WebSocket listener in a background thread."""
        if self._running:
            return
        if websockets is None:
            logger.error("websockets package not installed — liquidation listener disabled")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_thread, daemon=True, name="liq-ws")
        self._thread.start()
        logger.info(f"⚡ Liquidation WebSocket listener started → {WS_URL}")

    def stop(self):
        """Stop the listener."""
        self._running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        logger.info("⚡ Liquidation WebSocket listener stopped")

    def get_events(self, lookback_seconds: float) -> list[LiqEvent]:
        """Get liquidation events within the lookback window (thread-safe)."""
        cutoff = time.time() - lookback_seconds
        with self._lock:
            return [e for e in self._events if e.timestamp >= cutoff]

    def _prune_old(self):
        """Remove events older than max buffer age."""
        cutoff = time.time() - _BUFFER_MAX_AGE_SECONDS
        with self._lock:
            while self._events and self._events[0].timestamp < cutoff:
                self._events.popleft()

    def _run_thread(self):
        """Entry point for the background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._ws_loop())
        except Exception as e:
            logger.error(f"⚡ Liquidation listener thread crashed: {e}")
        finally:
            self._loop.close()

    async def _ws_loop(self):
        """Main WebSocket loop with auto-reconnect."""
        while self._running:
            try:
                async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=10) as ws:
                    self._connected = True
                    logger.info("⚡ Liquidation WebSocket connected")

                    while self._running:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
                            self._handle_message(raw)
                        except asyncio.TimeoutError:
                            # No liquidation in 30s — normal during quiet markets
                            continue
                        except websockets.ConnectionClosed:
                            logger.warning("⚡ Liquidation WebSocket disconnected")
                            break

            except Exception as e:
                logger.warning(f"⚡ Liquidation WebSocket error: {e}")

            self._connected = False
            if self._running:
                logger.info("⚡ Liquidation WebSocket reconnecting in 5s...")
                await asyncio.sleep(5)

    def _handle_message(self, raw: str):
        """Parse a forceOrder WebSocket message and store it."""
        try:
            msg = json.loads(raw)
            # Format: {"e": "forceOrder", "E": <ts_ms>, "o": {order details}}
            order = msg.get("o", {})
            side = order.get("S", "")        # "SELL" or "BUY"
            qty = float(order.get("q", 0))   # Original quantity
            price = float(order.get("p", 0)) # Price
            ts_ms = order.get("T", msg.get("E", int(time.time() * 1000)))

            if side not in ("BUY", "SELL") or qty <= 0 or price <= 0:
                return

            event = LiqEvent(
                timestamp=ts_ms / 1000.0,
                side=side,
                qty=qty,
                price=price,
            )

            with self._lock:
                self._events.append(event)
            self._total_received += 1

            # Periodic pruning (every 100 events)
            if self._total_received % 100 == 0:
                self._prune_old()

            logger.debug(
                f"⚡ Liq event: {side} {qty:.3f} BNB @ {price:.2f} "
                f"= ${event.vol_usdt:.0f}"
            )

        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            logger.warning(f"⚡ Failed to parse liq event: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Singleton listener (shared across strategy instances)
# ─────────────────────────────────────────────────────────────────────────────

_global_listener: Optional[LiquidationListener] = None
_listener_lock = threading.Lock()


def _get_listener() -> LiquidationListener:
    """Get or create the global liquidation listener."""
    global _global_listener
    with _listener_lock:
        if _global_listener is None:
            _global_listener = LiquidationListener()
            _global_listener.start()
        return _global_listener


# ─────────────────────────────────────────────────────────────────────────────
# Strategy
# ─────────────────────────────────────────────────────────────────────────────

class LiquidationReversalStrategy(BaseStrategy):
    """
    ⚡ Liquidation Reversal Strategy for BNB Up/Down 5mn.

    Monitors recent forced liquidations on BNB perpetual futures via WebSocket.
    When a significant imbalance (z-score) is detected, bets on mean-reversion:
      - Large LONG liq volume → UP bet
      - Large SHORT liq volume → DOWN bet

    Config keys (under "liquidation_reversal" in strategy config):
      liquidation_min_volume_usdt (float): minimum net liq volume to trigger (default 50_000)
      liquidation_lookback_minutes (float): look-back window in minutes (default 5)
    """

    @property
    def name(self) -> str:
        return "⚡ Liquidation Reversal"

    def __init__(self, config: dict):
        super().__init__(config)

        # Strategy-specific config
        liq_cfg = config.get("liquidation_reversal", {})
        self.min_volume_usdt: float = liq_cfg.get("liquidation_min_volume_usdt", 50_000.0)
        self.lookback_minutes: float = liq_cfg.get("liquidation_lookback_minutes", 5.0)

        # Rolling history of (long_vol_usdt, short_vol_usdt) per window
        self._history: list[tuple[float, float]] = []

        # Use fair-odds or pool-derived price
        cfg_s = config.get("strategy", {})
        self.use_fair_odds: bool = cfg_s.get("use_fair_odds", True)
        self.FAIR_ODDS_PRICE: float = 0.50

        # Listener initialized lazily on first evaluate() call
        self._listener: Optional[LiquidationListener] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Volume aggregation
    # ─────────────────────────────────────────────────────────────────────────

    def _ensure_listener(self):
        """Lazily start the WebSocket listener on first use."""
        if self._listener is None:
            self._listener = _get_listener()

    def _aggregate_volumes(self, lookback_minutes: float) -> tuple[float, float]:
        """
        Sum liquidation volumes by side within the lookback window.

        Returns:
            (long_vol_usdt, short_vol_usdt)
              long_vol_usdt : USDT value of liquidated LONG positions (side == "SELL")
              short_vol_usdt: USDT value of liquidated SHORT positions (side == "BUY")
        """
        events = self._listener.get_events(lookback_minutes * 60)
        long_vol = 0.0
        short_vol = 0.0

        for e in events:
            if e.side == "SELL":
                long_vol += e.vol_usdt
            elif e.side == "BUY":
                short_vol += e.vol_usdt

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

        if sigma < 1.0:
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
        """
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

        # Ensure WebSocket listener is running
        self._ensure_listener()

        # Check WebSocket status
        if not self._listener.connected and self._listener.total_received == 0:
            self.last_skip_reason = "⏸ Liquidation WebSocket not yet connected"
            return None

        # ── 1. Aggregate volumes ──────────────────────────────────────────────
        long_vol, short_vol = self._aggregate_volumes(self.lookback_minutes)
        net_imbalance = long_vol - short_vol
        total_vol = long_vol + short_vol

        logger.info(
            f"⚡ Liq volumes ({self.lookback_minutes:.0f}m): "
            f"LONG={long_vol/1000:.1f}k SHORT={short_vol/1000:.1f}k "
            f"TOTAL={total_vol/1000:.1f}k USDT "
            f"(ws_events={self._listener.total_received})"
        )

        # ── 2. Minimum volume threshold ───────────────────────────────────────
        if total_vol < self.min_volume_usdt:
            self.last_skip_reason = (
                f"⏸ Total liq volume too low "
                f"({total_vol/1000:.1f}k < {self.min_volume_usdt/1000:.0f}k USDT)"
            )
            self._update_history(long_vol, short_vol)
            return None

        # ── 3. Z-score computation ────────────────────────────────────────────
        zscore = self._compute_zscore(net_imbalance)
        self._update_history(long_vol, short_vol)

        logger.info(
            f"⚡ Liq imbalance: net={net_imbalance/1000:.1f}k USDT "
            f"z={zscore:+.2f} (history={len(self._history)} buckets)"
        )

        # NOTE: zscore < 1.0 pre-filter removed — zscore is used in _estimate_p_up
        # to compute p_up, which feeds the edge. A low z-score → p_up ≈ 0.50 →
        # near-zero edge → filtered naturally by edge_threshold.
        # Safety: zscore == 0 → p_up = 0.50 → edge = 0 → Kelly = 0, handled below.

        # ── 4. P(Up) from liquidation signal ─────────────────────────────────
        p_up = self._estimate_p_up(zscore)

        # ── 5. Compare vs pool/fair price ────────────────────────────────────
        effective_yes_price = (
            self.FAIR_ODDS_PRICE if self.use_fair_odds else yes_price
        )
        edge, side = compute_edge(p_up, effective_yes_price)

        logger.info(
            f"⚡ Liq signal: z={zscore:+.2f} P(Up)={p_up:.3f} | "
            f"price={effective_yes_price:.3f} edge={edge:.3f} side={side}"
        )

        # ── 6. Edge threshold ─────────────────────────────────────────────────
        if edge <= self.edge_threshold:
            self.last_skip_reason = (
                f"⏸ Edge too low ({edge:.3f} ≤ {self.edge_threshold:.2f}) | "
                f"P(Up)={p_up:.2f} z={zscore:+.2f}"
            )
            return None

        # ── 7. Position sizing ────────────────────────────────────────────────
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
            self.last_skip_reason = "⏸ Flat sizing → 0 position (below min or cap)"
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

        # ── 8. Build signal ───────────────────────────────────────────────────
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
            f"⚡ Liquidation Reversal signal: {signal} | "
            f"liq_long={long_vol/1000:.1f}k liq_short={short_vol/1000:.1f}k z={zscore:+.2f}"
        )
        return signal
