"""
base.py — Abstract base class for all trading strategies.

All strategies share a common interface: evaluate() returns a Signal or None.
Utility functions (Kelly, edge, position sizing) live in strategy.py.
"""

import sys
import os

# Ensure parent src/ is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from abc import ABC, abstractmethod
from typing import Optional

from strategy import Signal, WindowInfo


class BaseStrategy(ABC):
    """Interface commune pour toutes les stratégies de trading."""

    def __init__(self, config: dict):
        cfg = config.get("strategy", {})
        self.edge_threshold = cfg.get("edge_threshold", 0.10)
        self.position_size_usdc = cfg.get("position_size_usdc", 10.0)
        self.entry_window_seconds = cfg.get("entry_window_seconds", 25)
        self.bankroll = cfg.get("starting_bankroll_usdc", 1000.0)
        self.last_skip_reason: Optional[str] = None

        # Position sizing guards
        self.min_position_usdc = cfg.get("min_position_usdc", 5.0)
        self.max_bankroll_pct = cfg.get("max_bankroll_pct", 1.0)   # default: no cap
        self.max_pool_pct = cfg.get("max_pool_pct", 1.0)           # default: no cap

        pcfg = config.get("pancake", {})
        self.min_pool_bnb = pcfg.get("min_pool_bnb", 0.4)
        self.max_bet_share_of_side = pcfg.get("max_bet_share_of_side", 0.25)
        self.pancake_fee = pcfg.get("fee", 0.03)

        # Prefetch cache: keyed by epoch, cleared on new epoch
        self._prefetch_epoch: Optional[int] = None
        self._prefetch_cache: dict = {}

    def update_bankroll(self, new_bankroll: float):
        """Update the current bankroll after PnL changes."""
        self.bankroll = new_bankroll
        self.last_skip_reason = None

    def prefetch(self, prices: list[float], epoch: Optional[int] = None) -> None:
        """
        Pre-fetch slow external data (API calls, RPC calls) before the sniper window.

        Called during Phase 1 (T-15s → T-8s before lock) to warm up caches.
        Default implementation invalidates the cache for the new epoch and does nothing.
        Override in strategies that make external API calls.

        Args:
            prices: Current price series (may be used for context/filtering).
            epoch: Current epoch number (used for cache invalidation).
        """
        if epoch is not None and epoch != self._prefetch_epoch:
            self._prefetch_epoch = epoch
            self._prefetch_cache.clear()

    @abstractmethod
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
        """Evaluate and return a Signal or None."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name for display."""
        ...
