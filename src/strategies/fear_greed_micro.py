"""
fear_greed_micro.py — Fear & Greed Micro Strategy

Hypothèse : Le spread bid/ask est un proxy du sentiment de marché à très court terme.
- Spread large + volatilité élevée = Fear (hedging intensif) → rebond attendu → UP
- Spread très serré + volume spike = Greed (FOMO) → correction attendue → DOWN

Le "Micro Fear Index" (MFI) combine ces deux signaux pour détecter des états extrêmes
qui ont tendance à se corriger dans les 5 minutes suivantes.

Sources de données Binance :
- bookTicker : meilleur bid/ask en temps réel
- 24hr ticker : volatilité (priceChangePercent) + volume
"""

import logging
import time
from typing import Optional

import requests

from .base import BaseStrategy
from strategy import (
    Signal, WindowInfo,
    compute_edge, compute_position_size,
)

logger = logging.getLogger(__name__)

BINANCE_BOOK_URL = "https://api.binance.com/api/v3/ticker/bookTicker?symbol=BNBUSDT"
BINANCE_24HR_URL = "https://api.binance.com/api/v3/ticker/24hr?symbol=BNBUSDT"

# Empirical baseline for MFI (calibré sur range BNB/USDT 24h)
# Typical 24h range ~1-2% → MFI ~0.01-0.02, stressed ~0.03-0.05
_MFI_BASELINE_DEFAULT = 0.015        # ~1.5% daily range in calm market
_MFI_BASELINE_STD_DEFAULT = 0.008    # std empirique observée

# HTTP timeout pour les appels Binance
_REQUEST_TIMEOUT = 3.0  # secondes


class FearGreedMicroStrategy(BaseStrategy):
    """
    Fear & Greed Micro Strategy for PancakeSwap Prediction V2 (BNB/USD).

    Signal logic :
    - Fetch bid/ask spread instantané (bookTicker)
    - Fetch volatilité 24h et volume (24hr ticker)
    - MFI = spread_pct * volatility_factor
    - MFI > fear_threshold  → Fear extrême → bet UP  (mean reversion haussière)
    - MFI < greed_threshold + volume_spike → Greed extrême → bet DOWN (correction)
    - Edge = abs(MFI - baseline) / baseline_std  (normalisé, capped à [0, 1])
    """

    @property
    def name(self) -> str:
        return "😱 Fear & Greed Micro"

    def __init__(self, config: dict):
        super().__init__(config)
        cfg = config.get("strategy", {})

        # Seuils Fear & Greed
        # Fear = high MFI (wide range, panic), Greed = low MFI (tight range, complacency)
        self.fear_threshold: float = cfg.get("fgm_fear_threshold", 0.025)
        self.greed_threshold: float = cfg.get("fgm_greed_threshold", 0.008)

        # Volume spike : ratio volume_now/volume_avg_24h > ce seuil = spike
        self.volume_spike_factor: float = cfg.get("fgm_volume_spike_factor", 1.5)

        # Baselines pour normalisation du edge (peuvent être calibrées via config)
        self.mfi_baseline: float = cfg.get("fgm_mfi_baseline", _MFI_BASELINE_DEFAULT)
        self.mfi_baseline_std: float = cfg.get("fgm_mfi_baseline_std", _MFI_BASELINE_STD_DEFAULT)

        # Probabilité UP/DOWN utilisée en cas de signal Fear/Greed
        # Fear → UP avec p_up > 0.5 ; Greed → DOWN avec p_up < 0.5
        self.fear_p_up: float = cfg.get("fgm_fear_p_up", 0.60)
        self.greed_p_up: float = cfg.get("fgm_greed_p_up", 0.40)

        # Dernier état fetché (pour debug/log)
        self._last_mfi: Optional[float] = None
        self._last_spread_pct: Optional[float] = None
        self._last_vol_factor: Optional[float] = None
        self._last_volume_spike: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # Data fetching
    # ─────────────────────────────────────────────────────────────────────────

    def _fetch_book_ticker(self) -> Optional[dict]:
        """Fetch best bid/ask from Binance REST API."""
        try:
            resp = requests.get(BINANCE_BOOK_URL, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return {
                "bid": float(data["bidPrice"]),
                "ask": float(data["askPrice"]),
                "bid_qty": float(data["bidQty"]),
                "ask_qty": float(data["askQty"]),
            }
        except Exception as exc:
            logger.warning(f"[FGM] bookTicker fetch failed: {exc}")
            return None

    def _fetch_24hr_ticker(self) -> Optional[dict]:
        """Fetch 24h stats from Binance REST API."""
        try:
            resp = requests.get(BINANCE_24HR_URL, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return {
                "price_change_pct": abs(float(data["priceChangePercent"])),  # abs → volatilité
                "volume": float(data["volume"]),
                "weighted_avg_price": float(data["weightedAvgPrice"]),
                "high": float(data["highPrice"]),
                "low": float(data["lowPrice"]),
                "count": int(data["count"]),
            }
        except Exception as exc:
            logger.warning(f"[FGM] 24hr ticker fetch failed: {exc}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # MFI computation
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_mfi(
        self,
        book: dict,
        stats: dict,
    ) -> tuple[float, float, float, bool]:
        """
        Compute the Micro Fear Index.

        Returns:
            (mfi, spread_pct, volatility_factor, volume_spike)

        Formula (v2 — volatility-based, not spread-based):
            BNB spot spread is always 1 tick (~0.001%) on Binance — useless as a signal.
            Instead we use the 24h high-low range as a volatility proxy:
                range_pct = (high - low) / weighted_avg_price
            Combined with price change momentum:
                vol_factor = abs(priceChangePercent) / 100
            MFI = range_pct * (1 + vol_factor)

            High MFI → wide range + strong momentum = Fear (panic selling/buying)
            Low MFI  → tight range + low momentum = Greed/complacency
        """
        bid = book["bid"]
        ask = book["ask"]
        mid_price = (bid + ask) / 2.0

        spread_pct = (ask - bid) / mid_price if mid_price > 0 else 0.0

        # Volatility factor from 24h price change
        vol_pct = stats["price_change_pct"]  # already abs
        volatility_factor = vol_pct / 100.0

        # Range-based MFI: (high - low) / avg_price gives true intraday volatility
        high = stats.get("high", 0)
        low = stats.get("low", 0)
        avg_price = stats.get("weighted_avg_price", mid_price)
        range_pct = (high - low) / avg_price if avg_price > 0 else 0.0

        # MFI = range amplified by directional momentum
        # Typical: range=1% + vol=0.5% → MFI = 0.01 * 1.005 ≈ 0.01
        # Stressed: range=4% + vol=3% → MFI = 0.04 * 1.03 ≈ 0.041
        mfi = range_pct * (1.0 + volatility_factor)

        # Volume spike detection
        volume_spike = stats["count"] > (800_000 * self.volume_spike_factor)

        return mfi, spread_pct, volatility_factor, volume_spike

    def _compute_edge_from_mfi(self, mfi: float) -> float:
        """
        Normalise le MFI en edge [0, 1].

        Edge = abs(MFI - baseline) / baseline_std
        Capped à [0, 1] pour rester dans le domaine valid d'un edge.
        """
        if self.mfi_baseline_std <= 0:
            return 0.0
        raw_edge = abs(mfi - self.mfi_baseline) / self.mfi_baseline_std
        return min(1.0, max(0.0, raw_edge))

    # ─────────────────────────────────────────────────────────────────────────
    # Main evaluate() — called by the bot engine
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

        # Seulement dans la fenêtre d'entrée
        if not window.is_entry_window and window.seconds_remaining > self.entry_window_seconds:
            return None

        # Fetch données de marché Binance
        book = self._fetch_book_ticker()
        stats = self._fetch_24hr_ticker()

        if book is None or stats is None:
            self.last_skip_reason = "⏸ [FGM] Binance API unavailable — skip"
            logger.warning("[FGM] Cannot evaluate: Binance data unavailable")
            return None

        # Calcul MFI
        mfi, spread_pct, vol_factor, volume_spike = self._compute_mfi(book, stats)

        # Persistance pour debug
        self._last_mfi = mfi
        self._last_spread_pct = spread_pct
        self._last_vol_factor = vol_factor
        self._last_volume_spike = volume_spike

        logger.info(
            f"[FGM] bid={book['bid']:.4f} ask={book['ask']:.4f} "
            f"spread_pct={spread_pct:.6f} vol_factor={vol_factor:.4f} "
            f"MFI={mfi:.6f} vol_spike={volume_spike} "
            f"fear_thr={self.fear_threshold} greed_thr={self.greed_threshold}"
        )

        # Détermination du signal
        side: Optional[str] = None
        p_up: float = 0.5

        if mfi > self.fear_threshold:
            # Fear extrême → mean reversion haussière → UP
            side = "YES"
            p_up = self.fear_p_up
            logger.info(f"[FGM] 😱 FEAR signal | MFI={mfi:.6f} > {self.fear_threshold} → UP")

        elif mfi < self.greed_threshold and volume_spike:
            # Greed extrême (spread serré + FOMO volume) → correction → DOWN
            side = "NO"
            p_up = self.greed_p_up
            logger.info(
                f"[FGM] 🤑 GREED signal | MFI={mfi:.6f} < {self.greed_threshold} "
                f"+ vol_spike={volume_spike} → DOWN"
            )

        if side is None:
            self.last_skip_reason = (
                f"⏸ [FGM] Neutral | MFI={mfi:.6f} "
                f"(fear>{self.fear_threshold} or greed<{self.greed_threshold}+spike needed)"
            )
            return None

        # Calcul edge normalisé
        edge = self._compute_edge_from_mfi(mfi)

        if edge <= self.edge_threshold:
            self.last_skip_reason = (
                f"⏸ [FGM] Edge too low ({edge:.3f} ≤ {self.edge_threshold:.2f}) | "
                f"MFI={mfi:.6f}"
            )
            return None

        # Sizing via Kelly (on utilise 0.5 comme yes_price de référence — fair odds)
        effective_yes_price = 0.50
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
            self.last_skip_reason = f"⏸ [FGM] Position size = 0 (Kelly={raw_k:.3f})"
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

        logger.info(f"[FGM] 🎯 Signal: {signal} | MFI={mfi:.6f} spread={spread_pct:.6f}")
        return signal
