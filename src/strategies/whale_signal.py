"""
whale_signal.py — Whale On-Chain Signal Strategy

Hypothèse : les gros transferts de BNB (whales) précèdent souvent des
mouvements de prix.
  - Gros transfert VERS un exchange  → pression baissière → bet DOWN (NO)
  - Gros transfert DEPUIS un exchange → accumulation     → bet UP  (YES)

Méthode :
  1. Scanner les N derniers blocs BSC via JSON-RPC (eth_getBlockByNumber).
  2. Pour chaque tx dont la valeur dépasse `whale_min_bnb` BNB :
       - Si to_address ∈ KNOWN_EXCHANGES  → net_flow -= value  (outflow vers exchange)
       - Si from_address ∈ KNOWN_EXCHANGES → net_flow += value (inflow depuis exchange)
  3. net_flow > 0 → pression UP ; net_flow < 0 → pression DOWN
  4. Fallback : si la RPC est indisponible, utilise le momentum des prix
     (ratio last_candle_volume / rolling_mean).

Config keys (sous "strategy") :
  - whale_min_bnb         : float  seuil minimum en BNB (défaut 1000)
  - whale_lookback_blocks : int    nombre de blocs à scanner (défaut 15, ≈ 45 s)
  - whale_rpc_url         : str    URL BSC RPC (défaut Ankr public)
  - whale_edge_threshold  : float  override de l'edge_threshold pour cette strat
"""

import logging
import time
from typing import Optional

import requests

from .base import BaseStrategy
from strategy import (
    Signal,
    WindowInfo,
    compute_edge,
    compute_position_size,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Adresses des exchanges majeurs sur BSC (hot wallets connus)
# ─────────────────────────────────────────────────────────────────────────────
KNOWN_EXCHANGES: set[str] = {
    # Binance
    "0x3f5CE5FBFe3E9af3971dD833D26bA9b5C936f0bE".lower(),  # Binance hot wallet 1
    "0xD551234Ae421e3BCBA99A0Da6d736074f22192FF".lower(),  # Binance hot wallet 2
    "0x564286362092D8e7936f0549571a803B203aAceD".lower(),  # Binance hot wallet 3
    "0x0681d8Db095565FE8A346fA0277bFfDE9C0A6c00".lower(),  # Binance BSC
    "0xF977814e90dA44bFA03b6295A0616a897441aceC".lower(),  # Binance BSC hot 2
    "0x8894E0a0c962CB723c1976a4421c95949bE2D4E3".lower(),  # Binance BSC cold
    # OKX / OKEx
    "0x6cC5F688a315f3dC28A7781717a9A798a59fDA7b".lower(),  # OKX hot wallet
    "0x236F9F97e0E62388479bf9E5BA4889e46B0273C3".lower(),  # OKX hot wallet 2
    "0x98EC059Dc3aDFBdd63429454FeB14722a4086587".lower(),  # OKX BSC
    # KuCoin
    "0x2B5634C42055806a59e9107ED44D43c426E58258".lower(),  # KuCoin hot wallet
    "0xa1d8d972560C2f8144AF871Db508F0B0B10a3fBb".lower(),  # KuCoin hot wallet 2
    # Bybit
    "0xf89d7b9c864f589bbF53a82105107622B35EaA40".lower(),  # Bybit hot wallet
    # Huobi / HTX
    "0xaB5C66752a9e8167967685F1450532fB96d5d24f".lower(),  # Huobi hot wallet
    "0x6748f50f686bFbcA6Fe8ad62b22228b87F31ff2B".lower(),  # Huobi hot wallet 2
    # Gate.io
    "0x0D0707963952f2fBA59dD06f2b425ace40b492Fe".lower(),  # Gate.io hot wallet
    # MEXC
    "0x75e89d5979E4f6Fba9F97c104c2F0AFB3F1dcB88".lower(),  # MEXC hot wallet
    # Crypto.com
    "0x6262998Ced04146fA42253a5C0AF90CA02dfd2A3".lower(),  # Crypto.com hot wallet
    # PancakeSwap itself (ne compte pas comme exchange ici, mais utile pour filtrage)
    # "0x18B2A687610328590Bc8F2e5fEdDe3b582A49cdA".lower(),  # PancakeSwap Prediction V2
}

# BSC RPC public par défaut (Binance official dataseed — no API key needed)
DEFAULT_BSC_RPC = "https://bsc-dataseed.binance.org"
WEI_PER_BNB = 10**18


# ─────────────────────────────────────────────────────────────────────────────
# Helpers RPC
# ─────────────────────────────────────────────────────────────────────────────

def _rpc_call(url: str, method: str, params: list, timeout: float = 3.0) -> Optional[dict]:
    """Appel JSON-RPC BSC. Retourne result ou None en cas d'erreur."""
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            logger.debug(f"RPC error ({method}): {data['error']}")
            return None
        return data.get("result")
    except Exception as exc:
        logger.debug(f"RPC call failed ({method}): {exc}")
        return None


def _get_latest_block_number(rpc_url: str) -> Optional[int]:
    result = _rpc_call(rpc_url, "eth_blockNumber", [])
    if result is None:
        return None
    try:
        return int(result, 16)
    except (ValueError, TypeError):
        return None


def _get_block_transactions(rpc_url: str, block_number: int) -> list[dict]:
    """Récupère les transactions d'un bloc BSC."""
    hex_block = hex(block_number)
    result = _rpc_call(rpc_url, "eth_getBlockByNumber", [hex_block, True])
    if result is None:
        return []
    return result.get("transactions", []) or []


# ─────────────────────────────────────────────────────────────────────────────
# Stratégie principale
# ─────────────────────────────────────────────────────────────────────────────

class WhaleSignalStrategy(BaseStrategy):
    """
    🐋 Whale On-Chain Signal Strategy

    Scan les derniers blocs BSC pour détecter de gros transferts de BNB
    vers / depuis les exchanges majeurs et en déduire un biais directionnel.
    """

    FAIR_ODDS_PRICE = 0.50

    def __init__(self, config: dict):
        super().__init__(config)
        cfg = config.get("strategy", {})
        self.whale_min_bnb: float = float(cfg.get("whale_min_bnb", 1000.0))
        self.whale_lookback_blocks: int = int(cfg.get("whale_lookback_blocks", 15))
        self.rpc_url: str = cfg.get("whale_rpc_url", DEFAULT_BSC_RPC)

        # Cache interne : (timestamp, net_flow)
        self._flow_cache: Optional[tuple[float, float]] = None
        self._cache_ttl: float = 30.0  # secondes

    @property
    def name(self) -> str:
        return "🐋 Whale On-Chain Signal"

    # ──────────────────────────────────────────────────────────────────────────
    # Prefetch (sniper pre-load phase)
    # ──────────────────────────────────────────────────────────────────────────

    def prefetch(self, prices: list[float], epoch=None) -> None:
        """Pre-fetch whale on-chain flow for the sniper window."""
        super().prefetch(prices, epoch)
        logger.debug("Whale: prefetching on-chain flow...")
        net_flow = self._compute_net_flow()
        if net_flow is not None:
            self._prefetch_cache["net_flow"] = net_flow
            logger.info(f"Whale prefetch: net_flow={net_flow:+.0f} BNB")
        else:
            logger.warning("Whale prefetch: RPC unavailable — will use price fallback")
            self._prefetch_cache["net_flow_failed"] = True

    # ──────────────────────────────────────────────────────────────────────────
    # On-chain flow computation
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_net_flow(self) -> Optional[float]:
        """
        Scanne les `whale_lookback_blocks` derniers blocs BSC.

        Retourne net_flow en BNB :
          > 0 → flux entrant depuis exchanges (signal UP)
          < 0 → flux sortant vers exchanges   (signal DOWN)
          None → données indisponibles (fallback)
        """
        # Vérification du cache
        if self._flow_cache is not None:
            cached_ts, cached_flow = self._flow_cache
            if time.time() - cached_ts < self._cache_ttl:
                logger.debug(f"Whale: using cached net_flow={cached_flow:.2f} BNB")
                return cached_flow

        latest = _get_latest_block_number(self.rpc_url)
        if latest is None:
            logger.warning("Whale: impossible de récupérer le dernier bloc BSC")
            return None

        net_flow = 0.0
        whale_min_wei = int(self.whale_min_bnb * WEI_PER_BNB)
        n_txs_scanned = 0
        n_whale_txs = 0

        for block_num in range(latest - self.whale_lookback_blocks + 1, latest + 1):
            txs = _get_block_transactions(self.rpc_url, block_num)
            for tx in txs:
                try:
                    value_hex = tx.get("value", "0x0")
                    value_wei = int(value_hex, 16)
                except (ValueError, TypeError):
                    continue

                if value_wei < whale_min_wei:
                    continue

                n_whale_txs += 1
                value_bnb = value_wei / WEI_PER_BNB
                from_addr = (tx.get("from") or "").lower()
                to_addr = (tx.get("to") or "").lower()

                if to_addr in KNOWN_EXCHANGES:
                    # Argent envoyé VERS exchange → bearish
                    net_flow -= value_bnb
                    logger.debug(
                        f"Whale ↓ {value_bnb:.0f} BNB → exchange {to_addr[:10]}... "
                        f"(block {block_num})"
                    )
                elif from_addr in KNOWN_EXCHANGES:
                    # Argent retiré DEPUIS exchange → bullish
                    net_flow += value_bnb
                    logger.debug(
                        f"Whale ↑ {value_bnb:.0f} BNB ← exchange {from_addr[:10]}... "
                        f"(block {block_num})"
                    )

            n_txs_scanned += len(txs)

        logger.info(
            f"Whale scan: {self.whale_lookback_blocks} blocs | "
            f"{n_txs_scanned} txs | {n_whale_txs} whales | "
            f"net_flow={net_flow:+.0f} BNB"
        )

        # Mise en cache
        self._flow_cache = (time.time(), net_flow)
        return net_flow

    # ──────────────────────────────────────────────────────────────────────────
    # Fallback basé sur le momentum des prix
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _fallback_p_up(prices: list[float]) -> float:
        """
        Fallback si le RPC est indisponible.

        Compare la variation des `n_recent` derniers prix à la variation
        globale pour détecter une accélération du momentum.
        Retourne une probabilité UP dans [0.40, 0.60].
        """
        if len(prices) < 10:
            return 0.50

        # Variation globale sur la fenêtre
        global_ret = (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0.0

        # Variation récente (dernier tiers)
        split = max(2, len(prices) * 2 // 3)
        recent = prices[split:]
        recent_ret = (recent[-1] - recent[0]) / recent[0] if recent[0] != 0 else 0.0

        # Momentum net = récent vs global
        momentum = recent_ret - global_ret * 0.5

        # Mapper en probabilité : ±1 % de momentum → ±10 % de prob
        p_up = 0.50 + momentum * 10.0
        return max(0.40, min(0.60, p_up))

    # ──────────────────────────────────────────────────────────────────────────
    # evaluate()
    # ──────────────────────────────────────────────────────────────────────────

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

        # Fenêtre d'entrée : on n'agit que dans les dernières secondes
        if not window.is_entry_window and window.seconds_remaining > self.entry_window_seconds:
            return None

        if len(prices) < 5:
            self.last_skip_reason = f"⏸ Not enough price data ({len(prices)} pts)"
            return None

        # ── Récupération du signal on-chain (use prefetch cache if available) ──
        if "net_flow" in self._prefetch_cache:
            net_flow = self._prefetch_cache["net_flow"]
            logger.debug(f"Whale: using prefetched net_flow={net_flow:+.0f} BNB")
        elif self._prefetch_cache.get("net_flow_failed"):
            net_flow = None  # known failure, skip RPC call
        else:
            net_flow = self._compute_net_flow()
        using_fallback = False

        if net_flow is None:
            # Fallback : signal issu du momentum des prix
            p_up = self._fallback_p_up(prices)
            using_fallback = True
            logger.info(
                f"Whale: fallback momentum P(Up)={p_up:.3f} "
                f"(RPC indisponible)"
            )
        else:
            # Convertir le net_flow en P(Up)
            # Normaliser : 1000 BNB de net flow ≈ ±5 % de biais
            bias = net_flow / 20_000.0  # scale factor empirique
            p_up = max(0.35, min(0.65, 0.50 + bias))
            logger.info(
                f"Whale: net_flow={net_flow:+.0f} BNB → bias={bias:+.4f} → "
                f"P(Up)={p_up:.3f}"
            )

        # ── Edge vs fair odds ─────────────────────────────────────────────────
        effective_yes_price = self.FAIR_ODDS_PRICE
        edge, side = compute_edge(p_up, effective_yes_price)

        source = "fallback" if using_fallback else "on-chain"
        logger.info(
            f"Whale [{source}]: P(Up)={p_up:.3f} | edge={edge:.3f} | "
            f"side={side} | remaining={window.seconds_remaining:.1f}s"
        )

        if edge <= self.edge_threshold:
            self.last_skip_reason = (
                f"⏸ Edge too low ({edge:.3f} ≤ {self.edge_threshold:.2f}) | "
                f"P(Up)={p_up:.3f} | net_flow="
                f"{net_flow:+.0f} BNB" if net_flow is not None else
                f"⏸ Edge too low ({edge:.3f} ≤ {self.edge_threshold:.2f}) | "
                f"P(Up)={p_up:.3f} | fallback"
            )
            return None

        # ── Sizing via Kelly ──────────────────────────────────────────────────
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

        logger.info(f"🎯 Whale signal: {signal}")
        return signal
