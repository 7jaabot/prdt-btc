"""
pancake.py — PancakeSwap Prediction V2 client (BNB/USD on BSC)

Pool binary market: bull vs bear, 5-minute rounds, 3% fee.
Winners split losers' pool proportionally.
Settlement via Chainlink oracle (no order book).

Contract: 0x18B2A687610328590Bc8F2e5fEdDe3b582A49cdA (BSC, BNB/USD)
Docs: https://developer.pancakeswap.finance/contracts/prediction

Note on ABI decoding:
  The rounds() struct has a non-standard bool encoding in the last slot.
  We bypass web3's ABI decoder for rounds() and read slots directly.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("PancakeClient")

BSC_RPC_URLS = [
    "https://bsc-dataseed.binance.org/",
    "https://bsc-dataseed1.defibit.io/",
    "https://bsc-dataseed2.defibit.io/",
    "https://1rpc.io/bnb",
]

# PancakeSwap Prediction V2 — BNB/USD (high volume)
PANCAKE_BNB_CONTRACT = "0x18B2A687610328590Bc8F2e5fEdDe3b582A49cdA"
PANCAKE_FEE = 0.03


@dataclass
class PancakeRound:
    epoch: int
    start_ts: int
    lock_ts: int
    close_ts: int
    total_bnb: float
    bull_bnb: float
    bear_bnb: float
    bull_ratio: float
    bear_ratio: float
    bull_payout: float
    bear_payout: float
    lock_price: Optional[float]   # BNB/USD at lock (Chainlink)
    oracle_called: bool
    is_mock: bool = False

    @property
    def seconds_remaining(self) -> float:
        return max(0.0, self.close_ts - time.time())

    @property
    def seconds_to_lock(self) -> float:
        return max(0.0, self.lock_ts - time.time())

    @property
    def is_betting_open(self) -> bool:
        return time.time() < self.lock_ts

    @property
    def yes_price_equiv(self) -> float:
        """
        Polymarket-equivalent YES price for the BULL side.
        = break-even probability to bet bull = bull_bnb / (total_bnb * (1 - fee))
        < 0.5 → bull is underbet → +EV to bet bull
        > 0.5 → bear is underbet → +EV to bet bear
        """
        if self.total_bnb <= 0:
            return 0.5
        return min(max(self.bull_ratio / (1.0 - PANCAKE_FEE), 0.01), 0.99)


def _mock_round() -> PancakeRound:
    import random
    now = int(time.time())
    start = (now // 300) * 300
    lock  = start + 270
    close = start + 600   # PancakeSwap: lock phase + close phase = 2×5min

    bull_pct = random.uniform(0.35, 0.65)
    bear_pct = 1.0 - bull_pct
    total = random.uniform(1, 20)   # BNB
    bull = total * bull_pct
    bear = total * bear_pct

    return PancakeRound(
        epoch=now // 300,
        start_ts=start, lock_ts=lock, close_ts=close,
        total_bnb=total, bull_bnb=bull, bear_bnb=bear,
        bull_ratio=bull_pct, bear_ratio=bear_pct,
        bull_payout=(total * (1 - PANCAKE_FEE)) / bull if bull > 0 else 2.0,
        bear_payout=(total * (1 - PANCAKE_FEE)) / bear if bear > 0 else 2.0,
        lock_price=None, oracle_called=False, is_mock=True,
    )


def _decode_round_raw(raw: bytes, epoch: int) -> PancakeRound:
    """
    Decode rounds() return data by reading 32-byte slots directly.
    Bypasses web3's strict bool padding validation.
    """
    def slot(i) -> int:
        return int.from_bytes(raw[i*32:(i+1)*32], "big")

    def slot_int256(i) -> int:
        v = slot(i)
        return v if v < 2**255 else v - 2**256

    total = slot(8) / 1e18
    bull  = slot(9) / 1e18
    bear  = slot(10) / 1e18

    if total > 0:
        bull_ratio = bull / total
        bear_ratio = bear / total
        bull_payout = (total * (1 - PANCAKE_FEE)) / bull if bull > 0 else 0.0
        bear_payout = (total * (1 - PANCAKE_FEE)) / bear if bear > 0 else 0.0
    else:
        bull_ratio = bear_ratio = 0.5
        bull_payout = bear_payout = 2.0 * (1 - PANCAKE_FEE)

    lock_p_raw  = slot_int256(4)
    close_p_raw = slot_int256(5)
    lock_price  = lock_p_raw  / 1e8 if lock_p_raw  != 0 else None
    oracle_called = bool(slot(13) & 0xFF)

    return PancakeRound(
        epoch=slot(0),
        start_ts=slot(1), lock_ts=slot(2), close_ts=slot(3),
        total_bnb=total, bull_bnb=bull, bear_bnb=bear,
        bull_ratio=bull_ratio, bear_ratio=bear_ratio,
        bull_payout=bull_payout, bear_payout=bear_payout,
        lock_price=lock_price, oracle_called=oracle_called,
        is_mock=False,
    )


class PancakeClient:
    """
    Reads PancakeSwap Prediction V2 (BNB/USD) from BSC via raw JSON-RPC.
    Falls back to mock data if chain unreachable.
    """

    def __init__(
        self,
        rpc_url: Optional[str] = None,
        contract_address: str = PANCAKE_BNB_CONTRACT,
        use_mock_on_failure: bool = True,
        timeout: int = 8,
    ):
        self.rpc_url = rpc_url
        self.contract_address = contract_address
        self.use_mock_on_failure = use_mock_on_failure
        self.timeout = timeout
        self._w3 = None
        self._connected = False
        self._sel_epoch = None
        self._sel_rounds = None
        self._init_web3()

    def _init_web3(self):
        try:
            from web3 import Web3
            from web3.middleware import ExtraDataToPOAMiddleware

            urls = [self.rpc_url] if self.rpc_url else BSC_RPC_URLS
            for url in urls:
                try:
                    w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": self.timeout}))
                    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
                    _ = w3.eth.block_number
                    self._w3 = w3
                    # Pre-compute function selectors
                    self._sel_epoch  = w3.keccak(text="currentEpoch()")[:4]
                    self._sel_rounds = w3.keccak(text="rounds(uint256)")[:4]
                    self._connected = True
                    logger.info(f"✅ PancakeSwap BNB/USD: connected via {url}")
                    break
                except Exception as e:
                    logger.debug(f"RPC {url} failed: {e}")

            if not self._connected:
                logger.warning("PancakeSwap: no BSC RPC reachable — mock mode")

        except ImportError:
            logger.warning("web3.py not installed — pip install web3")
        except Exception as e:
            logger.warning(f"PancakeSwap init error: {e}")

    def check_connectivity(self) -> bool:
        if self._connected and self._w3:
            try:
                self._w3.eth.block_number
                return True
            except Exception:
                pass
        return False

    def get_current_round(self) -> Optional[PancakeRound]:
        if self._connected and self._w3:
            try:
                return self._fetch_round_onchain()
            except Exception as e:
                logger.warning(f"PancakeSwap on-chain read failed: {e}")

        if self.use_mock_on_failure:
            return _mock_round()
        return None

    def _fetch_round_onchain(self) -> PancakeRound:
        import eth_abi
        addr = self._w3.to_checksum_address(self.contract_address)

        # currentEpoch()
        raw_epoch = self._w3.eth.call({"to": addr, "data": self._sel_epoch})
        epoch = int.from_bytes(raw_epoch, "big")

        # rounds(epoch) — raw bytes, decoded manually
        data = self._sel_rounds + eth_abi.encode(["uint256"], [epoch])
        raw = self._w3.eth.call({"to": addr, "data": data})
        return _decode_round_raw(raw, epoch)
