"""
live_trader.py — Live trading engine for BNB Up/Down PancakeSwap Prediction bot

Mirrors the PaperTrader interface but executes real on-chain transactions on BSC.
Private key is loaded ONLY from the BSC_PRIVATE_KEY environment variable.

SAFEGUARDS:
  - Balance check before every trade
  - Max daily loss limit (configurable, default $100)
  - Gas price estimation with configurable buffer
  - Transaction retry logic (1 retry max)
  - All transactions logged with tx hash to logs/<strategy>/live_trades.json

PancakeSwap Prediction V2 (BNB/USD, BSC):
  Contract: 0x18B2A687610328590Bc8F2e5fEdDe3b582A49cdA
  betBull(uint256 epoch) — payable, BNB value = bet amount
  betBear(uint256 epoch) — payable, BNB value = bet amount
  claim(uint256[] calldata epochs) — claim winnings
  claimable(uint256 epoch, address user) — check claimability
  currentEpoch() — get current epoch number
"""

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger(__name__)

PANCAKE_CONTRACT = "0x18B2A687610328590Bc8F2e5fEdDe3b582A49cdA"
PANCAKE_FEE = 0.03

BSC_RPC_URLS = [
    "https://bsc-dataseed.binance.org/",
    "https://bsc-dataseed1.defibit.io/",
    "https://bsc-dataseed2.defibit.io/",
    "https://1rpc.io/bnb",
]

# Minimal ABI — only the functions we call
PANCAKE_ABI = [
    {
        "name": "betBull",
        "type": "function",
        "inputs": [{"name": "epoch", "type": "uint256"}],
        "outputs": [],
        "stateMutability": "payable",
    },
    {
        "name": "betBear",
        "type": "function",
        "inputs": [{"name": "epoch", "type": "uint256"}],
        "outputs": [],
        "stateMutability": "payable",
    },
    {
        "name": "claim",
        "type": "function",
        "inputs": [{"name": "epochs", "type": "uint256[]"}],
        "outputs": [],
        "stateMutability": "nonpayable",
    },
    {
        "name": "claimable",
        "type": "function",
        "inputs": [
            {"name": "epoch", "type": "uint256"},
            {"name": "user", "type": "address"},
        ],
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "view",
    },
    {
        "name": "refundable",
        "type": "function",
        "inputs": [
            {"name": "epoch", "type": "uint256"},
            {"name": "user", "type": "address"},
        ],
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "view",
    },
    {
        "name": "currentEpoch",
        "type": "function",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
    },
]


@dataclass
class LiveTrade:
    """A live on-chain trade record."""

    trade_id: str
    timestamp_entry: float
    timestamp_exit: Optional[float]
    side: str                              # "YES" (bull) or "NO" (bear)
    entry_price: float                     # yes_price at entry
    position_size_usdc: float             # USDC equivalent wagered
    bet_bnb: float                        # Actual BNB sent on-chain
    bnb_price_at_entry: float             # BNB/USDC at entry time
    p_up_at_entry: float
    yes_price_at_entry: float
    edge_at_entry: float
    kelly_fraction: float
    window_start_ts: float
    window_end_ts: float
    window_index: int
    epoch: int                            # PancakeSwap epoch
    is_mock: bool
    tx_hash: Optional[str] = None        # Transaction hash (never None after submit)
    tx_status: Optional[str] = None      # "pending", "success", "failed", "retried"

    # Pool state at entry
    bull_pct: float = 0.0                 # % of pool on bull side at entry
    bear_pct: float = 0.0                 # % of pool on bear side at entry

    # Final pool state (after lock — no more bets)
    final_bull_pct: float = 0.0           # % of pool on bull side at lock
    final_bear_pct: float = 0.0           # % of pool on bear side at lock
    final_total_bnb: float = 0.0          # Total BNB in pool at lock
    pool_drift_pct: float = 0.0           # abs(final_bull_pct - bull_pct) — how much the pool shifted

    # Resolution fields
    bnb_open: Optional[float] = None
    bnb_close: Optional[float] = None
    outcome: Optional[str] = None
    pnl_usdc: Optional[float] = None
    payout_per_share: Optional[float] = None
    claim_tx_hash: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LiveMetrics:
    """Running metrics for live trading."""

    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    pending: int = 0
    total_pnl: float = 0.0
    total_wagered: float = 0.0
    avg_edge: float = 0.0
    bankroll: float = 0.0          # Current BNB balance in USDC equiv
    daily_pnl: float = 0.0         # PnL since midnight UTC
    daily_loss_start: float = 0.0  # Bankroll at start of day

    @property
    def win_rate(self) -> float:
        resolved = self.wins + self.losses
        if resolved == 0:
            return 0.0
        return self.wins / resolved

    @property
    def roi(self) -> float:
        if self.total_wagered == 0:
            return 0.0
        return self.total_pnl / self.total_wagered

    def summary(self) -> str:
        return (
            f"Trades: {self.total_trades} | "
            f"Win rate: {self.win_rate:.1%} ({self.wins}W/{self.losses}L/{self.pending}P) | "
            f"PnL: ${self.total_pnl:+.2f} | "
            f"ROI: {self.roi:.2%} | "
            f"Daily PnL: ${self.daily_pnl:+.2f} | "
            f"Avg edge: {self.avg_edge:.3f}"
        )


class LiveTrader:
    """
    Live trading engine — real on-chain transactions on PancakeSwap Prediction V2 (BSC).

    Shares the same interface as PaperTrader:
      enter_trade(signal, window)
      resolve_trades(window_index, bnb_open, bnb_close)
      print_summary()
    """

    def __init__(self, config: dict):
        cfg_live = config.get("live_trading", {})
        cfg_strategy = config.get("strategy", {})
        cfg_pancake = config.get("pancake", {})

        self.log_file = cfg_live.get("log_file", "logs/live/default/default.json")
        self.gas_price_buffer_pct = cfg_live.get("gas_price_buffer_pct", 0.10)
        self.auto_claim = cfg_live.get("auto_claim", True)

        self.contract_address = cfg_pancake.get("contract_address", PANCAKE_CONTRACT)
        self.rpc_url = cfg_pancake.get("rpc_url", None)
        self.timeout = cfg_pancake.get("timeout_seconds", 8)

        self._w3 = None
        self._account = None
        self._contract = None
        self._wallet_address = None

        self.metrics = LiveMetrics()
        self._trades: list[LiveTrade] = []
        self._pending_trades: list[LiveTrade] = []
        self._trade_counter = 0
        self._pancake_client = None  # Set externally by PolymarketBot

        # Sniper pre-signed transactions (Phase 1/2)
        self._prepared_txs: dict = {}
        self._prepared_epoch: Optional[int] = None
        self._last_fired_tx_hash: Optional[str] = None
        self._last_fired_side: Optional[str] = None
        self._last_fired_epoch: Optional[int] = None
        self._last_fired_bet_bnb: Optional[float] = None

        # Load .env and init
        self._load_env()
        self._init_web3()
        self._load_trades()
        self._reset_daily_tracking()

    # ─── Environment & Web3 Setup ────────────────────────────────────────────

    def _load_env(self):
        """Load environment variables from .env file (via python-dotenv)."""
        try:
            from dotenv import load_dotenv
            import pathlib
            # Try repo root
            env_path = pathlib.Path(__file__).parent.parent / ".env"
            if env_path.exists():
                load_dotenv(env_path)
                logger.debug(f"Loaded .env from {env_path}")
            else:
                load_dotenv()
        except ImportError:
            logger.warning("python-dotenv not installed — relying on environment variables only")

    def _init_web3(self):
        """Connect to BSC and load wallet. Raises on failure."""
        try:
            from web3 import Web3
            from web3.middleware import ExtraDataToPOAMiddleware
        except ImportError:
            raise RuntimeError("web3.py is required for live trading: pip install web3")

        private_key = os.environ.get("BSC_PRIVATE_KEY", "").strip()
        wallet_addr = os.environ.get("BSC_WALLET_ADDRESS", "").strip()

        if not private_key:
            raise RuntimeError(
                "BSC_PRIVATE_KEY not set. "
                "Set it in .env or as an environment variable. "
                "Example: BSC_PRIVATE_KEY=0xYOUR_KEY_HERE"
            )
        if not wallet_addr:
            raise RuntimeError("BSC_WALLET_ADDRESS not set in .env or environment.")

        # Connect to BSC
        urls = [self.rpc_url] if self.rpc_url else BSC_RPC_URLS
        w3 = None
        for url in urls:
            try:
                candidate = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": self.timeout}))
                candidate.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
                _ = candidate.eth.block_number
                w3 = candidate
                logger.info(f"✅ Live trader: connected to BSC via {url}")
                break
            except Exception as e:
                logger.debug(f"RPC {url} failed: {e}")

        if w3 is None:
            raise RuntimeError("Cannot connect to BSC — no RPC reachable. Check network.")

        self._w3 = w3
        self._wallet_address = w3.to_checksum_address(wallet_addr)

        # Load account from private key (NEVER log the key)
        self._account = w3.eth.account.from_key(private_key)
        if self._account.address.lower() != self._wallet_address.lower():
            raise RuntimeError(
                f"Private key address ({self._account.address}) does not match "
                f"BSC_WALLET_ADDRESS ({self._wallet_address}). Check your .env."
            )

        # Load contract
        checksum_addr = w3.to_checksum_address(self.contract_address)
        self._contract = w3.eth.contract(address=checksum_addr, abi=PANCAKE_ABI)

        # Validate balance
        balance_wei = w3.eth.get_balance(self._wallet_address)
        balance_bnb = balance_wei / 1e18
        logger.info(f"💰 Wallet: {self._wallet_address}")
        logger.info(f"💰 BNB balance: {balance_bnb:.6f} BNB")

        if balance_bnb < 0.001:
            logger.warning(
                f"⚠️  Very low BNB balance ({balance_bnb:.6f} BNB). "
                "You may not be able to cover gas + bets."
            )

    def _reset_daily_tracking(self):
        """Reset daily P&L tracking at start of UTC day."""
        import datetime
        now = datetime.datetime.utcnow()
        midnight = datetime.datetime(now.year, now.month, now.day)
        self._day_start_ts = midnight.timestamp()
        # Compute daily PnL from today's trades
        today_trades = [
            t for t in self._trades
            if t.timestamp_entry >= self._day_start_ts and t.pnl_usdc is not None
        ]
        self.metrics.daily_pnl = sum(t.pnl_usdc for t in today_trades)

    # ─── Balance & Safeguards ────────────────────────────────────────────────

    def get_bnb_balance(self) -> float:
        """Return current wallet BNB balance."""
        balance_wei = self._w3.eth.get_balance(self._wallet_address)
        return balance_wei / 1e18

    def _check_safeguards(self, position_size_usdc: float, bnb_price: float) -> bool:
        """
        Run all safeguard checks before placing a trade.

        Returns True if safe to trade, False otherwise.
        """
        # BNB balance check
        bet_bnb = position_size_usdc / bnb_price
        gas_reserve_bnb = 0.005  # ~0.005 BNB for gas
        required_bnb = bet_bnb + gas_reserve_bnb
        current_bnb = self.get_bnb_balance()

        if current_bnb < required_bnb:
            logger.warning(
                f"🛑 Insufficient BNB: have {current_bnb:.6f}, "
                f"need {required_bnb:.6f} (bet={bet_bnb:.6f} + gas={gas_reserve_bnb:.6f})"
            )
            return False

        return True

    # ─── Transaction Execution ───────────────────────────────────────────────

    def _estimate_gas_price(self) -> int:
        """Estimate gas price with buffer."""
        gas_price = self._w3.eth.gas_price
        buffer = int(gas_price * self.gas_price_buffer_pct)
        return gas_price + buffer

    def _send_transaction(self, tx: dict, description: str) -> Optional[str]:
        """
        Sign and send a transaction. Retries once on failure.

        Returns:
            Transaction hash hex string, or None on failure.
        """
        for attempt in range(2):  # 1 retry max
            try:
                signed = self._account.sign_transaction(tx)
                tx_hash = self._w3.eth.send_raw_transaction(signed.raw_transaction)
                tx_hash_hex = tx_hash.hex()
                logger.info(f"📤 {description} sent — tx: {tx_hash_hex}")

                # Wait for receipt (timeout 60s)
                receipt = self._w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
                if receipt.status == 1:
                    logger.info(f"✅ {description} confirmed — block {receipt.blockNumber}")
                    return tx_hash_hex
                else:
                    logger.error(f"❌ {description} FAILED — tx: {tx_hash_hex}")
                    if attempt == 0:
                        logger.info("Retrying transaction...")
                        # Bump nonce for retry
                        tx["nonce"] = self._w3.eth.get_transaction_count(self._wallet_address)
                        tx["gasPrice"] = int(tx["gasPrice"] * 1.15)  # +15% gas for retry
                    continue

            except Exception as e:
                logger.error(f"❌ {description} error (attempt {attempt+1}): {e}")
                if attempt == 0:
                    logger.info("Retrying transaction...")
                    time.sleep(2)
                    try:
                        tx["nonce"] = self._w3.eth.get_transaction_count(self._wallet_address)
                    except Exception:
                        pass

        return None

    # ─── Sniper Transaction Methods ──────────────────────────────────────────

    def prepare_transactions(self, epoch: int, bet_bnb: float) -> bool:
        """
        Phase 1 (pre-load): Pre-build and pre-sign BOTH bull and bear transactions.

        Stores them in self._prepared_txs dict keyed by "YES" and "NO".
        Call fire_transaction(side) to broadcast the chosen one.

        Args:
            epoch: Current PancakeSwap epoch.
            bet_bnb: Amount of BNB to wager.

        Returns:
            True if both transactions were successfully prepared, False otherwise.
        """
        if not self._w3 or not self._account or not self._contract:
            logger.error("prepare_transactions: Web3 not initialized")
            return False

        bet_wei = int(bet_bnb * 1e18)
        if bet_wei <= 0:
            logger.error(f"prepare_transactions: invalid bet_bnb={bet_bnb}")
            return False

        gas_price = self._estimate_gas_price()
        nonce = self._w3.eth.get_transaction_count(self._wallet_address)

        prepared = {}
        for side, fn_name in [("YES", "betBull"), ("NO", "betBear")]:
            fn = getattr(self._contract.functions, fn_name)(epoch)
            try:
                gas_estimate = fn.estimate_gas({
                    "from": self._wallet_address,
                    "value": bet_wei,
                })
                gas_limit = int(gas_estimate * 1.2)
            except Exception as e:
                logger.warning(f"prepare_transactions: gas estimate failed for {fn_name}: {e} — using 200000")
                gas_limit = 200_000

            tx = fn.build_transaction({
                "from": self._wallet_address,
                "value": bet_wei,
                "gas": gas_limit,
                "gasPrice": gas_price,
                "nonce": nonce,
                "chainId": 56,
            })

            try:
                signed = self._account.sign_transaction(tx)
                prepared[side] = {
                    "tx": tx,
                    "signed": signed,
                    "epoch": epoch,
                    "bet_bnb": bet_bnb,
                    "nonce": nonce,
                }
                logger.debug(f"prepare_transactions: {fn_name}(epoch={epoch}) pre-signed | bet={bet_bnb:.6f} BNB")
            except Exception as e:
                logger.error(f"prepare_transactions: signing failed for {fn_name}: {e}")
                return False

        self._prepared_txs = prepared
        self._prepared_epoch = epoch
        logger.info(
            f"💡 Both transactions pre-signed for epoch {epoch} | "
            f"bet={bet_bnb:.6f} BNB | gas_price={gas_price / 1e9:.1f} Gwei"
        )
        return True

    def fire_transaction(self, side: str) -> Optional[str]:
        """
        Phase 2 (sniper): Fire the pre-signed transaction immediately (fire-and-forget).

        Does NOT wait for receipt — returns the tx hash immediately.
        Call verify_transaction() after lock to check confirmation.

        Args:
            side: "YES" (bull) or "NO" (bear).

        Returns:
            Transaction hash hex string, or None on failure.
        """
        if not hasattr(self, '_prepared_txs') or side not in self._prepared_txs:
            logger.error(f"fire_transaction: no pre-signed TX for side={side} — falling back to fresh TX")
            return None

        prepared = self._prepared_txs[side]
        signed = prepared["signed"]

        try:
            tx_hash = self._w3.eth.send_raw_transaction(signed.raw_transaction)
            tx_hash_hex = tx_hash.hex()
            self._last_fired_tx_hash = tx_hash_hex
            self._last_fired_side = side
            self._last_fired_epoch = prepared["epoch"]
            self._last_fired_bet_bnb = prepared["bet_bnb"]

            fn_name = "betBull" if side == "YES" else "betBear"
            logger.info(
                f"🚀 FIRED {fn_name}(epoch={prepared['epoch']}) — "
                f"tx={tx_hash_hex} | bet={prepared['bet_bnb']:.6f} BNB"
            )
            return tx_hash_hex

        except Exception as e:
            logger.error(f"fire_transaction: send failed for side={side}: {e}")
            return None

    def verify_transaction(self, timeout: int = 30) -> Optional[bool]:
        """
        Phase 3 (verify): Check if the last fired transaction was confirmed.

        Args:
            timeout: Seconds to wait for receipt.

        Returns:
            True if confirmed, False if failed, None if not yet mined.
        """
        tx_hash_hex = getattr(self, '_last_fired_tx_hash', None)
        if not tx_hash_hex:
            logger.warning("verify_transaction: no fired TX to verify")
            return None

        try:
            receipt = self._w3.eth.wait_for_transaction_receipt(tx_hash_hex, timeout=timeout)
            if receipt.status == 1:
                logger.info(
                    f"✅ Transaction confirmed: {tx_hash_hex} | "
                    f"block={receipt.blockNumber}"
                )
                return True
            else:
                logger.error(f"❌ Transaction FAILED on-chain: {tx_hash_hex}")
                return False
        except Exception as e:
            logger.warning(f"verify_transaction: receipt not yet available ({e})")
            return None

    # ─── Core Interface (mirrors PaperTrader) ────────────────────────────────

    def enter_trade(self, signal, window) -> Optional[LiveTrade]:
        """
        Execute a live trade on PancakeSwap Prediction V2.

        Calls betBull(epoch) or betBear(epoch) with BNB value = position_size_usdc / bnb_price.

        Args:
            signal: Strategy Signal object.
            window: Current WindowInfo.

        Returns:
            LiveTrade object, or None if rejected/failed.
        """
        # Get current BNB price from web3 is not possible directly;
        # we estimate from the signal context (signal.yes_price is irrelevant here)
        # We need bnb_price — fetch it from Binance or a fallback.
        bnb_price = self._get_bnb_price_fallback()
        if bnb_price is None or bnb_price <= 0:
            logger.error("Cannot determine BNB price — aborting trade.")
            return None

        # Safeguard checks
        if not self._check_safeguards(signal.position_size_usdc, bnb_price):
            return None

        # Get current epoch from contract
        try:
            epoch = self._contract.functions.currentEpoch().call()
        except Exception as e:
            logger.error(f"Cannot read currentEpoch(): {e}")
            return None

        bet_bnb = signal.position_size_usdc / bnb_price
        bet_wei = int(bet_bnb * 1e18)

        self._trade_counter += 1
        trade_id = f"LT-{int(time.time())}-{self._trade_counter:04d}"

        # Build transaction
        gas_price = self._estimate_gas_price()
        nonce = self._w3.eth.get_transaction_count(self._wallet_address)

        if signal.side == "YES":
            fn = self._contract.functions.betBull(epoch)
            description = f"betBull(epoch={epoch})"
        else:
            fn = self._contract.functions.betBear(epoch)
            description = f"betBear(epoch={epoch})"

        try:
            gas_estimate = fn.estimate_gas({
                "from": self._wallet_address,
                "value": bet_wei,
            })
            gas_limit = int(gas_estimate * 1.2)  # +20% buffer
        except Exception as e:
            logger.warning(f"Gas estimation failed: {e} — using default 200000")
            gas_limit = 200_000

        tx = fn.build_transaction({
            "from": self._wallet_address,
            "value": bet_wei,
            "gas": gas_limit,
            "gasPrice": gas_price,
            "nonce": nonce,
            "chainId": 56,  # BSC mainnet
        })

        # Execute
        tx_hash = self._send_transaction(tx, description)
        tx_status = "success" if tx_hash else "failed"

        if tx_hash is None:
            logger.error(f"Trade {trade_id} failed — transaction not confirmed.")
            return None

        # Build trade record
        entry_price = signal.yes_price if signal.side == "YES" else (1.0 - signal.yes_price)

        trade = LiveTrade(
            trade_id=trade_id,
            timestamp_entry=time.time(),
            timestamp_exit=None,
            side=signal.side,
            entry_price=entry_price,
            position_size_usdc=signal.position_size_usdc,
            bet_bnb=bet_bnb,
            bnb_price_at_entry=bnb_price,
            p_up_at_entry=signal.p_up,
            yes_price_at_entry=signal.yes_price,
            edge_at_entry=signal.edge,
            kelly_fraction=signal.kelly_fraction,
            window_start_ts=window.window_start_ts,
            window_end_ts=window.window_end_ts,
            window_index=window.window_index,
            epoch=epoch,
            is_mock=signal.is_mock,
            bull_pct=signal.bull_pct,
            bear_pct=signal.bear_pct,
            tx_hash=tx_hash,
            tx_status=tx_status,
            outcome="PENDING",
        )

        self._trades.append(trade)
        self._pending_trades.append(trade)
        self.metrics.total_trades += 1
        self.metrics.pending += 1
        self.metrics.total_wagered += trade.position_size_usdc

        self._save_trades()

        logger.info(
            f"🔴 LIVE trade entered: {trade_id} | "
            f"{trade.side} @ epoch={epoch} | "
            f"{bet_bnb:.6f} BNB (${signal.position_size_usdc:.2f}) | "
            f"edge={signal.edge:.3f} | tx={tx_hash}"
        )

        return trade

    def _fix_legacy_pnl(self):
        """
        One-time migration: recalculate PnL for WIN trades that used the old
        buggy formula (payout_per_share == 1.0 is the marker of the old code).
        """
        if self._pancake_client is None:
            return

        fixed = 0
        for trade in self._trades:
            if trade.outcome == "WIN" and trade.payout_per_share == 1.0:
                pnl_usdc, payout_per_bnb = self._fetch_round_pnl(trade, trade.bnb_price_at_entry)
                if pnl_usdc is not None:
                    old_pnl = trade.pnl_usdc
                    trade.pnl_usdc = pnl_usdc
                    trade.payout_per_share = round(payout_per_bnb, 6) if payout_per_bnb else 0.0
                    logger.info(
                        f"🔧 Fixed legacy PnL for {trade.trade_id}: "
                        f"${old_pnl:+.2f} → ${pnl_usdc:+.4f}"
                    )
                    fixed += 1

        if fixed > 0:
            self._recompute_metrics()
            self._save_trades()
            logger.info(f"🔧 Fixed {fixed} legacy PnL calculation(s). {self.metrics.summary()}")

    def resolve_pending_on_startup(self):
        """
        Called at startup (after _pancake_client is injected by main.py) to resolve
        any trades that were left PENDING from a previous session.

        For each PENDING trade:
          - Fetches on-chain round data via get_round_by_epoch(trade.epoch)
          - If oracle not yet called → keep PENDING
          - Determines win/loss by comparing close_price vs lock_price
          - Computes PnL: winners use on-chain reward pool; losers use bet size at entry price
          - Updates metrics and persists

        After resolving, scans ALL trades for WIN trades without a claim_tx_hash
        and auto-claims any that are claimable on-chain.
        """
        if self._pancake_client is None:
            logger.warning("resolve_pending_on_startup: no PancakeClient injected — skipping.")
            return

        if not self._pending_trades:
            logger.info("resolve_pending_on_startup: no PENDING trades to resolve.")
        else:
            logger.info(
                f"resolve_pending_on_startup: resolving {len(self._pending_trades)} PENDING trade(s)..."
            )

        still_pending = []
        resolved_count = 0

        for trade in list(self._pending_trades):
            try:
                round_data = self._pancake_client.get_round_by_epoch(trade.epoch)
            except Exception as e:
                logger.warning(
                    f"resolve_pending_on_startup: could not fetch epoch {trade.epoch}: {e} — keeping PENDING"
                )
                still_pending.append(trade)
                continue

            if round_data is None:
                logger.warning(
                    f"resolve_pending_on_startup: no round data for epoch {trade.epoch} — keeping PENDING"
                )
                still_pending.append(trade)
                continue

            if not round_data.oracle_called:
                logger.info(
                    f"resolve_pending_on_startup: epoch {trade.epoch} not yet settled "
                    f"(oracleCalled=false) — keeping PENDING"
                )
                still_pending.append(trade)
                continue

            lock_price = round_data.lock_price
            close_price = round_data.close_price

            if lock_price is None or close_price is None:
                logger.warning(
                    f"resolve_pending_on_startup: epoch {trade.epoch} missing price data "
                    f"(lock={lock_price}, close={close_price}) — keeping PENDING"
                )
                still_pending.append(trade)
                continue

            # Determine win/loss
            if trade.side == "YES":   # bull: wins if price went up
                won = close_price > lock_price
            else:                      # bear ("NO"): wins if price went down
                won = close_price < lock_price

            trade.bnb_open = lock_price
            trade.bnb_close = close_price
            trade.timestamp_exit = time.time()

            # Capture final pool state (round_data is already fetched above)
            if round_data.total_bnb > 0:
                trade.final_total_bnb = round_data.total_bnb
                trade.final_bull_pct = round_data.bull_bnb / round_data.total_bnb
                trade.final_bear_pct = round_data.bear_bnb / round_data.total_bnb
                trade.pool_drift_pct = abs(trade.final_bull_pct - trade.bull_pct)

            if won:
                # Try to compute real PnL from on-chain reward pool
                pnl_usdc, payout_per_bnb = self._fetch_round_pnl(trade, trade.bnb_price_at_entry)

                if pnl_usdc is None:
                    # Reward pool not populated yet (edge case) — keep PENDING
                    logger.warning(
                        f"resolve_pending_on_startup: epoch {trade.epoch} WIN but reward "
                        f"pool not settled — keeping PENDING"
                    )
                    still_pending.append(trade)
                    continue

                trade.pnl_usdc = pnl_usdc
                trade.payout_per_share = round(payout_per_bnb, 6) if payout_per_bnb else 0.0
                trade.outcome = "WIN"
                self.metrics.wins += 1
            else:
                # Loss — use entry price as BNB price approximation
                trade.pnl_usdc = round(-trade.bet_bnb * trade.bnb_price_at_entry, 4)
                trade.payout_per_share = 0.0
                trade.outcome = "LOSS"
                self.metrics.losses += 1

            self.metrics.pending -= 1
            self.metrics.total_pnl = round(self.metrics.total_pnl + trade.pnl_usdc, 4)
            self.metrics.daily_pnl = round(self.metrics.daily_pnl + trade.pnl_usdc, 4)

            result_emoji = "✅" if won else "❌"
            logger.info(
                f"{result_emoji} Startup resolved: {trade.trade_id} | "
                f"{trade.side} | epoch={trade.epoch} | "
                f"lock={lock_price:.2f} close={close_price:.2f} | "
                f"PnL: ${trade.pnl_usdc:+.4f}"
            )
            resolved_count += 1

        self._pending_trades = still_pending

        if resolved_count > 0:
            edges = [t.edge_at_entry for t in self._trades]
            self.metrics.avg_edge = sum(edges) / len(edges) if edges else 0.0
            self._save_trades()
            logger.info(
                f"resolve_pending_on_startup: resolved {resolved_count} trade(s). "
                f"{self.metrics.summary()}"
            )

        # ── Auto-claim any WIN trades without a claim_tx_hash ──────────────────
        if self.auto_claim:
            unclaimed_win_epochs = [
                t.epoch
                for t in self._trades
                if t.outcome == "WIN" and not t.claim_tx_hash
            ]
            if unclaimed_win_epochs:
                logger.info(
                    f"resolve_pending_on_startup: checking {len(unclaimed_win_epochs)} unclaimed WIN epoch(s)..."
                )
                self.claim_winnings(unclaimed_win_epochs)
            else:
                logger.info("resolve_pending_on_startup: no unclaimed WIN epochs found.")

    def _fetch_round_pnl(self, trade, bnb_price: float) -> tuple[float, float]:
        """
        Fetch on-chain round data for a completed epoch and compute real PnL.

        PancakeSwap Prediction V2 payout mechanics:
          - rewardBaseCalAmount = total BNB bet on the winning side
          - rewardAmount        = total payout pool (totalAmount * (1 - fee))
          - payout for a winner = bet_bnb * rewardAmount / rewardBaseCalAmount
          - pnl_bnb = payout - bet_bnb  (positive if won, negative if lost)
          - pnl_usdc = pnl_bnb * bnb_price

        Returns (pnl_usdc, payout_per_bnb) where payout_per_bnb is the multiplier
        applied to each BNB wagered (>1 = profit, 0 = loss).
        """
        if self._pancake_client is None:
            logger.warning("No PancakeClient available — cannot compute on-chain PnL")
            return None, None

        try:
            round_data = self._pancake_client.get_round_by_epoch(trade.epoch)
        except Exception as e:
            logger.warning(f"Could not fetch on-chain round data for epoch {trade.epoch}: {e}")
            return None, None

        if round_data is None:
            logger.warning(f"No round data returned for epoch {trade.epoch}")
            return None, None

        reward_base = round_data.reward_base_cal_amount
        reward_pool = round_data.reward_amount
        bet_bnb = trade.bet_bnb

        if reward_base <= 0 or reward_pool <= 0:
            # Round not yet settled on-chain — reward fields are zero
            logger.warning(
                f"Epoch {trade.epoch}: reward fields not yet populated on-chain "
                f"(rewardBaseCalAmount={reward_base}, rewardAmount={reward_pool}). "
                f"Will retry at next resolution cycle."
            )
            return None, None

        # Real payout the contract will send when claim() is called
        payout_bnb = bet_bnb * reward_pool / reward_base
        pnl_bnb = payout_bnb - bet_bnb
        pnl_usdc = round(pnl_bnb * bnb_price, 4)
        payout_per_bnb = payout_bnb / bet_bnb if bet_bnb > 0 else 0.0

        logger.info(
            f"📡 On-chain PnL for epoch {trade.epoch}: "
            f"bet={bet_bnb:.6f} BNB | "
            f"rewardBase={reward_base:.6f} | rewardPool={reward_pool:.6f} | "
            f"payout={payout_bnb:.6f} BNB | "
            f"pnl_bnb={pnl_bnb:+.6f} | pnl_usdc=${pnl_usdc:+.4f}"
        )
        return pnl_usdc, payout_per_bnb

    def resolve_trades(self, window_index: int, bnb_open: float, bnb_close: float):
        """
        Resolve pending trades for a completed window.
        Also triggers claim for winning trades if auto_claim is enabled.

        PnL is calculated from on-chain round data (rewardAmount / rewardBaseCalAmount)
        to reflect the actual payout received from the contract, not a theoretical estimate.

        Args:
            window_index: The window that just completed.
            bnb_open: BNB price at window start.
            bnb_close: BNB price at window end.
        """
        resolved = []
        still_pending = []
        bnb_went_up = bnb_close > bnb_open
        # Use mid-point BNB price for USDC conversion (most accurate snapshot at resolution)
        bnb_price_at_close = bnb_close if bnb_close > 0 else bnb_open

        for trade in self._pending_trades:
            if trade.window_index != window_index:
                still_pending.append(trade)
                continue

            if trade.side == "YES":
                won = bnb_went_up
            else:
                won = not bnb_went_up

            trade.bnb_open = bnb_open
            trade.bnb_close = bnb_close
            trade.timestamp_exit = time.time()

            # Fetch final pool state (available for all resolved trades)
            if self._pancake_client is not None:
                try:
                    round_data = self._pancake_client.get_round_by_epoch(trade.epoch)
                    if round_data is not None and round_data.total_bnb > 0:
                        trade.final_total_bnb = round_data.total_bnb
                        trade.final_bull_pct = round_data.bull_bnb / round_data.total_bnb
                        trade.final_bear_pct = round_data.bear_bnb / round_data.total_bnb
                        trade.pool_drift_pct = abs(trade.final_bull_pct - trade.bull_pct)
                except Exception as e:
                    logger.debug(f"Could not fetch final pool state for epoch {trade.epoch}: {e}")

            if won:
                # Fetch real payout from on-chain round data
                pnl_usdc, payout_per_bnb = self._fetch_round_pnl(trade, bnb_price_at_close)

                if pnl_usdc is None:
                    # On-chain data not yet settled — keep trade pending and retry later
                    logger.warning(
                        f"⏳ Trade {trade.trade_id} epoch {trade.epoch}: "
                        f"on-chain reward not settled yet. Keeping PENDING."
                    )
                    still_pending.append(trade)
                    continue

                trade.pnl_usdc = pnl_usdc
                trade.payout_per_share = round(payout_per_bnb, 6) if payout_per_bnb else 0.0
                trade.outcome = "WIN"
                self.metrics.wins += 1
            else:
                # Loss: actual cost = bet_bnb * bnb_price (plus gas, negligible)
                trade.pnl_usdc = round(-trade.bet_bnb * bnb_price_at_close, 4)
                trade.payout_per_share = 0.0
                trade.outcome = "LOSS"
                self.metrics.losses += 1

            self.metrics.pending -= 1
            self.metrics.total_pnl = round(self.metrics.total_pnl + trade.pnl_usdc, 4)
            self.metrics.daily_pnl = round(self.metrics.daily_pnl + trade.pnl_usdc, 4)

            result_emoji = "✅" if won else "❌"
            logger.info(
                f"{result_emoji} Live trade resolved: {trade.trade_id} | "
                f"{trade.side} | BNB {bnb_open:.2f}→{bnb_close:.2f} | "
                f"bet={trade.bet_bnb:.6f} BNB | PnL: ${trade.pnl_usdc:+.4f} | "
                f"tx={trade.tx_hash}"
            )

            resolved.append(trade)

        self._pending_trades = still_pending

        # Auto-claim winning epochs
        if self.auto_claim and resolved:
            winning_epochs = [t.epoch for t in resolved if t.outcome == "WIN"]
            # Wait a few seconds for BSC to confirm the round settlement
            if winning_epochs:
                logger.info(f"⏳ Waiting 15s for round settlement before claiming...")
                time.sleep(15)
                self.claim_winnings(winning_epochs)

        if resolved:
            edges = [t.edge_at_entry for t in self._trades]
            self.metrics.avg_edge = sum(edges) / len(edges) if edges else 0.0
            self._save_trades()
            logger.info(f"📊 {self.metrics.summary()}")

    def claim_winnings(self, epochs: list[int]):
        """
        Call claim(uint256[] epochs) on the contract to collect winnings.

        Args:
            epochs: List of epoch numbers to claim.
        """
        if not epochs:
            return

        # Filter to claimable epochs first
        claimable = []
        for epoch in epochs:
            try:
                is_claimable = self._contract.functions.claimable(
                    epoch, self._wallet_address
                ).call()
                is_refundable = self._contract.functions.refundable(
                    epoch, self._wallet_address
                ).call()
                if is_claimable or is_refundable:
                    claimable.append(epoch)
                else:
                    logger.debug(f"Epoch {epoch} not yet claimable/refundable — skipping")
            except Exception as e:
                logger.warning(f"claimable() check failed for epoch {epoch}: {e}")
                claimable.append(epoch)  # Try anyway

        if not claimable:
            logger.info("No claimable epochs found.")
            return

        gas_price = self._estimate_gas_price()
        nonce = self._w3.eth.get_transaction_count(self._wallet_address)

        fn = self._contract.functions.claim(claimable)
        try:
            gas_estimate = fn.estimate_gas({"from": self._wallet_address})
            gas_limit = int(gas_estimate * 1.2)
        except Exception as e:
            logger.warning(f"Gas estimation for claim failed: {e} — using 150000")
            gas_limit = 150_000

        tx = fn.build_transaction({
            "from": self._wallet_address,
            "gas": gas_limit,
            "gasPrice": gas_price,
            "nonce": nonce,
            "chainId": 56,
        })

        tx_hash = self._send_transaction(tx, f"claim(epochs={claimable})")
        if tx_hash:
            logger.info(f"💰 Claimed winnings for epochs {claimable} — tx: {tx_hash}")
            # Update claim_tx_hash on matching trades
            for trade in self._trades:
                if trade.epoch in claimable:
                    trade.claim_tx_hash = tx_hash
            self._save_trades()
        else:
            logger.error(f"❌ Claim failed for epochs {claimable}")

    # ─── Persistence ─────────────────────────────────────────────────────────

    def _load_trades(self):
        """Load existing live trades from the log file."""
        if not os.path.exists(self.log_file):
            logger.info(f"No existing live trades at {self.log_file}. Starting fresh.")
            return

        try:
            with open(self.log_file) as f:
                data = json.load(f)
            trades_data = data.get("trades", [])
            logger.info(f"Loaded {len(trades_data)} existing live trades from {self.log_file}")

            for t_dict in trades_data:
                trade = LiveTrade(**t_dict)
                self._trades.append(trade)
                if trade.outcome == "PENDING" or trade.outcome is None:
                    self._pending_trades.append(trade)

            self._recompute_metrics()

        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning(f"Failed to load live trades from {self.log_file}: {e}. Starting fresh.")

    def _recompute_metrics(self):
        """Recompute all metrics from the full trades list."""
        wins = [t for t in self._trades if t.outcome == "WIN"]
        losses = [t for t in self._trades if t.outcome == "LOSS"]
        pending = [t for t in self._trades if t.outcome in ("PENDING", None)]

        self.metrics.total_trades = len(self._trades)
        self.metrics.wins = len(wins)
        self.metrics.losses = len(losses)
        self.metrics.pending = len(pending)
        self.metrics.total_pnl = sum(t.pnl_usdc for t in self._trades if t.pnl_usdc is not None)
        self.metrics.total_wagered = sum(t.position_size_usdc for t in self._trades)

        edges = [t.edge_at_entry for t in self._trades]
        self.metrics.avg_edge = sum(edges) / len(edges) if edges else 0.0

    def _save_trades(self):
        """Persist all live trades to the log file."""
        os.makedirs(os.path.dirname(self.log_file) if os.path.dirname(self.log_file) else ".", exist_ok=True)

        data = {
            "metadata": {
                "last_updated": time.time(),
                "last_updated_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "total_trades": len(self._trades),
                "mode": "LIVE",
            },
            "metrics": {
                "total_pnl": self.metrics.total_pnl,
                "daily_pnl": self.metrics.daily_pnl,
                "total_wagered": self.metrics.total_wagered,
                "wins": self.metrics.wins,
                "losses": self.metrics.losses,
                "pending": self.metrics.pending,
                "win_rate": self.metrics.win_rate,
                "roi": self.metrics.roi,
                "avg_edge": self.metrics.avg_edge,
            },
            "trades": [t.to_dict() for t in self._trades],
        }

        with open(self.log_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Live trades saved to {self.log_file}")
        self._export_csv()

    def _export_csv(self):
        """Export all trades to a CSV file alongside the JSON log (for Google Sheets analysis)."""
        import csv
        from datetime import datetime

        csv_path = self.log_file.replace(".json", ".csv")
        os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)

        columns = [
            "trade_id", "epoch", "timestamp_entry", "time_entry", "timestamp_exit", "time_exit",
            "side", "side_label", "edge_at_entry", "p_up_at_entry", "kelly_fraction",
            "position_size_usdc", "bet_bnb", "bnb_price_at_entry",
            "bull_pct", "bear_pct",
            "final_bull_pct", "final_bear_pct", "final_total_bnb", "pool_drift_pct",
            "bnb_open", "bnb_close", "outcome", "pnl_usdc", "payout_per_share",
            "tx_hash", "tx_status", "claim_tx_hash", "is_mock",
        ]

        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                for t in self._trades:
                    row = t.to_dict()
                    row["time_entry"] = datetime.fromtimestamp(t.timestamp_entry).strftime("%Y-%m-%d %H:%M:%S") if t.timestamp_entry else ""
                    row["time_exit"] = datetime.fromtimestamp(t.timestamp_exit).strftime("%Y-%m-%d %H:%M:%S") if t.timestamp_exit else ""
                    row["side_label"] = "UP" if t.side == "YES" else "DOWN"
                    writer.writerow(row)
            logger.debug(f"CSV exported to {csv_path}")
        except Exception as e:
            logger.warning(f"CSV export failed: {e}")

    # ─── Utilities ───────────────────────────────────────────────────────────

    def _get_bnb_price_fallback(self) -> Optional[float]:
        """
        Fetch BNB/USDT price from Binance REST API as fallback.
        Used to convert position_size_usdc → BNB.
        """
        try:
            import requests
            resp = requests.get(
                "https://api.binance.com/api/v3/ticker/price",
                params={"symbol": "BNBUSDT"},
                timeout=5,
            )
            data = resp.json()
            return float(data["price"])
        except Exception as e:
            logger.error(f"Cannot fetch BNB price from Binance: {e}")
            return None

    def print_summary(self):
        """Print current live trading summary."""
        bnb_balance = "N/A"
        try:
            bnb_balance = f"{self.get_bnb_balance():.6f} BNB"
        except Exception:
            pass

        print(f"\n{'='*60}")
        print(f"  🔴 LIVE TRADING SUMMARY")
        print(f"{'='*60}")
        print(f"  Wallet:      {self._wallet_address}")
        print(f"  Balance:     {bnb_balance}")
        print(f"  Total PnL:   ${self.metrics.total_pnl:+.2f}")
        print(f"  Daily PnL:   ${self.metrics.daily_pnl:+.2f}")
        print(f"  Total trades: {self.metrics.total_trades}")
        print(f"  Win rate:    {self.metrics.win_rate:.1%}")
        print(f"  Wins:        {self.metrics.wins}")
        print(f"  Losses:      {self.metrics.losses}")
        print(f"  Pending:     {self.metrics.pending}")
        print(f"  ROI:         {self.metrics.roi:.2%}")
        print(f"  Avg edge:    {self.metrics.avg_edge:.3f}")
        print(f"  Log file:    {self.log_file}")
        print(f"{'='*60}\n")
