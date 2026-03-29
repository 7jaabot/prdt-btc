"""
main.py — Entry point for BNB Up/Down PancakeSwap 5mn trading bot

Architecture:
  - asyncio event loop runs the Binance WebSocket feed (async)
  - Main loop polls PRDT contract every N seconds (sync via executor)
  - Strategy evaluated in the last 60s of each window
  - PaperTrader or LiveTrader logs and resolves trades

Key difference vs Polymarket:
  - No order book YES/NO price — reads on-chain pool (bull/bear amounts)
  - "yes_price equivalent" = bull_ratio / (1 - fee) (break-even probability)
  - Same Kelly/edge logic applies on top of this

Run with:
  python src/main.py              (interactive menu)
  python src/main.py --live       (live trading, non-interactive)
  python src/main.py --paper      (paper trading, continue)
  python src/main.py --fresh      (paper trading, fresh start)
  python src/main.py --dry-run    (connectivity test only)
  python src/main.py --config config.json
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

# Adjust path so we can import sibling modules
sys.path.insert(0, str(Path(__file__).parent))

from market_data import BinanceFeed, PricePoint
from pancake import PancakeClient, PancakeRound
from strategy import WindowInfo, window_from_round
from strategies import STRATEGIES
from paper_trader import PaperTrader


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def _cleanup_old_logs(log_dir: str, max_days: int = 30):
    """Delete log files older than max_days in log_dir."""
    import glob
    import time as _time
    cutoff = _time.time() - max_days * 86400
    for path in glob.glob(os.path.join(log_dir, "run-*.log")):
        if os.path.getmtime(path) < cutoff:
            try:
                os.remove(path)
                logging.debug(f"Deleted old log: {path}")
            except OSError:
                pass


def setup_logging(config: dict, dashboard_mode: bool = False, strategy_key: str = "", trading_mode: str = ""):
    """Configure logging from config.

    When dashboard_mode=True, reduces console handler to WARNING level
    (dashboard is the primary display) but keeps file logging at full level.
    Log files are written as logs/<mode>/<strategy>/run-YYYY-MM-DD.log.
    Files older than 30 days are deleted automatically.
    """
    import datetime
    cfg = config.get("logging", {})
    level_str = cfg.get("level", "INFO")
    level = getattr(logging, level_str, logging.INFO)
    fmt = cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Daily log file: logs/<mode>/<strategy>/run-YYYY-MM-DD.log
    parts = ["logs"]
    if trading_mode:
        parts.append(trading_mode)
    if strategy_key:
        parts.append(strategy_key)
    log_dir = os.path.join(*parts) if len(parts) > 1 else "logs"
    os.makedirs(log_dir, exist_ok=True)
    today = datetime.date.today().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, f"run-{today}.log")

    # Purge logs older than 30 days
    _cleanup_old_logs(log_dir, max_days=30)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(fmt))
    root_logger.addHandler(file_handler)

    # Console handler — WARNING only when dashboard is active
    console_level = logging.WARNING if dashboard_mode else level
    try:
        import colorlog
        console_handler = colorlog.StreamHandler()
        console_handler.setFormatter(colorlog.ColoredFormatter(
            f"%(log_color)s{fmt}",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            }
        ))
    except ImportError:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(fmt))

    console_handler.setLevel(console_level)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.DEBUG)


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    """Load configuration from JSON file."""
    config_path = Path(path)
    if not config_path.exists():
        # Try relative to repo root
        repo_root = Path(__file__).parent.parent
        config_path = repo_root / "config.json"

    if not config_path.exists():
        logging.warning(f"Config not found at {path}, using defaults.")
        return {}

    with open(config_path) as f:
        config = json.load(f)
    logging.info(f"Config loaded from {config_path}")
    return config


# ─────────────────────────────────────────────────────────────────────────────
# Mode selection
# ─────────────────────────────────────────────────────────────────────────────

def reset_paper_trades(config: dict):
    """Reset paper trading history."""
    log_file = config.get("paper_trading", {}).get("log_file", "logs/paper/default/default.json")
    empty = {
        "metadata": {"last_updated": 0, "last_updated_iso": "", "total_trades": 0},
        "metrics": {
            "bankroll": config.get("strategy", {}).get("starting_bankroll_usdc", 1000.0),
            "total_pnl": 0.0, "win_rate": 0.0, "roi": 0.0,
            "total_wagered": 0.0, "wins": 0, "losses": 0, "pending": 0, "avg_edge": 0.0
        },
        "trades": []
    }
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w") as f:
        json.dump(empty, f, indent=2)
    logging.info(f"🧹 Fresh start: reset {log_file}")


def prompt_edge_filter(strategy_label: str = "") -> 'EdgeFilter':
    """
    Prompt user for edge range filter.

    Syntax:
        (empty) → no filter
        >0.1    → edge > 0.1
        <0.3    → edge < 0.3
        >0.1 <0.3 → 0.1 < edge < 0.3
    """
    from strategies.combined import EdgeFilter
    label = f" for {strategy_label}" if strategy_label else ""
    print(f"  Edge filter{label} (e.g. >0.1 <0.3, or ENTER for no filter):")
    text = input("  Edge range: ").strip()
    ef = EdgeFilter.parse(text)
    if ef.min_edge is not None or ef.max_edge is not None:
        print(f"  → Filter: {ef}")
    else:
        print(f"  → No edge filter")
    return ef


def select_strategy_interactive(config: dict):
    """Display strategy selection menu and return a strategy instance."""
    from strategies.combined import CombinedStrategy, EdgeFilter

    strategy_list = list(STRATEGIES.items())

    print("\n" + "=" * 50)
    print("  Select Strategy")
    print("=" * 50)
    print("  [0] 🔗 Combined (multiple strategies consensus)")
    for i, (key, cls) in enumerate(strategy_list, 1):
        try:
            name = cls(config).name
        except Exception:
            name = key
        print(f"  [{i}] {name}")
    print("=" * 50)

    while True:
        choice = input("Select strategy: ").strip()
        try:
            idx = int(choice)
            if 0 <= idx <= len(strategy_list):
                break
        except ValueError:
            pass
        print(f"Invalid choice. Please enter 0-{len(strategy_list)}.")

    if idx == 0:
        # Combined strategy mode
        print("\n  🔗 Combined Strategy — select strategies to combine")
        print("  Available strategies:")
        for i, (key, cls) in enumerate(strategy_list, 1):
            try:
                name = cls(config).name
            except Exception:
                name = key
            print(f"    [{i}] {name}")

        while True:
            combo_input = input("  Enter strategy numbers separated by commas (e.g. 1,4,11): ").strip()
            try:
                indices = [int(x.strip()) - 1 for x in combo_input.split(",")]
                if all(0 <= i < len(strategy_list) for i in indices) and len(indices) >= 2:
                    break
            except ValueError:
                pass
            print(f"  Invalid. Enter at least 2 numbers between 1-{len(strategy_list)}, separated by commas.")

        # Create sub-strategies and prompt edge filter for each
        strategies = []
        edge_filters = {}
        for idx in indices:
            key, cls = strategy_list[idx]
            strat = cls(config)
            strategies.append((key, strat))
            print(f"\n  ── {strat.name} ──")
            ef = prompt_edge_filter(strat.name)
            if ef.min_edge is not None or ef.max_edge is not None:
                edge_filters[key] = ef

        combined = CombinedStrategy(config, strategies, edge_filters)
        print(f"\n  → {combined.name}")
        if edge_filters:
            for name, ef in edge_filters.items():
                print(f"    Edge filter {name}: {ef}")
        print()
        return combined

    else:
        # Single strategy mode
        key, cls = strategy_list[idx - 1]
        strategy = cls(config)
        print(f"  → Strategy: {strategy.name}")

        # Prompt edge filter
        ef = prompt_edge_filter(strategy.name)
        if ef.min_edge is not None or ef.max_edge is not None:
            # Wrap in a simple edge-filtering wrapper
            original_evaluate = strategy.evaluate

            def filtered_evaluate(prices, yes_price, window, is_mock_data=False,
                                  pool_total_bnb=0.0, pool_bull_bnb=0.0, pool_bear_bnb=0.0):
                sig = original_evaluate(
                    prices=prices, yes_price=yes_price, window=window,
                    is_mock_data=is_mock_data, pool_total_bnb=pool_total_bnb,
                    pool_bull_bnb=pool_bull_bnb, pool_bear_bnb=pool_bear_bnb,
                )
                if sig is not None and not ef.passes(sig.edge):
                    strategy.last_skip_reason = (
                        f"⏸ Edge {sig.edge:.3f} outside filter ({ef})"
                    )
                    return None
                return sig

            strategy.evaluate = filtered_evaluate

        print()
        return strategy


def select_mode_interactive(config: dict) -> tuple[str, object, object]:
    """
    Display an interactive menu and return (mode, trader, strategy).

    Returns:
        (mode, trader, strategy) where mode is "live" or "paper",
        trader is a LiveTrader or PaperTrader instance,
        and strategy is a BaseStrategy instance.
    """
    print("\n" + "=" * 40)
    print("  BNB Up/Down 5mn Trading Bot")
    print("=" * 40)
    print("  [1] 🔴 Live Trading")
    print("  [2] 📝 Paper Trading")
    print("=" * 40)

    while True:
        choice = input("Select mode: ").strip()
        if choice in ("1", "2"):
            break
        print("Invalid choice. Please enter 1 or 2.")

    # Select strategy FIRST so we can set paths before creating the trader
    strategy = select_strategy_interactive(config)

    # For CombinedStrategy, build a unique key with PID to allow parallel instances
    from strategies.combined import CombinedStrategy
    if isinstance(strategy, CombinedStrategy):
        sub_names = [s[0] for s in strategy.strategies]
        strategy_key = "combined_" + "_".join(sorted(sub_names)) + f"_{os.getpid()}"
    else:
        strategy_key = next((k for k, v in STRATEGIES.items() if isinstance(strategy, v)), "unknown")

    if choice == "1":
        config.setdefault("live_trading", {})["log_file"] = f"logs/live/{strategy_key}/{strategy_key}.json"
        mode, trader = _init_live_mode(config)
    else:
        config.setdefault("paper_trading", {})["log_file"] = f"logs/paper/{strategy_key}/{strategy_key}.json"
        trader = PaperTrader(config)
        mode = "paper"

    return mode, trader, strategy


def _init_live_mode(config: dict) -> tuple[str, object]:
    """Initialize live trading mode with confirmation."""
    from live_trader import LiveTrader

    print("\n🔴 Initializing Live Trading...")
    print("   Loading wallet and connecting to BSC...")

    try:
        trader = LiveTrader(config)
    except RuntimeError as e:
        print(f"\n❌ Live trading setup failed: {e}")
        print("   → Falling back to paper trading.\n")
        return "paper", PaperTrader(config)

    # Show wallet info and ask for confirmation
    bnb_balance = trader.get_bnb_balance()
    print(f"\n💰 Wallet: {trader._wallet_address}")
    print(f"💰 BNB Balance: {bnb_balance:.6f} BNB")
    print(f"⚠️  Auto-claim: {'enabled' if trader.auto_claim else 'disabled'}")

    print("\n🚀 Live trading starting...\n")
    return "live", trader


# ─────────────────────────────────────────────────────────────────────────────
# Main bot
# ─────────────────────────────────────────────────────────────────────────────

class PolymarketBot:
    """
    Main bot orchestrator.

    Coordinates:
    - BinanceFeed: real-time BNB price via WebSocket
    - PancakeClient: on-chain pool polling
    - Strategy: signal generation
    - PaperTrader or LiveTrader: trade execution and logging
    - Dashboard: Rich live terminal display
    """

    def __init__(self, config: dict, trader, mode: str, strategy=None):
        self.config = config
        self.logger = logging.getLogger("PolymarketBot")
        self.trader = trader
        self.mode = mode  # "paper" or "live"

        # Components
        self.binance = BinanceFeed(buffer_seconds=310)
        self.pancake = PancakeClient(
            rpc_url=config.get("pancake", {}).get("rpc_url", None),
            contract_address=config.get("pancake", {}).get("contract_address", "0x18B2A687610328590Bc8F2e5fEdDe3b582A49cdA"),
            use_mock_on_failure=config.get("pancake", {}).get("use_mock_on_failure", True),
            timeout=config.get("pancake", {}).get("timeout_seconds", 8),
        )
        if strategy is not None:
            self.strategy = strategy
        else:
            # Fallback: default GBM strategy
            self.strategy = STRATEGIES["gbm"](config)

        # Share PancakeClient with trader for on-chain PnL resolution (live + paper)
        if hasattr(trader, '_pancake_client'):
            trader._pancake_client = self.pancake

        # Share Binance feed with paper trader so it can read current BNB price
        if hasattr(trader, '_binance_feed'):
            trader._binance_feed = self.binance

        if mode == "live" and hasattr(trader, '_fix_legacy_pnl'):
            # Fix any trades with buggy PnL from old formula (live only)
            trader._fix_legacy_pnl()

        # Resolve any PENDING trades from previous sessions (both modes)
        if hasattr(trader, 'resolve_pending_on_startup'):
            trader.resolve_pending_on_startup()

        # Dashboard — imported here so Rich is only required when running the bot
        from dashboard import Dashboard
        self.dashboard = Dashboard(trader=trader, binance=self.binance, mode=mode)
        self.dashboard._strategy_name = self.strategy.name
        # Store strategy key for logging
        self._strategy_key = next(
            (k for k, v in STRATEGIES.items() if isinstance(self.strategy, v)), ""
        )

        self._running = False
        self._last_epoch: int = -1
        self._last_bnb_window_open: float | None = None
        self._traded_this_epoch = False
        self._last_round_data: PancakeRound | None = None  # cached round data
        self._last_poll_ts: float = 0.0  # when we last polled the contract
        self._in_entry_window: bool = False  # whether we're currently in entry window
        self._poll_interval = config.get("polymarket", {}).get("poll_interval_seconds", 5)
        self._min_seconds_before_lock = config.get("strategy", {}).get("min_seconds_before_lock", 4)
        self._live_ref = None  # Rich Live instance (set during run())
        self._live_close_ts: float | None = None  # close_ts of the live round (N-1)

        # Sniper phase state (per epoch)
        _scfg = config.get("strategy", {})
        self._sniper_window_seconds: float = _scfg.get("sniper_window_seconds", 7)
        self._prefetch_done: bool = False    # Phase 1 complete
        self._sniped_this_epoch: bool = False  # Phase 2 complete (sniper shot fired)
        self._sniper_tx_hash: str | None = None  # tx hash from fire_transaction (live mode)

    def _refresh_display(self):
        """Push a fresh render to the Live display if it's active."""
        if self._live_ref is not None:
            try:
                self._live_ref.update(self.dashboard.render())
            except Exception:
                pass

    def _on_price(self, pp: PricePoint):
        """Called on each new Binance trade price."""
        self.logger.debug(f"BNB price: {pp.price:.2f}")

    async def _run_main_loop(self):
        """Main async loop: poll Polymarket + evaluate strategy."""
        self.logger.info("Starting main strategy loop...")

        while self._running:
            try:
                await self._tick()
            except Exception as e:
                self.logger.error(f"Error in main loop tick: {e}", exc_info=True)

            await asyncio.sleep(self._poll_interval)

    async def _update_live_balance(self):
        """Periodically refresh cached live wallet balance (non-blocking)."""
        while self._running:
            if self.mode == "live":
                try:
                    loop = asyncio.get_event_loop()
                    balance = await loop.run_in_executor(None, self.trader.get_bnb_balance)
                    self.dashboard._cached_bnb_balance = balance
                    self.dashboard._last_balance_update = time.time()
                except Exception as e:
                    self.logger.debug(f"Balance refresh failed: {e}")
            await asyncio.sleep(30)

    async def _update_chainlink_price(self):
        """Periodically fetch BNB/USD price from Chainlink oracle on BSC (same as PancakeSwap)."""
        while self._running:
            try:
                loop = asyncio.get_event_loop()
                price = await loop.run_in_executor(None, self.pancake.get_chainlink_bnb_price)
                if price is not None:
                    self.dashboard._chainlink_bnb_price = price
            except Exception as e:
                self.logger.debug(f"Chainlink price refresh failed: {e}")
            await asyncio.sleep(10)

    async def _tick(self):
        """
        One iteration of the main loop.

        Polling strategy to minimize RPC calls:
        - When outside the entry window: poll every ~30s to detect epoch change.
        - When inside the entry window (<60s before lock): poll every tick (5s).
        """
        now = time.time()
        loop = asyncio.get_event_loop()

        # Determine if we should poll the contract this tick.
        # In entry window → always poll. Outside → every 30s.
        poll_interval_outside = 30.0
        should_poll = (
            self._in_entry_window
            or self._last_round_data is None
            or (now - self._last_poll_ts) >= poll_interval_outside
        )

        round_data = self._last_round_data

        if should_poll:
            fetched = await loop.run_in_executor(None, self.pancake.get_current_round)
            if fetched is not None:
                round_data = fetched
                self._last_round_data = round_data
                self._last_poll_ts = now
            elif round_data is None:
                self.logger.warning("Could not get PancakeSwap round — skipping this tick.")
                self.dashboard.update_status("⚠️ PancakeSwap unavailable")
                self._refresh_display()
                return

        # Detect epoch change
        current_epoch = round_data.epoch
        if current_epoch != self._last_epoch:
            self._on_new_epoch(current_epoch, round_data)

        # Calculate time until lock_ts (end of bet phase)
        seconds_to_lock = round_data.lock_ts - now
        self._in_entry_window = 0 < seconds_to_lock <= self.strategy.entry_window_seconds

        # Build WindowInfo from contract round data
        window = window_from_round(round_data, entry_window_seconds=self.strategy.entry_window_seconds)

        # Update dashboard window info
        self.dashboard.update_window(window)

        # Update round card with current round/signal data
        self.dashboard.update_round_info(
            epoch=current_epoch,
            lock_price=getattr(round_data, 'lock_price', None),
            pool_total_bnb=getattr(round_data, 'total_bnb', None),
            p_up=None,  # will be set after p_up_display is computed below
            edge=None,  # will be set if signal fires
            signal_side=None if not self._traded_this_epoch else self.dashboard._round_signal_side,
            bet_bnb=None if not self._traded_this_epoch else self.dashboard._round_bet_bnb,
            bet_usdc=None if not self._traded_this_epoch else self.dashboard._round_bet_usdc,
        )

        # Periodic status log (every 30s) — file only, dashboard shows live
        if int(now) % 30 == 0:
            bnb = self.binance.last_price
            if hasattr(self.trader, 'metrics'):
                bankroll_str = f"${self.trader.metrics.bankroll:.2f}" if hasattr(self.trader.metrics, 'bankroll') else "N/A"
            else:
                bankroll_str = "N/A"
            phase = "BET OPEN" if seconds_to_lock > self.strategy.entry_window_seconds else (
                "EVALUATING" if seconds_to_lock > 0 else "LOCKED"
            )
            self.logger.info(
                f"Epoch #{current_epoch} | Phase: {phase} | "
                f"Lock in: {seconds_to_lock:.0f}s | "
                f"BNB: {f'{bnb:.2f}' if bnb is not None else 'N/A'} | "
                f"Balance: {bankroll_str}"
            )

        # Get price series from round's start_ts (used in all phases)
        window_prices_raw = self.binance.get_window_prices(round_data.start_ts)
        prices = [pp.price for pp in window_prices_raw]

        # ── Phase 0: Outside entry window → wait ─────────────────────────────
        if not self._in_entry_window:
            if seconds_to_lock <= 0:
                self.dashboard.update_status("🔒 Round locked")
                # Phase 3: Verify fired TX (live mode only, after lock)
                if (
                    self.mode == "live"
                    and self._sniper_tx_hash is not None
                    and not getattr(self, '_sniper_verified', False)
                    and hasattr(self.trader, 'verify_transaction')
                ):
                    result = self.trader.verify_transaction(timeout=15)
                    self._sniper_verified = True
                    if result is True:
                        self.dashboard.log(f"✅ Sniper TX confirmed: {self._sniper_tx_hash}")
                    elif result is False:
                        self.dashboard.log(f"❌ Sniper TX FAILED: {self._sniper_tx_hash}")
                    else:
                        self.dashboard.log(f"⏳ Sniper TX pending: {self._sniper_tx_hash}")
            else:
                self.dashboard.update_status("⏳ Waiting for entry window")
            self._refresh_display()
            return

        if not prices:
            self.logger.debug("No prices in epoch window yet.")
            self._refresh_display()
            return

        yes_price_equiv = round_data.yes_price_equiv

        # When use_fair_odds=True, pool price is logged for monitoring only
        use_fair_odds = self.config.get("strategy", {}).get("use_fair_odds", True)
        pool_tag = " [INFO ONLY — fair_odds=0.50]" if use_fair_odds else ""
        self.logger.info(
            f"Tick: {len(prices)} prices | "
            f"BNB: {prices[-1]:.2f} | "
            f"Pool: bull={round_data.bull_ratio:.1%} bear={round_data.bear_ratio:.1%} "
            f"({round_data.total_bnb:.3f} BNB) | "
            f"Pool YES≡: {yes_price_equiv:.3f}{pool_tag} | "
            f"Lock in: {seconds_to_lock:.0f}s"
            + (" [MOCK]" if round_data.is_mock else "")
        )

        # Compute P(close > lock) for dashboard display (before strategy filters)
        from strategy import estimate_p_up_momentum
        p_up_display = estimate_p_up_momentum(
            prices=prices,
            seconds_to_lock=seconds_to_lock,
            round_duration=300.0,
        ) if len(prices) >= 5 else None

        # Update round card p_up
        self.dashboard.update_round_info(
            epoch=current_epoch,
            lock_price=getattr(round_data, 'lock_price', None),
            pool_total_bnb=getattr(round_data, 'total_bnb', None),
            p_up=p_up_display,
            edge=self.dashboard._round_edge,
            signal_side=self.dashboard._round_signal_side,
            bet_bnb=self.dashboard._round_bet_bnb,
            bet_usdc=self.dashboard._round_bet_usdc,
        )

        # ── Phase 1: Pre-load (T-sniper_window_seconds > seconds_to_lock > sniper_window) ──
        # Pre-fetch all slow data when we're in the outer part of the entry window
        # (between entry_window_seconds and sniper_window_seconds before lock)
        if (
            not self._prefetch_done
            and not self._traded_this_epoch
            and seconds_to_lock > self._sniper_window_seconds
        ):
            self.dashboard.update_status("📡 Pre-loading strategy data...")
            self.dashboard.log(
                f"Epoch #{current_epoch} | Lock in {seconds_to_lock:.0f}s | "
                f"Phase 1: pre-loading data..."
            )
            self._refresh_display()

            # Run prefetch in executor so it doesn't block the event loop
            try:
                await loop.run_in_executor(
                    None,
                    lambda: self.strategy.prefetch(prices, epoch=current_epoch)
                )
            except Exception as e:
                self.logger.warning(f"Prefetch failed (non-fatal): {e}")

            # For live mode: pre-build and pre-sign both transactions
            if self.mode == "live" and hasattr(self.trader, 'prepare_transactions'):
                bnb_price = self.binance.last_price or 600.0
                bet_usdc = self.config.get("strategy", {}).get("max_position_usdc", 15.0)
                bet_bnb = bet_usdc / bnb_price
                try:
                    ok = await loop.run_in_executor(
                        None,
                        lambda: self.trader.prepare_transactions(current_epoch, bet_bnb)
                    )
                    if ok:
                        self.logger.info(
                            f"✅ Both TXs pre-signed for epoch {current_epoch} | "
                            f"bet={bet_bnb:.6f} BNB"
                        )
                except Exception as e:
                    self.logger.warning(f"TX pre-signing failed (non-fatal): {e}")

            self._prefetch_done = True
            self.dashboard.update_status("📡 Data pre-loaded — awaiting sniper window")
            self._refresh_display()
            return

        # ── Phase 2: Sniper shot (seconds_to_lock <= sniper_window_seconds) ────
        # ONE single poll + evaluate + fire as close to lock as possible
        if (
            not self._sniped_this_epoch
            and not self._traded_this_epoch
            and seconds_to_lock <= self._sniper_window_seconds
            and seconds_to_lock >= self._min_seconds_before_lock
        ):
            self.dashboard.update_status("🎯 SNIPER — taking final snapshot...")
            self._refresh_display()

            # Fresh on-chain poll for latest pool snapshot
            fresh_round = await loop.run_in_executor(None, self.pancake.get_current_round)
            if fresh_round is not None:
                round_data = fresh_round
                self._last_round_data = round_data
                self._last_poll_ts = now

            # Re-fetch latest prices for sniper window
            window_prices_raw = self.binance.get_window_prices(round_data.start_ts)
            prices = [pp.price for pp in window_prices_raw]

            if not prices:
                self.logger.warning("Sniper: no prices available — aborting sniper shot")
                self.dashboard.update_status("⚠️ Sniper: no prices")
                self._refresh_display()
                return

            # Recalculate seconds_to_lock with fresh data
            seconds_to_lock_fresh = round_data.lock_ts - time.time()

            # Rebuild window with fresh round data
            window = window_from_round(round_data, entry_window_seconds=self.strategy.entry_window_seconds)

            # Log evaluation to dashboard
            p_up_str = f"{p_up_display:.2f}" if p_up_display is not None else "?"
            self.dashboard.log(
                f"🎯 SNIPER Epoch #{current_epoch} | Lock in {seconds_to_lock_fresh:.1f}s | "
                f"P(Up)={p_up_str} | Pool: {round_data.total_bnb:.1f} BNB "
                f"(bull {round_data.bull_ratio:.0%} / bear {round_data.bear_ratio:.0%})"
                + (" [MOCK]" if round_data.is_mock else "")
            )

            signal = self.strategy.evaluate(
                prices=prices,
                yes_price=round_data.yes_price_equiv,
                window=window,
                is_mock_data=round_data.is_mock,
                pool_total_bnb=round_data.total_bnb,
                pool_bull_bnb=round_data.bull_bnb,
                pool_bear_bnb=round_data.bear_bnb,
            )

            self._sniped_this_epoch = True  # Only attempt once per epoch

            if signal:
                # Safety guard: don't bet if too close to lock
                if seconds_to_lock_fresh < self._min_seconds_before_lock:
                    msg = f"⏱ Sniper too late ({seconds_to_lock_fresh:.1f}s < {self._min_seconds_before_lock}s min) — skipping"
                    self.logger.warning(msg)
                    self.dashboard.log(msg)
                    self.dashboard.update_status("⏱ Too late to bet")
                    self._refresh_display()
                    return

                self.dashboard.log(
                    f"🎯 Signal: {signal.side} @ edge={signal.edge:.2f} | P(Up)={signal.p_up:.2f} | ${signal.position_size_usdc:.2f}"
                )

                if self.mode == "live" and hasattr(self.trader, 'fire_transaction'):
                    # Fire the pre-signed transaction immediately (fire-and-forget)
                    self.dashboard.update_status(f"🚀 Firing {signal.side} TX...")
                    self._refresh_display()

                    tx_hash = await loop.run_in_executor(
                        None,
                        lambda: self.trader.fire_transaction(signal.side)
                    )

                    if tx_hash:
                        self._sniper_tx_hash = tx_hash
                        self._sniper_verified = False
                        self._traded_this_epoch = True

                        # Record the trade (without waiting for receipt)
                        bnb_price = self.binance.last_price or 600.0
                        bet_bnb = signal.position_size_usdc / bnb_price
                        from live_trader import LiveTrade
                        import time as _time
                        self._trade_counter = getattr(self, '_trade_counter', 0) + 1
                        trade_id = f"LT-{int(_time.time())}-{self._trade_counter:04d}"
                        entry_price = signal.yes_price if signal.side == "YES" else (1.0 - signal.yes_price)
                        trade = LiveTrade(
                            trade_id=trade_id,
                            timestamp_entry=_time.time(),
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
                            epoch=current_epoch,
                            is_mock=signal.is_mock,
                            bull_pct=signal.bull_pct,
                            bear_pct=signal.bear_pct,
                            tx_hash=tx_hash,
                            tx_status="pending",
                            outcome="PENDING",
                        )
                        self.trader._trades.append(trade)
                        self.trader._pending_trades.append(trade)
                        self.trader.metrics.total_trades += 1
                        self.trader.metrics.pending += 1
                        self.trader.metrics.total_wagered += trade.position_size_usdc
                        self.trader._save_trades()

                        self.dashboard.update_round_info(
                            epoch=current_epoch,
                            lock_price=getattr(round_data, 'lock_price', None),
                            pool_total_bnb=getattr(round_data, 'total_bnb', None),
                            p_up=p_up_display,
                            edge=signal.edge,
                            signal_side=signal.side,
                            bet_bnb=bet_bnb,
                            bet_usdc=signal.position_size_usdc,
                        )
                        self.dashboard.update_status(f"🚀 TX fired: {signal.side} | hash={tx_hash[:12]}...")
                        self.dashboard.log(f"🚀 Sniper TX fired: bet{signal.side.capitalize()} epoch #{current_epoch} | tx={tx_hash[:16]}...")
                    else:
                        # fire_transaction failed — fall back to normal enter_trade
                        self.logger.warning("fire_transaction failed — falling back to enter_trade()")
                        self.dashboard.update_status(f"⚠️ Fire failed — using enter_trade...")
                        self._refresh_display()
                        trade_result = self.trader.enter_trade(signal, window)
                        self._traded_this_epoch = True
                        bet_bnb_display = getattr(trade_result, 'bet_bnb', None) if trade_result else signal.position_size_usdc / (self.binance.last_price or 600)
                        self.dashboard.update_round_info(
                            epoch=current_epoch,
                            lock_price=getattr(round_data, 'lock_price', None),
                            pool_total_bnb=getattr(round_data, 'total_bnb', None),
                            p_up=p_up_display,
                            edge=signal.edge,
                            signal_side=signal.side,
                            bet_bnb=bet_bnb_display,
                            bet_usdc=signal.position_size_usdc,
                        )
                        self.dashboard.update_status(f"✅ Trade entered: {signal.side}")
                        self.dashboard.log(f"✅ Trade entered: bet{signal.side.capitalize()} epoch #{current_epoch}")
                else:
                    # Paper trading: use normal enter_trade with fresh snapshot
                    self.dashboard.update_status(f"🎯 Entering {signal.side} trade (paper)...")
                    self._refresh_display()
                    trade_result = self.trader.enter_trade(signal, window)
                    self._traded_this_epoch = True

                    bet_bnb_display = getattr(trade_result, 'bet_bnb', None) if trade_result else signal.position_size_usdc / (self.binance.last_price or 600)
                    self.dashboard.update_round_info(
                        epoch=current_epoch,
                        lock_price=getattr(round_data, 'lock_price', None),
                        pool_total_bnb=getattr(round_data, 'total_bnb', None),
                        p_up=p_up_display,
                        edge=signal.edge,
                        signal_side=signal.side,
                        bet_bnb=bet_bnb_display,
                        bet_usdc=signal.position_size_usdc,
                    )
                    self.dashboard.update_status(f"✅ Trade entered: {signal.side}")
                    self.dashboard.log(f"✅ Trade entered: bet{signal.side.capitalize()} epoch #{current_epoch}")
            else:
                skip_reason = getattr(self.strategy, "last_skip_reason", None)
                msg = skip_reason or "🔍 No signal at sniper time"
                self.dashboard.log(f"Sniper: {msg}")
                self.dashboard.update_status(msg)

            self._refresh_display()
            return

        # ── In entry window but not yet in sniper zone: show status ─────────
        if self._traded_this_epoch or self._sniped_this_epoch:
            if self._sniped_this_epoch and not self._traded_this_epoch:
                self.dashboard.update_status("⏸ Sniper fired — no signal")
            else:
                self.dashboard.update_status("⏸ Already traded this epoch")
        elif seconds_to_lock > self._sniper_window_seconds:
            phase_str = "📡 Pre-loaded" if self._prefetch_done else "🔍 Evaluating signal"
            p_up_str = f"{p_up_display:.2f}" if p_up_display is not None else "?"
            self.dashboard.log(
                f"Epoch #{current_epoch} | Lock in {seconds_to_lock:.0f}s | "
                f"P(Up)={p_up_str} | Pool: {round_data.total_bnb:.1f} BNB "
                f"(bull {round_data.bull_ratio:.0%} / bear {round_data.bear_ratio:.0%})"
                + (" [MOCK]" if round_data.is_mock else "")
            )
            self.dashboard.update_status(f"{phase_str} | Sniper in {seconds_to_lock - self._sniper_window_seconds:.0f}s")
        else:
            self.dashboard.update_status("🎯 Awaiting sniper window...")

        self._refresh_display()

    def _on_new_epoch(self, new_epoch: int, round_data: PancakeRound):
        """
        Called when currentEpoch() advances to a new value.

        Epoch lifecycle when we detect epoch N as current:
          - Epoch N    → currently in BET phase (lock_ts in the future)
          - Epoch N-1  → just locked (no more bets), waiting for oracle close
          - Epoch N-2  → just CLOSED (oracle called) → resolve trades for this epoch

        We resolve epoch N-2 using prices from its start_ts to close_ts.
        """
        if self._last_epoch > 0:
            # Epoch that just closed = last_epoch - 1
            resolve_epoch = self._last_epoch - 1

            # Fetch round data for the closed epoch to get its time range
            loop = None
            try:
                closed_round = self.pancake.get_round_by_epoch(resolve_epoch)
            except Exception as e:
                self.logger.warning(f"Could not fetch data for closed epoch {resolve_epoch}: {e}")
                closed_round = None

            if closed_round is not None:
                # Use Chainlink oracle prices from the contract (same as PancakeSwap settlement)
                bnb_open = closed_round.lock_price
                bnb_close = closed_round.close_price

                if bnb_open is not None and bnb_close is not None and closed_round.oracle_called:
                    direction = "UP" if bnb_close > bnb_open else "DOWN"
                    pct_change = abs(bnb_close / bnb_open - 1)
                    self.logger.info(
                        f"🔚 Epoch #{resolve_epoch} closed (Chainlink): "
                        f"BNB {bnb_open:.4f} → {bnb_close:.4f} "
                        f"({direction} {pct_change:.3%})"
                    )
                    self.dashboard.log(
                        f"🔚 Round #{resolve_epoch} result: BNB {direction} {pct_change:.2%}"
                    )
                    self.trader.resolve_trades(resolve_epoch, bnb_open, bnb_close)
                else:
                    self.logger.warning(
                        f"Epoch #{resolve_epoch}: oracle not yet called or prices missing "
                        f"(lock={bnb_open}, close={bnb_close}, oracle={closed_round.oracle_called})"
                    )
                    self.dashboard.log(
                        f"⚠️ Epoch #{resolve_epoch}: oracle not settled — will retry"
                    )
            else:
                self.logger.warning(
                    f"Could not fetch closed round data for epoch #{resolve_epoch} — skipping resolution."
                )
                self.dashboard.log(
                    f"⚠️ Epoch #{resolve_epoch}: round data unavailable — cannot resolve"
                )

        # ── Update LIVE round card (epoch N-1 just locked) ──
        live_epoch = self._last_epoch  # the epoch that just locked becomes the live round
        if live_epoch > 0:
            try:
                live_round = self.pancake.get_round_by_epoch(live_epoch)
            except Exception as e:
                self.logger.warning(f"Could not fetch live round data for epoch {live_epoch}: {e}")
                live_round = None

            live_lock_price = None
            live_close_ts = None
            if live_round is not None:
                live_lock_price = live_round.lock_price
                live_close_ts = float(live_round.close_ts)
                self._live_close_ts = live_close_ts

            # Check if we have a pending/recent trade for this epoch
            live_signal_side = None
            live_bet_bnb = None
            live_bet_usdc = None
            live_edge = None

            # Look in all trades (pending + resolved) for this epoch
            all_trades = list(getattr(self.trader, '_trades', []))
            for t in reversed(all_trades):
                if getattr(t, 'epoch', None) == live_epoch:
                    live_signal_side = getattr(t, 'side', None)
                    live_bet_bnb = getattr(t, 'bet_bnb', None)
                    live_bet_usdc = getattr(t, 'position_size_usdc', None)
                    live_edge = getattr(t, 'edge_at_entry', None)
                    break

            self.dashboard.update_live_round(
                epoch=live_epoch,
                lock_price=live_lock_price,
                close_ts=live_close_ts,
                signal_side=live_signal_side,
                bet_bnb=live_bet_bnb,
                bet_usdc=live_bet_usdc,
                edge=live_edge,
            )

        # Advance to new epoch
        self._last_epoch = new_epoch
        self._traded_this_epoch = False
        self._in_entry_window = False
        self._prefetch_done = False
        self._sniped_this_epoch = False
        self._sniper_tx_hash = None

        # Reset NEXT round card for the new epoch
        self.dashboard.update_round_info(
            epoch=new_epoch,
            lock_price=getattr(round_data, 'lock_price', None),
            pool_total_bnb=getattr(round_data, 'total_bnb', None),
            p_up=None,
            edge=None,
            signal_side=None,
            bet_bnb=None,
            bet_usdc=None,
        )

        bnb = self.binance.last_price
        self._last_bnb_window_open = bnb
        bnb_str = f"${bnb:.2f}" if bnb is not None else "N/A"

        self.logger.info(
            f"🆕 Epoch #{new_epoch} started | BNB: {bnb_str} | "
            f"Lock at: {round_data.lock_ts:.0f} (in {round_data.lock_ts - time.time():.0f}s)"
        )
        self.dashboard.log(
            f"🎲 Betting open: round #{new_epoch} | BNB: {bnb_str} | "
            f"Lock in {round_data.lock_ts - time.time():.0f}s"
        )
        self.dashboard.update_status("⏳ Waiting for entry window")

        # Update strategy bankroll
        if self.mode == "live":
            # LiveMetrics.bankroll is never populated from on-chain balance → use
            # starting_bankroll_usdc + cumulative PnL as the effective bankroll.
            starting = self.config.get("strategy", {}).get("starting_bankroll_usdc", 1000.0)
            effective_bankroll = starting + self.trader.metrics.total_pnl
            self.strategy.update_bankroll(effective_bankroll)
        elif hasattr(self.trader.metrics, 'bankroll'):
            self.strategy.update_bankroll(self.trader.metrics.bankroll)

        self._refresh_display()

    async def run(self):
        """Start the bot: Binance feed + main loop in parallel, inside Rich Live."""
        self._running = True

        # Set up graceful shutdown on Ctrl+C
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self.stop)

        # Re-configure logging: reduce console to WARNING now that dashboard is active
        setup_logging(self.config, dashboard_mode=True, strategy_key=self._strategy_key, trading_mode=self.mode)

        # Start the Rich Live display
        live = self.dashboard.start()
        live.start()
        self._live_ref = live

        try:
            self.logger.info("="*60)
            self.logger.info(f"  BNB Up/Down 5mn Trading Bot [{self.mode.upper()}]")
            self.logger.info("="*60)

            # Test PancakeSwap / BSC connectivity
            self.dashboard.log("Testing PancakeSwap/BSC connectivity...")
            self._refresh_display()

            loop = asyncio.get_event_loop()
            api_ok = await loop.run_in_executor(None, self.pancake.check_connectivity)
            if api_ok:
                self.logger.info("✅ PancakeSwap: BSC reachable — live on-chain data")
                self.dashboard.log("✅ PancakeSwap connected — live on-chain data")
            else:
                self.logger.warning(
                    "⚠️  PancakeSwap: BSC unreachable → Running in MOCK DATA mode."
                )
                self.dashboard.log("⚠️  PancakeSwap unreachable — MOCK DATA mode")

            self.trader.print_summary()
            self._refresh_display()

            # Run Binance feed + main loop + balance refresh concurrently
            self.binance.on_price = self._on_price
            await asyncio.gather(
                self.binance.run(),
                self._run_main_loop(),
                self._update_live_balance(),
                self._update_chainlink_price(),
            )
        finally:
            live.stop()
            self._live_ref = None

    def stop(self):
        """Graceful shutdown."""
        self.logger.info("Shutting down bot...")
        self._running = False
        self.binance.stop()
        self.trader.print_summary()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BNB Up/Down 5mn Trading Bot")
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to config.json (default: config.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test connectivity only, do not start the bot",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Paper trading: reset history and start from scratch",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Paper trading mode (continue from existing data)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Live trading mode (real on-chain transactions)",
    )
    parser.add_argument(
        "--strategy",
        default="gbm",
        choices=list(STRATEGIES.keys()),
        help="Trading strategy to use (default: gbm)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)

    if args.dry_run:
        logging.info("Dry run mode — testing PancakeSwap/BSC connectivity.")
        client = PancakeClient(
            rpc_url=config.get("pancake", {}).get("rpc_url", None),
        )
        ok = client.check_connectivity()
        round_data = client.get_current_round()
        logging.info(f"Sample PancakeSwap round: {round_data}")
        sys.exit(0 if ok else 1)

    # Determine strategy first (needed for per-strategy file paths)
    if args.live or args.fresh or args.paper:
        strategy_key = args.strategy
        strategy_cls = STRATEGIES.get(strategy_key, STRATEGIES["gbm"])
        strategy = strategy_cls(config)
    else:
        strategy = None  # will be set by interactive menu
        strategy_key = None

    # Helper: inject per-strategy file paths into config to isolate parallel runs
    def _apply_strategy_paths(cfg: dict, s_key: str):
        """Rewrite trade log paths to include strategy key for isolation."""
        if s_key:
            cfg.setdefault("paper_trading", {})["log_file"] = f"logs/paper/{s_key}/{s_key}.json"
            cfg.setdefault("live_trading", {})["log_file"] = f"logs/live/{s_key}/{s_key}.json"

    # Determine trader based on CLI args or interactive menu
    if args.live:
        _apply_strategy_paths(config, strategy_key)
        from live_trader import LiveTrader
        try:
            trader = LiveTrader(config)
            mode = "live"
        except RuntimeError as e:
            logging.error(f"Live trading setup failed: {e}")
            sys.exit(1)
    elif args.fresh:
        _apply_strategy_paths(config, strategy_key)
        reset_paper_trades(config)
        trader = PaperTrader(config)
        mode = "paper"
    elif args.paper:
        _apply_strategy_paths(config, strategy_key)
        trader = PaperTrader(config)
        mode = "paper"
    else:
        # Interactive menu — strategy paths are set inside before trader creation
        mode, trader, strategy = select_mode_interactive(config)
        from strategies.combined import CombinedStrategy
        if isinstance(strategy, CombinedStrategy):
            sub_names = [s[0] for s in strategy.strategies]
            strategy_key = "combined_" + "_".join(sorted(sub_names)) + f"_{os.getpid()}"
        else:
            strategy_key = next((k for k, v in STRATEGIES.items() if isinstance(strategy, v)), "unknown")

    # Reconfigure logging with strategy- and mode-specific log dir
    if strategy_key:
        setup_logging(config, strategy_key=strategy_key, trading_mode=mode)

    bot = PolymarketBot(config, trader, mode=mode, strategy=strategy)

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
