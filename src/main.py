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
from strategy import Strategy, WindowInfo, window_from_round
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


def setup_logging(config: dict, dashboard_mode: bool = False):
    """Configure logging from config.

    When dashboard_mode=True, reduces console handler to WARNING level
    (dashboard is the primary display) but keeps file logging at full level.
    Log files are written as data/logs/run-YYYY-MM-DD.log (one per day).
    Files older than 30 days are deleted automatically.
    """
    import datetime
    cfg = config.get("logging", {})
    level_str = cfg.get("level", "INFO")
    level = getattr(logging, level_str, logging.INFO)
    fmt = cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Daily log file: data/logs/run-YYYY-MM-DD.log
    log_dir = "data/logs"
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
    log_file = config.get("paper_trading", {}).get("log_file", "data/paper_trades.json")
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


def select_mode_interactive(config: dict) -> tuple[str, object]:
    """
    Display an interactive menu and return (mode, trader).

    Returns:
        (mode, trader) where mode is "live" or "paper"
        and trader is a LiveTrader or PaperTrader instance.
    """
    print("\n" + "=" * 40)
    print("  BNB Up/Down 5mn Trading Bot")
    print("=" * 40)
    print("  [1] 🔴 Live Trading")
    print("  [2] 📝 Paper Trading (fresh run)")
    print("  [3] 📝 Paper Trading (continue)")
    print("=" * 40)

    while True:
        choice = input("Select mode: ").strip()
        if choice in ("1", "2", "3"):
            break
        print("Invalid choice. Please enter 1, 2, or 3.")

    if choice == "1":
        return _init_live_mode(config)
    elif choice == "2":
        reset_paper_trades(config)
        trader = PaperTrader(config)
        return "paper", trader
    else:
        trader = PaperTrader(config)
        return "paper", trader


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
    print(f"⚠️  Max daily loss: ${trader.max_daily_loss_usdc:.2f}")
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

    def __init__(self, config: dict, trader, mode: str):
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
        self.strategy = Strategy(config)

        # Dashboard — imported here so Rich is only required when running the bot
        from dashboard import Dashboard
        self.dashboard = Dashboard(trader=trader, binance=self.binance, mode=mode)

        self._running = False
        self._last_epoch: int = -1
        self._last_bnb_window_open: float | None = None
        self._traded_this_epoch = False
        self._last_round_data: PancakeRound | None = None  # cached round data
        self._last_poll_ts: float = 0.0  # when we last polled the contract
        self._in_entry_window: bool = False  # whether we're currently in entry window
        self._poll_interval = config.get("polymarket", {}).get("poll_interval_seconds", 5)
        self._live_ref = None  # Rich Live instance (set during run())

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

        # Only evaluate in the entry window (last N seconds before lock)
        if not self._in_entry_window:
            if seconds_to_lock <= 0:
                self.dashboard.update_status("🔒 Round locked")
            else:
                self.dashboard.update_status("⏳ Waiting for entry window")
            self._refresh_display()
            return

        self.dashboard.update_status("🔍 Evaluating signal")

        # Get price series from round's start_ts (not clock-aligned window)
        window_prices_raw = self.binance.get_window_prices(round_data.start_ts)
        prices = [pp.price for pp in window_prices_raw]

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

        # Compute P(Up) for dashboard display (before strategy filters)
        from strategy import estimate_p_up_momentum
        p_up_display = estimate_p_up_momentum(
            prices=prices,
            window_seconds=300.0,
            seconds_remaining=seconds_to_lock,
        ) if len(prices) >= 5 else None

        # Log evaluation to dashboard
        p_up_str = f"{p_up_display:.2f}" if p_up_display is not None else "?"
        self.dashboard.log(
            f"Epoch #{current_epoch} | Lock in {seconds_to_lock:.0f}s | "
            f"P(Up)={p_up_str} | Pool: {round_data.total_bnb:.1f} BNB "
            f"(bull {round_data.bull_ratio:.0%} / bear {round_data.bear_ratio:.0%})"
            + (" [MOCK]" if round_data.is_mock else "")
        )

        signal = self.strategy.evaluate(
            prices=prices,
            yes_price=yes_price_equiv,
            window=window,
            is_mock_data=round_data.is_mock,
            pool_total_bnb=round_data.total_bnb,
            pool_bull_bnb=round_data.bull_bnb,
            pool_bear_bnb=round_data.bear_bnb,
        )

        if signal:
            if not self._traded_this_epoch:
                # Log signal to dashboard
                self.dashboard.log(
                    f"🎯 Signal: {signal.side} @ edge={signal.edge:.2f} | P(Up)={signal.p_up:.2f} | ${signal.position_size_usdc:.2f}"
                )
                self.dashboard.update_status(f"🎯 Entering {signal.side} trade...")
                self._refresh_display()

                # Enter trade (one trade per epoch max)
                self.trader.enter_trade(signal, window)
                self._traded_this_epoch = True

                self.dashboard.update_status(f"✅ Trade entered: {signal.side}")
                self.dashboard.log(
                    f"✅ Trade entered: bet{signal.side.capitalize()} epoch #{current_epoch}"
                )
            else:
                self.dashboard.update_status("⏸ Already traded this epoch")
        else:
            skip_reason = getattr(self.strategy, "last_skip_reason", None)
            if skip_reason:
                self.dashboard.log(skip_reason)
                self.dashboard.update_status(skip_reason)
            else:
                self.dashboard.update_status("🔍 Evaluating signal")

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
                # Use prices from that epoch's start_ts to close_ts
                epoch_prices = self.binance.get_window_prices(
                    closed_round.start_ts,
                    closed_round.close_ts,
                )
                if epoch_prices:
                    bnb_open = epoch_prices[0].price
                    bnb_close = epoch_prices[-1].price
                    direction = "UP" if bnb_close > bnb_open else "DOWN"
                    pct_change = abs(bnb_close / bnb_open - 1)
                    self.logger.info(
                        f"🔚 Epoch #{resolve_epoch} closed: "
                        f"BNB {bnb_open:.2f} → {bnb_close:.2f} "
                        f"({direction} {pct_change:.3%})"
                    )
                    self.dashboard.log(
                        f"🔚 Round #{resolve_epoch} result: BNB {direction} {pct_change:.2%}"
                    )
                    self.trader.resolve_trades(resolve_epoch, bnb_open, bnb_close)
                else:
                    # Fallback: use prices from a window ending at close_ts
                    fallback_start = closed_round.start_ts
                    fallback_prices = self.binance.get_window_prices(fallback_start)
                    if fallback_prices:
                        bnb_open = fallback_prices[0].price
                        bnb_close = fallback_prices[-1].price
                        direction = "UP" if bnb_close > bnb_open else "DOWN"
                        pct_change = abs(bnb_close / bnb_open - 1)
                        self.logger.info(
                            f"🔚 Epoch #{resolve_epoch} closed (fallback prices): "
                            f"BNB {bnb_open:.2f} → {bnb_close:.2f} "
                            f"({direction} {pct_change:.3%})"
                        )
                        self.dashboard.log(
                            f"🔚 Round #{resolve_epoch} result (approx): BNB {direction} {pct_change:.2%}"
                        )
                        self.trader.resolve_trades(resolve_epoch, bnb_open, bnb_close)
                    else:
                        self.logger.warning(
                            f"No price data for epoch #{resolve_epoch} — cannot resolve trades."
                        )
                        self.dashboard.log(
                            f"⚠️ Epoch #{resolve_epoch}: no price data — cannot resolve"
                        )
            else:
                self.logger.warning(
                    f"Could not fetch closed round data for epoch #{resolve_epoch} — skipping resolution."
                )
                self.dashboard.log(
                    f"⚠️ Epoch #{resolve_epoch}: round data unavailable — cannot resolve"
                )

        # Advance to new epoch
        self._last_epoch = new_epoch
        self._traded_this_epoch = False
        self._in_entry_window = False

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
        setup_logging(self.config, dashboard_mode=True)

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

    # Determine trader based on CLI args or interactive menu
    if args.live:
        # Non-interactive live mode
        from live_trader import LiveTrader
        try:
            trader = LiveTrader(config)
            mode = "live"
        except RuntimeError as e:
            logging.error(f"Live trading setup failed: {e}")
            sys.exit(1)
    elif args.fresh:
        reset_paper_trades(config)
        trader = PaperTrader(config)
        mode = "paper"
    elif args.paper:
        trader = PaperTrader(config)
        mode = "paper"
    else:
        # Interactive menu (plain print — before Rich Live starts)
        mode, trader = select_mode_interactive(config)

    bot = PolymarketBot(config, trader, mode=mode)

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
