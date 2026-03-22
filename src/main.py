"""
main.py — Entry point for PRDT Finance BTC 5mn paper trading bot

Architecture:
  - asyncio event loop runs the Binance WebSocket feed (async)
  - Main loop polls PRDT contract every N seconds (sync via executor)
  - Strategy evaluated in the last 60s of each window
  - PaperTrader logs and resolves simulated trades

Key difference vs Polymarket:
  - No order book YES/NO price — reads on-chain pool (bull/bear amounts)
  - "yes_price equivalent" = bull_ratio / (1 - fee) (break-even probability)
  - Same Kelly/edge logic applies on top of this

Run with:
  python src/main.py
  python src/main.py --config config.json
  python src/main.py --dry-run
  python src/main.py --fresh    (reset paper trading history)
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
from strategy import Strategy, get_current_window
from paper_trader import PaperTrader


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(config: dict):
    """Configure logging from config."""
    cfg = config.get("logging", {})
    level_str = cfg.get("level", "INFO")
    level = getattr(logging, level_str, logging.INFO)
    fmt = cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    try:
        import colorlog
        handler = colorlog.StreamHandler()
        handler.setFormatter(colorlog.ColoredFormatter(
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
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt))

    logging.basicConfig(level=level, handlers=[handler], force=True)


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
# Main bot
# ─────────────────────────────────────────────────────────────────────────────

class PolymarketBot:
    """
    Main bot orchestrator.

    Coordinates:
    - BinanceFeed: real-time BTC price via WebSocket
    - PolymarketClient: order book polling
    - Strategy: signal generation
    - PaperTrader: trade simulation and logging
    """

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger("PolymarketBot")

        # Components
        self.binance = BinanceFeed(buffer_seconds=310)
        self.pancake = PancakeClient(
            rpc_url=config.get("pancake", {}).get("rpc_url", None),
            contract_address=config.get("pancake", {}).get("contract_address", "0x18B2A687610328590Bc8F2e5fEdDe3b582A49cdA"),
            use_mock_on_failure=config.get("pancake", {}).get("use_mock_on_failure", True),
            timeout=config.get("pancake", {}).get("timeout_seconds", 8),
        )
        self.strategy = Strategy(config)
        self.paper_trader = PaperTrader(config)

        self._running = False
        self._last_window_index = -1
        self._last_btc_window_open: float | None = None
        self._traded_this_window = False
        self._poll_interval = config.get("polymarket", {}).get("poll_interval_seconds", 5)

    def _on_price(self, pp: PricePoint):
        """Called on each new Binance trade price."""
        self.logger.debug(f"BTC price: {pp.price:.2f}")

    async def _run_main_loop(self):
        """Main async loop: poll Polymarket + evaluate strategy."""
        self.logger.info("Starting main strategy loop...")

        while self._running:
            try:
                await self._tick()
            except Exception as e:
                self.logger.error(f"Error in main loop tick: {e}", exc_info=True)

            await asyncio.sleep(self._poll_interval)

    async def _tick(self):
        """One iteration of the main loop."""
        now = time.time()
        window = get_current_window(now)

        # Detect new window
        if window.window_index != self._last_window_index:
            self._on_new_window(window)

        # Log status periodically
        if int(now) % 30 == 0:
            remaining = window.seconds_remaining
            btc = self.binance.last_price
            self.logger.info(
                f"Window {window.window_index} | "
                f"Remaining: {remaining:.0f}s | "
                f"BTC: {f'{btc:.2f}' if btc is not None else 'N/A'} | "
                f"Bankroll: ${self.paper_trader.metrics.bankroll:.2f}"
            )

        # Only evaluate in the entry window
        if window.seconds_remaining > self.strategy.entry_window_seconds:
            return

        # Get current price series for this window
        window_prices_raw = self.binance.get_window_prices(window.window_start_ts)
        prices = [pp.price for pp in window_prices_raw]

        if not prices:
            self.logger.debug("No prices in window yet.")
            return

        # Fetch current PancakeSwap BNB/USD round from BSC
        loop = asyncio.get_event_loop()
        round_data = await loop.run_in_executor(None, self.pancake.get_current_round)

        if round_data is None:
            self.logger.warning("Could not get PancakeSwap round — skipping this tick.")
            return

        yes_price_equiv = round_data.yes_price_equiv

        self.logger.info(
            f"Tick: {len(prices)} prices | "
            f"BNB: {prices[-1]:.2f} | "
            f"Pool: bull={round_data.bull_ratio:.1%} bear={round_data.bear_ratio:.1%} "
            f"({round_data.total_bnb:.3f} BNB) | "
            f"YES≡: {yes_price_equiv:.3f} | "
            f"Remaining: {window.seconds_remaining:.0f}s"
            + (" [MOCK]" if round_data.is_mock else "")
        )

        # Evaluate strategy (same Kelly logic, pool ratio as price signal)
        signal = self.strategy.evaluate(
            prices=prices,
            yes_price=yes_price_equiv,
            window=window,
            is_mock_data=round_data.is_mock,
        )

        if signal and not self._traded_this_window:
            # Enter paper trade (one trade per window max)
            self.paper_trader.enter_trade(signal, window)
            self._traded_this_window = True

    def _on_new_window(self, window):
        """Called when a new 5-minute window starts."""
        # Resolve previous window's trades
        if self._last_window_index > 0:
            prev_window_prices = self.binance.get_window_prices(
                window.window_start_ts - 300,
                window.window_start_ts,
            )
            if prev_window_prices:
                btc_open = prev_window_prices[0].price
                btc_close = prev_window_prices[-1].price
                self.logger.info(
                    f"🔚 Window {self._last_window_index} ended: "
                    f"BTC {btc_open:.2f} → {btc_close:.2f} "
                    f"({'UP' if btc_close > btc_open else 'DOWN'} "
                    f"{abs(btc_close/btc_open - 1):.3%})"
                )
                self.paper_trader.resolve_trades(
                    self._last_window_index, btc_open, btc_close
                )
            else:
                self.logger.warning(
                    f"No prices for window {self._last_window_index} — cannot resolve trades."
                )

        # Record new window open price
        self._last_window_index = window.window_index
        self._traded_this_window = False
        btc = self.binance.last_price
        self._last_btc_window_open = btc

        self.logger.info(
            f"🆕 Window {window.window_index} started | "
            f"BTC open: {f'{btc:.2f}' if btc is not None else 'N/A'}"
        )

        # Update strategy bankroll
        self.strategy.update_bankroll(self.paper_trader.metrics.bankroll)

    async def run(self):
        """Start the bot: Binance feed + main loop in parallel."""
        self._running = True

        # Set up graceful shutdown on Ctrl+C
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self.stop)

        self.logger.info("="*60)
        self.logger.info("  Polymarket BTC 5mn Paper Trading Bot")
        self.logger.info("="*60)

        # Test PancakeSwap / BSC connectivity
        self.logger.info("Testing PancakeSwap BNB/USD (BSC) connectivity...")
        loop = asyncio.get_event_loop()
        api_ok = await loop.run_in_executor(None, self.pancake.check_connectivity)
        if api_ok:
            self.logger.info("✅ PancakeSwap: BSC reachable — live on-chain data")
        else:
            self.logger.warning(
                "⚠️  PancakeSwap: BSC unreachable\n"
                "   → Running in MOCK DATA mode."
            )

        self.paper_trader.print_summary()

        # Run Binance feed + main loop concurrently
        self.binance.on_price = self._on_price
        await asyncio.gather(
            self.binance.run(),
            self._run_main_loop(),
        )

    def stop(self):
        """Graceful shutdown."""
        self.logger.info("Shutting down bot...")
        self._running = False
        self.binance.stop()
        self.paper_trader.print_summary()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Polymarket BTC 5mn Paper Trading Bot")
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
        help="Reset paper trading history and start from scratch",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.fresh:
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
        import os
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "w") as f:
            json.dump(empty, f, indent=2)
        logging.info(f"🧹 Fresh start: reset {log_file}")
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

    bot = PolymarketBot(config)

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
