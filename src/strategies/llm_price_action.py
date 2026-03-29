"""
llm_price_action.py — LLM Price Action Strategy

Hypothesis: A language model analysing multi-timeframe candlestick patterns
can detect confluences that single-indicator strategies miss.

Data: Binance spot klines (BNBUSDT) on 1m, 5m, 15m, 1h timeframes.
LLM: Groq API (free) with Llama 3.1 70B — fast (~500ms), no cost.

The LLM receives raw OHLCV data and returns:
  - direction: UP / DOWN / SKIP
  - confidence: 0.0 to 1.0
  - reasoning: brief explanation (logged)

Edge is derived from confidence: edge = (confidence - 0.5) * 2
Only trades when confidence >= 0.60 (configurable).
"""

import json
import logging
import os
import time
import urllib.request
from typing import Optional

from .base import BaseStrategy
from strategy import Signal, WindowInfo, compute_position_size

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

# Timeframes and how many candles to fetch for each
TIMEFRAMES = [
    ("1m", 10),   # last 10 minutes
    ("5m", 12),   # last 1 hour
    ("15m", 8),   # last 2 hours
    ("1h", 6),    # last 6 hours
]

SYSTEM_PROMPT = """You are a crypto price action analyst. You will receive BNB/USDT candlestick data across multiple timeframes.

Your task: predict whether BNB price will be HIGHER or LOWER than the current price in exactly 5 minutes.

Analyse:
- Trend direction and strength on each timeframe
- Support/resistance levels from recent candles
- Candlestick patterns (engulfing, doji, hammer, etc.)
- Volume anomalies
- Confluence across timeframes (e.g. 1m pullback in a 15m uptrend)

Respond in EXACTLY this JSON format, nothing else:
{"direction": "UP" or "DOWN" or "SKIP", "confidence": 0.50 to 1.00, "reasoning": "brief explanation"}

Rules:
- confidence must reflect TRUE probability — 0.50 = coin flip, 0.60 = slight edge, 0.75+ = strong signal
- Use SKIP when patterns are unclear or conflicting
- SKIP is better than a low-confidence trade
- Be conservative — only UP/DOWN when you see genuine confluence"""


class LLMPriceActionStrategy(BaseStrategy):
    """
    🧠 LLM Price Action — Multi-timeframe analysis via Groq/Llama 3.1

    Sends OHLCV candles on 1m/5m/15m/1h to an LLM which predicts
    the 5-minute direction based on price action patterns.
    """

    @property
    def name(self) -> str:
        return "🧠 LLM Price Action"

    def __init__(self, config: dict):
        super().__init__(config)
        cfg = config.get("strategy", {})

        # Groq API key — from config, env var, or .env file
        self.groq_api_key: str = (
            cfg.get("groq_api_key")
            or os.environ.get("GROQ_API_KEY")
            or self._load_env_key()
            or ""
        )

        # HTTP timeout for API calls
        self._http_timeout: float = cfg.get("llm_http_timeout", 10.0)

        # Cache klines to avoid refetching within the same entry window
        self._klines_cache: Optional[dict] = None
        self._klines_cache_ts: float = 0.0
        self._klines_cache_ttl: float = 15.0  # seconds

    def _load_env_key(self) -> Optional[str]:
        """Try to load GROQ_API_KEY from .env file in project root."""
        env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
        if os.path.exists(env_path):
            try:
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("GROQ_API_KEY="):
                            return line.split("=", 1)[1].strip().strip('"').strip("'")
            except Exception:
                pass
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Prefetch (sniper pre-load phase)
    # ─────────────────────────────────────────────────────────────────────────

    def prefetch(self, prices: list[float], epoch=None) -> None:
        """Pre-fetch klines and call the LLM ahead of the sniper window."""
        super().prefetch(prices, epoch)
        if not self.groq_api_key:
            logger.warning("[LLM] prefetch: no GROQ_API_KEY — skipping")
            return

        current_price = prices[-1] if prices else 0.0
        if current_price <= 0:
            return

        logger.debug("[LLM] prefetch: fetching klines + calling LLM...")
        klines = self._fetch_all_klines()
        if klines is None:
            logger.warning("[LLM] prefetch: failed to fetch klines")
            return

        prompt = self._format_klines_prompt(klines, current_price)
        result = self._call_llm(prompt)
        if result is not None:
            self._prefetch_cache["llm_result"] = result
            logger.info(
                f"[LLM] prefetch: {result['direction']} conf={result['confidence']:.2f} "
                f"({result['latency']:.1f}s) — {result.get('reasoning', '')}"
            )
        else:
            logger.warning("[LLM] prefetch: LLM call failed")

    # ─────────────────────────────────────────────────────────────────────────
    # Data fetching
    # ─────────────────────────────────────────────────────────────────────────

    def _fetch_klines(self, interval: str, limit: int) -> list[list]:
        """Fetch OHLCV klines from Binance REST API."""
        try:
            url = f"{BINANCE_KLINES_URL}?symbol=BNBUSDT&interval={interval}&limit={limit}"
            req = urllib.request.Request(url, headers={"User-Agent": "bnb-updown-bot/1.0"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.warning(f"[LLM] Failed to fetch {interval} klines: {e}")
            return []

    def _fetch_all_klines(self) -> Optional[dict]:
        """Fetch klines for all timeframes, with caching."""
        now = time.time()
        if self._klines_cache and (now - self._klines_cache_ts) < self._klines_cache_ttl:
            return self._klines_cache

        result = {}
        for interval, limit in TIMEFRAMES:
            raw = self._fetch_klines(interval, limit)
            if not raw:
                return None
            # Format: [open_time, open, high, low, close, volume, ...]
            candles = []
            for k in raw:
                candles.append({
                    "o": round(float(k[1]), 2),
                    "h": round(float(k[2]), 2),
                    "l": round(float(k[3]), 2),
                    "c": round(float(k[4]), 2),
                    "v": round(float(k[5]), 1),
                })
            result[interval] = candles

        self._klines_cache = result
        self._klines_cache_ts = now
        return result

    def _format_klines_prompt(self, klines: dict, current_price: float) -> str:
        """Format klines data into a concise prompt for the LLM."""
        lines = [f"Current BNB/USDT price: ${current_price:.2f}\n"]
        for interval, candles in klines.items():
            lines.append(f"=== {interval} candles (oldest→newest) ===")
            for c in candles:
                body = "▲" if c["c"] > c["o"] else "▼" if c["c"] < c["o"] else "─"
                lines.append(
                    f"  {body} O={c['o']:.2f} H={c['h']:.2f} L={c['l']:.2f} "
                    f"C={c['c']:.2f} V={c['v']:.0f}"
                )
            lines.append("")
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # LLM call
    # ─────────────────────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> Optional[dict]:
        """Call Groq API and parse the JSON response."""
        if not self.groq_api_key:
            logger.error("[LLM] No GROQ_API_KEY configured")
            return None

        import requests as req_lib

        try:
            start = time.time()
            resp = req_lib.post(
                GROQ_API_URL,
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 200,
                },
                headers={
                    "Authorization": f"Bearer {self.groq_api_key}",
                },
                timeout=self._http_timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            latency = time.time() - start

            content = data["choices"][0]["message"]["content"].strip()
            logger.info(f"[LLM] Response ({latency:.1f}s): {content}")

            # Parse JSON from response (handle markdown code blocks)
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json.loads(content)

            # Validate
            direction = result.get("direction", "SKIP").upper()
            confidence = float(result.get("confidence", 0.5))
            reasoning = result.get("reasoning", "")

            if direction not in ("UP", "DOWN", "SKIP"):
                direction = "SKIP"
            confidence = max(0.50, min(1.0, confidence))

            return {
                "direction": direction,
                "confidence": confidence,
                "reasoning": reasoning,
                "latency": latency,
            }

        except req_lib.exceptions.HTTPError as e:
            body = e.response.text[:200] if e.response else ""
            logger.error(f"[LLM] Groq API HTTP {e.response.status_code if e.response else '?'}: {body}")
        except json.JSONDecodeError as e:
            logger.error(f"[LLM] Failed to parse LLM response as JSON: {e}")
        except Exception as e:
            logger.error(f"[LLM] Groq API error: {e}")

        return None

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

        # Only evaluate within entry window
        if not window.is_entry_window and window.seconds_remaining > self.entry_window_seconds:
            return None

        # Check API key
        if not self.groq_api_key:
            self.last_skip_reason = "⏸ [LLM] No GROQ_API_KEY configured"
            return None

        # Get current price
        current_price = prices[-1] if prices else 0.0
        if current_price <= 0:
            self.last_skip_reason = "⏸ [LLM] No price data available"
            return None

        # Use prefetched LLM result if available, otherwise fetch + call LLM
        if "llm_result" in self._prefetch_cache:
            result = self._prefetch_cache["llm_result"]
            logger.debug(f"[LLM] using prefetched result: {result['direction']} conf={result['confidence']:.2f}")
        else:
            klines = self._fetch_all_klines()
            if klines is None:
                self.last_skip_reason = "⏸ [LLM] Failed to fetch klines"
                return None
            prompt = self._format_klines_prompt(klines, current_price)
            result = self._call_llm(prompt)
            if result is None:
                self.last_skip_reason = "⏸ [LLM] API call failed"
                return None

        direction = result["direction"]
        confidence = result["confidence"]
        reasoning = result["reasoning"]
        latency = result["latency"]

        logger.info(
            f"🧠 LLM: {direction} (conf={confidence:.2f}, {latency:.1f}s) — {reasoning}"
        )

        # Skip if LLM says skip or confidence too low
        if direction == "SKIP":
            self.last_skip_reason = f"⏸ [LLM] SKIP — {reasoning}"
            return None

        # Convert to Signal
        side = "YES" if direction == "UP" else "NO"
        p_up = confidence if direction == "UP" else (1.0 - confidence)

        # Edge from confidence: 0.55 → 0.10, 0.60 → 0.20, 0.75 → 0.50
        edge = abs(confidence - 0.5) * 2.0

        if edge <= self.edge_threshold:
            self.last_skip_reason = (
                f"⏸ [LLM] Edge too low ({edge:.3f} ≤ {self.edge_threshold:.2f})"
            )
            return None

        # Fair odds pricing
        effective_yes_price = 0.50

        # Position sizing via Kelly
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
            self.last_skip_reason = "⏸ [LLM] Kelly sizing → 0 position"
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

        logger.info(
            f"🧠 LLM Signal: {signal} | {direction} conf={confidence:.2f} "
            f"| {reasoning}"
        )
        return signal
