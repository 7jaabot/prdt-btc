"""
round_logger.py — Log ALL completed rounds (regardless of whether we traded)

Maintains a JSON file at logs/rounds/all_rounds.json and a CSV at
logs/rounds/all_rounds.csv with crowd accuracy data for every settled epoch.

Thread-safe via fcntl file locking (multiple bot instances may write).
Deduplicates by epoch.
"""

from __future__ import annotations

import csv
import fcntl
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ROUNDS_DIR = "logs/rounds"
JSON_FILE = "logs/rounds/all_rounds.json"
CSV_FILE = "logs/rounds/all_rounds.csv"
LOCK_FILE = "logs/rounds/.lock"

POOL_SNAPSHOTS_JSON_FILE = "logs/rounds/pool_snapshots.json"
POOL_SNAPSHOTS_CSV_FILE = "logs/rounds/pool_snapshots.csv"
POOL_SNAPSHOTS_LOCK_FILE = "logs/rounds/.pool_snapshots_lock"
POOL_SNAPSHOTS_MAX_EPOCHS = 1000

POOL_SNAPSHOTS_CSV_COLUMNS = [
    "epoch",
    "seconds_to_lock",
    "total_bnb",
    "bull_bnb",
    "bear_bnb",
    "bull_pct",
    "bear_pct",
]

CSV_COLUMNS = [
    "epoch",
    "lock_ts",
    "final_bull_bnb",
    "final_bear_bnb",
    "final_total_bnb",
    "final_bull_pct",
    "final_bear_pct",
    "crowd_side",
    "crowd_conviction_pct",
    "bnb_open",
    "bnb_close",
    "actual_direction",
    "crowd_correct",
    "timestamp_logged",
]


class RoundLogger:
    """
    Logs every completed round's crowd data and actual outcome.

    Thread-safe: uses fcntl file lock for atomic read-modify-write.
    Deduplicates: won't log the same epoch twice.
    Skips rounds where oracle hasn't been called (bnb_open/bnb_close is None/0).
    """

    def __init__(self, rounds_dir: str = ROUNDS_DIR):
        self._dir = rounds_dir
        self._json_path = os.path.join(rounds_dir, "all_rounds.json")
        self._csv_path = os.path.join(rounds_dir, "all_rounds.csv")
        self._lock_path = os.path.join(rounds_dir, ".lock")
        self._ensure_dirs()

    def _ensure_dirs(self):
        """Create directory structure if it doesn't exist."""
        Path(self._dir).mkdir(parents=True, exist_ok=True)

    def _load_rounds(self) -> dict:
        """Load existing rounds from JSON. Returns {epoch_str: record}."""
        if not os.path.exists(self._json_path):
            return {}
        try:
            with open(self._json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"RoundLogger: could not load {self._json_path}: {e}")
            return {}

    def _save_rounds(self, rounds: dict):
        """Persist rounds dict to JSON and export CSV."""
        with open(self._json_path, "w", encoding="utf-8") as f:
            json.dump(rounds, f, indent=2)
        self._export_csv(rounds)

    def _export_csv(self, rounds: dict):
        """Re-export all rounds to CSV (sorted by epoch)."""
        try:
            sorted_rounds = sorted(rounds.values(), key=lambda r: r.get("epoch", 0))
            with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
                writer.writeheader()
                for record in sorted_rounds:
                    writer.writerow(record)
        except OSError as e:
            logger.warning(f"RoundLogger: CSV export failed: {e}")

    def log_round(
        self,
        epoch: int,
        lock_ts: float,
        final_bull_bnb: float,
        final_bear_bnb: float,
        bnb_open: Optional[float],
        bnb_close: Optional[float],
    ) -> bool:
        """
        Log a completed round's crowd data and outcome.

        Args:
            epoch: PancakeSwap epoch number
            lock_ts: Timestamp when round locked (float)
            final_bull_bnb: Final bull pool size in BNB
            final_bear_bnb: Final bear pool size in BNB
            bnb_open: Chainlink lock price (None if oracle not called yet)
            bnb_close: Chainlink close price (None if oracle not called yet)

        Returns:
            True if logged successfully, False if skipped (duplicate/missing data).
        """
        # Skip if oracle not called yet
        if not bnb_open or not bnb_close:
            logger.debug(f"RoundLogger: epoch {epoch} skipped — oracle not called yet")
            return False
        if bnb_open <= 0 or bnb_close <= 0:
            logger.debug(f"RoundLogger: epoch {epoch} skipped — invalid prices ({bnb_open}, {bnb_close})")
            return False

        epoch_key = str(epoch)

        # Compute derived fields
        final_total_bnb = final_bull_bnb + final_bear_bnb
        if final_total_bnb <= 0:
            logger.debug(f"RoundLogger: epoch {epoch} skipped — zero pool size")
            return False

        final_bull_pct = final_bull_bnb / final_total_bnb
        final_bear_pct = final_bear_bnb / final_total_bnb
        crowd_side = "UP" if final_bull_bnb > final_bear_bnb else "DOWN"
        crowd_conviction_pct = max(final_bull_pct, final_bear_pct)

        if bnb_close > bnb_open:
            actual_direction = "UP"
        elif bnb_close < bnb_open:
            actual_direction = "DOWN"
        else:
            actual_direction = "FLAT"

        crowd_correct = crowd_side == actual_direction

        record = {
            "epoch": epoch,
            "lock_ts": lock_ts,
            "final_bull_bnb": round(final_bull_bnb, 6),
            "final_bear_bnb": round(final_bear_bnb, 6),
            "final_total_bnb": round(final_total_bnb, 6),
            "final_bull_pct": round(final_bull_pct, 6),
            "final_bear_pct": round(final_bear_pct, 6),
            "crowd_side": crowd_side,
            "crowd_conviction_pct": round(crowd_conviction_pct, 6),
            "bnb_open": bnb_open,
            "bnb_close": bnb_close,
            "actual_direction": actual_direction,
            "crowd_correct": crowd_correct,
            "timestamp_logged": time.time(),
        }

        # Acquire file lock for thread-safe write
        lock_fd = open(self._lock_path, "w")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)

            rounds = self._load_rounds()

            # Deduplicate
            if epoch_key in rounds:
                logger.debug(f"RoundLogger: epoch {epoch} already logged — skipping")
                return False

            rounds[epoch_key] = record
            self._save_rounds(rounds)

            logger.info(
                f"📊 Round logged: epoch={epoch} | crowd={crowd_side} ({crowd_conviction_pct:.1%}) | "
                f"actual={actual_direction} | correct={crowd_correct} | "
                f"pool={final_total_bnb:.3f} BNB"
            )
            return True

        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()

    def get_all_rounds(self) -> list[dict]:
        """Return all logged rounds sorted by epoch."""
        rounds = self._load_rounds()
        return sorted(rounds.values(), key=lambda r: r.get("epoch", 0))

    def round_count(self) -> int:
        """Return number of logged rounds."""
        return len(self._load_rounds())

    # ──────────────────────────────────────────────────────────────────────
    # Pool snapshot methods
    # ──────────────────────────────────────────────────────────────────────

    def _pool_snapshots_paths(self):
        json_path = os.path.join(self._dir, "pool_snapshots.json")
        csv_path = os.path.join(self._dir, "pool_snapshots.csv")
        lock_path = os.path.join(self._dir, ".pool_snapshots_lock")
        return json_path, csv_path, lock_path

    def _load_pool_snapshots(self, json_path: str) -> dict:
        """Load existing pool snapshots from JSON. Returns {epoch_str: [snapshots]}."""
        if not os.path.exists(json_path):
            return {}
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"RoundLogger: could not load pool_snapshots.json: {e}")
            return {}

    def _save_pool_snapshots(self, snapshots: dict, json_path: str, csv_path: str):
        """Persist pool snapshots dict to JSON and append new rows to CSV."""
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(snapshots, f)
        self._export_pool_snapshots_csv(snapshots, csv_path)

    def _export_pool_snapshots_csv(self, snapshots: dict, csv_path: str):
        """Re-export all pool snapshots to CSV (sorted by epoch then seconds_to_lock desc)."""
        try:
            rows = []
            for epoch_key in sorted(snapshots.keys(), key=lambda k: int(k)):
                epoch = int(epoch_key)
                for snap in snapshots[epoch_key]:
                    total = snap.get("total_bnb", 0)
                    bull = snap.get("bull_bnb", 0)
                    bear = snap.get("bear_bnb", 0)
                    bull_pct = round(bull / total, 6) if total > 0 else 0.0
                    bear_pct = round(bear / total, 6) if total > 0 else 0.0
                    rows.append({
                        "epoch": epoch,
                        "seconds_to_lock": snap.get("seconds_to_lock", 0),
                        "total_bnb": round(total, 6),
                        "bull_bnb": round(bull, 6),
                        "bear_bnb": round(bear, 6),
                        "bull_pct": bull_pct,
                        "bear_pct": bear_pct,
                    })
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=POOL_SNAPSHOTS_CSV_COLUMNS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
        except OSError as e:
            logger.warning(f"RoundLogger: pool_snapshots CSV export failed: {e}")

    def log_pool_snapshots(self, epoch: int, snapshots: list[dict]) -> bool:
        """
        Log pool snapshots for a completed epoch.

        Args:
            epoch: PancakeSwap epoch number
            snapshots: list of {seconds_to_lock, total_bnb, bull_bnb, bear_bnb, ts}

        Returns:
            True if logged, False if skipped (no snapshots / duplicate).
        """
        if not snapshots:
            return False

        epoch_key = str(epoch)
        json_path, csv_path, lock_path = self._pool_snapshots_paths()

        # Strip 'ts' field (not needed in persistent store) and clean up
        clean_snapshots = [
            {
                "seconds_to_lock": s.get("seconds_to_lock", 0),
                "total_bnb": s.get("total_bnb", 0),
                "bull_bnb": s.get("bull_bnb", 0),
                "bear_bnb": s.get("bear_bnb", 0),
            }
            for s in snapshots
        ]

        lock_fd = open(lock_path, "w")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)

            all_snapshots = self._load_pool_snapshots(json_path)

            # Deduplicate
            if epoch_key in all_snapshots:
                logger.debug(f"RoundLogger: pool snapshots for epoch {epoch} already logged — skipping")
                return False

            all_snapshots[epoch_key] = clean_snapshots

            # Rotate: keep only the last POOL_SNAPSHOTS_MAX_EPOCHS epochs
            if len(all_snapshots) > POOL_SNAPSHOTS_MAX_EPOCHS:
                sorted_keys = sorted(all_snapshots.keys(), key=lambda k: int(k))
                to_remove = len(all_snapshots) - POOL_SNAPSHOTS_MAX_EPOCHS
                for old_key in sorted_keys[:to_remove]:
                    del all_snapshots[old_key]

            self._save_pool_snapshots(all_snapshots, json_path, csv_path)

            logger.info(
                f"📸 Pool snapshots logged: epoch={epoch} | {len(clean_snapshots)} snapshots"
            )
            return True

        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
