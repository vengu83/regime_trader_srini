"""
SQLite state persistence for RegimeTrader.

WAL mode allows concurrent reads from the dashboard while the bot writes.
Four tables:
  snapshot       — latest equity, cash, detected regime, circuit-breaker flag
  equity_curve   — timestamped equity snapshots (for P&L chart)
  regime_history — per-bar regime labels, confidence, confirmation status
  trade_log      — every order with full attribution context

Usage:
    from core.state_persistence import StateDB, init_db
    init_db()
    db = StateDB()
    db.record_equity(equity)
    db.record_regime("SPY", "bull", 0.82, is_confirmed=True)
    db.record_trade("SPY", "buy", 10, 450.0, "bull", 0.82, kelly_fraction=0.12)
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime
from typing import List, Optional, Tuple

from config import settings

logger = logging.getLogger(__name__)


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(settings.SQLITE_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(settings.SQLITE_DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create all tables if they don't already exist."""
    conn = _connect()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS snapshot (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ts              TEXT    NOT NULL,
                equity          REAL,
                cash            REAL,
                regime          TEXT,
                circuit_breaker INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS equity_curve (
                id     INTEGER PRIMARY KEY AUTOINCREMENT,
                ts     TEXT NOT NULL,
                equity REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS regime_history (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                ts           TEXT NOT NULL,
                ticker       TEXT NOT NULL,
                regime       TEXT NOT NULL,
                confidence   REAL,
                is_confirmed INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS trade_log (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                ts             TEXT NOT NULL,
                ticker         TEXT NOT NULL,
                side           TEXT NOT NULL,
                qty            INTEGER,
                fill_price     REAL,
                regime         TEXT,
                confidence     REAL,
                realized_pnl   REAL    DEFAULT 0.0,
                kelly_fraction REAL    DEFAULT 0.0,
                order_type     TEXT    DEFAULT 'limit'
            );

            CREATE INDEX IF NOT EXISTS idx_equity_ts       ON equity_curve(ts);
            CREATE INDEX IF NOT EXISTS idx_regime_ticker   ON regime_history(ticker, ts);
            CREATE INDEX IF NOT EXISTS idx_trade_ticker    ON trade_log(ticker, ts);
        """)
        conn.commit()
        logger.info("SQLite DB ready: %s", settings.SQLITE_DB_PATH)
    finally:
        conn.close()


class StateDB:
    """
    Thread-safe SQLite wrapper. Create one instance per process and reuse it.
    Each write method commits immediately — no transaction batching needed
    for the single-writer bot loop.
    """

    def __init__(self):
        self._conn = _connect()

    # ── Writes ────────────────────────────────────────────────────────────────

    def upsert_snapshot(
        self,
        equity: float,
        cash: float,
        regime: str,
        circuit_breaker: bool,
    ) -> None:
        self._conn.execute(
            "INSERT INTO snapshot (ts, equity, cash, regime, circuit_breaker) "
            "VALUES (?,?,?,?,?)",
            (_now(), equity, cash, regime, int(circuit_breaker)),
        )
        self._conn.commit()

    def record_equity(self, equity: float) -> None:
        self._conn.execute(
            "INSERT INTO equity_curve (ts, equity) VALUES (?,?)",
            (_now(), equity),
        )
        self._conn.commit()

    def record_regime(
        self,
        ticker: str,
        regime: str,
        confidence: float,
        is_confirmed: bool = False,
    ) -> None:
        self._conn.execute(
            "INSERT INTO regime_history (ts, ticker, regime, confidence, is_confirmed) "
            "VALUES (?,?,?,?,?)",
            (_now(), ticker, regime, confidence, int(is_confirmed)),
        )
        self._conn.commit()

    def record_trade(
        self,
        ticker: str,
        side: str,
        qty: int,
        fill_price: float,
        regime: str,
        confidence: float,
        realized_pnl: float = 0.0,
        kelly_fraction: float = 0.0,
        order_type: str = "limit",
    ) -> None:
        self._conn.execute(
            "INSERT INTO trade_log "
            "(ts, ticker, side, qty, fill_price, regime, confidence, "
            " realized_pnl, kelly_fraction, order_type) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                _now(), ticker, side, qty, fill_price,
                regime, confidence, realized_pnl, kelly_fraction, order_type,
            ),
        )
        self._conn.commit()

    # ── Reads ─────────────────────────────────────────────────────────────────

    def get_equity_curve(self) -> List[Tuple]:
        cur = self._conn.execute(
            "SELECT ts, equity FROM equity_curve ORDER BY ts"
        )
        return cur.fetchall()

    def get_regime_history(self, ticker: str, limit: int = 200) -> List[Tuple]:
        cur = self._conn.execute(
            "SELECT ts, regime, confidence, is_confirmed "
            "FROM regime_history WHERE ticker=? ORDER BY ts DESC LIMIT ?",
            (ticker, limit),
        )
        return cur.fetchall()

    def get_recent_trades(self, n: int = 100) -> List[Tuple]:
        cur = self._conn.execute(
            "SELECT * FROM trade_log ORDER BY ts DESC LIMIT ?", (n,)
        )
        return cur.fetchall()

    def get_latest_snapshot(self) -> Optional[sqlite3.Row]:
        cur = self._conn.execute(
            "SELECT * FROM snapshot ORDER BY ts DESC LIMIT 1"
        )
        return cur.fetchone()

    def close(self) -> None:
        self._conn.close()


def _now() -> str:
    return datetime.utcnow().isoformat()
