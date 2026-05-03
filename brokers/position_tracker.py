"""
Position tracker — mirrors live Alpaca positions and computes P&L.
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from brokers.alpaca_broker import AlpacaBroker
from config import settings

logger = logging.getLogger(__name__)

_TRADE_CSV_FIELDS = [
    "timestamp", "ticker", "side", "qty", "entry_price",
    "exit_price", "realized_pnl", "regime", "confidence", "allocation_pct",
]


@dataclass
class Position:
    ticker: str
    qty: int
    avg_entry: float
    current_price: float
    notional: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    side: str = "long"
    opened_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TradeRecord:
    ticker: str
    side: str
    qty: int
    entry_price: float
    exit_price: Optional[float]
    realized_pnl: float
    regime: str
    confidence: float
    allocation_pct: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


class PositionTracker:
    def __init__(self, broker: AlpacaBroker):
        self.broker = broker
        self.trade_history: List[TradeRecord] = []

    def get_positions(self) -> Dict[str, Position]:
        """Fetch current open positions from Alpaca."""
        try:
            raw = self.broker._trading_client.get_all_positions()
            positions = {}
            for p in raw:
                positions[p.symbol] = Position(
                    ticker=p.symbol,
                    qty=int(p.qty),
                    avg_entry=float(p.avg_entry_price),
                    current_price=float(p.current_price),
                    notional=float(p.market_value),
                    unrealized_pnl=float(p.unrealized_pl),
                    unrealized_pnl_pct=float(p.unrealized_plpc),
                    side=p.side,
                )
            return positions
        except Exception as exc:
            logger.error("get_positions failed: %s", exc)
            return {}

    def record_trade(self, record: TradeRecord) -> None:
        self.trade_history.append(record)
        logger.info(
            "Trade recorded: %s %s %d @ %.2f | regime=%s | pnl=%.2f",
            record.side.upper(), record.ticker, record.qty,
            record.entry_price, record.regime, record.realized_pnl,
        )
        self._append_csv(record)

    def _append_csv(self, record: TradeRecord) -> None:
        path = settings.TRADE_LOG_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_header = not os.path.exists(path)
        try:
            with open(path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_TRADE_CSV_FIELDS)
                if write_header:
                    writer.writeheader()
                writer.writerow({
                    "timestamp":    record.timestamp.isoformat(),
                    "ticker":       record.ticker,
                    "side":         record.side,
                    "qty":          record.qty,
                    "entry_price":  record.entry_price,
                    "exit_price":   record.exit_price,
                    "realized_pnl": record.realized_pnl,
                    "regime":       record.regime,
                    "confidence":   round(record.confidence, 4),
                    "allocation_pct": round(record.allocation_pct, 4),
                })
        except Exception as exc:
            logger.warning("Failed to write trade log: %s", exc)

    def get_trade_history_df(self):
        import pandas as pd
        if not self.trade_history:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "timestamp":    t.timestamp,
                "ticker":       t.ticker,
                "side":         t.side,
                "qty":          t.qty,
                "entry_price":  t.entry_price,
                "exit_price":   t.exit_price,
                "realized_pnl": t.realized_pnl,
                "regime":       t.regime,
                "confidence":   t.confidence,
                "allocation_pct": t.allocation_pct,
            }
            for t in self.trade_history
        ])
