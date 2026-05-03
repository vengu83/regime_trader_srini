"""
Alpaca broker wrapper — authentication, account info, and market data.
Credentials are loaded from .env — NEVER hardcoded or passed to Claude.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class AlpacaBroker:
    """
    Thin wrapper around alpaca-py that exposes the methods
    needed by the trading system.
    """

    def __init__(self):
        self.api_key    = os.getenv("ALPACA_API_KEY", "")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        self.base_url   = os.getenv(
            "ALPACA_BASE_URL",
            "https://paper-api.alpaca.markets",
        )
        self.is_paper   = os.getenv("ALPACA_PAPER", "true").lower() == "true"
        self._trading_client = None
        self._data_client    = None

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Initialise Alpaca clients. Returns True on success."""
        if not self.api_key or not self.secret_key:
            logger.error(
                "Alpaca credentials missing. Populate ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY in your .env file."
            )
            return False
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient

            self._trading_client = TradingClient(
                self.api_key, self.secret_key, paper=self.is_paper
            )
            self._data_client = StockHistoricalDataClient(
                self.api_key, self.secret_key
            )
            account = self._trading_client.get_account()
            logger.info(
                "Alpaca connected — account %s | equity=$%.2f | "
                "buying_power=$%.2f | paper=%s",
                account.id,
                float(account.equity),
                float(account.buying_power),
                self.is_paper,
            )
            return True
        except Exception as exc:
            logger.error("Alpaca connection failed: %s", exc)
            return False

    # ── Account ───────────────────────────────────────────────────────────────

    def get_account(self) -> Optional[dict]:
        try:
            acc = self._trading_client.get_account()
            return {
                "equity":        float(acc.equity),
                "buying_power":  float(acc.buying_power),
                "cash":          float(acc.cash),
                "portfolio_value": float(acc.portfolio_value),
                "status":        acc.status,
            }
        except Exception as exc:
            logger.error("get_account failed: %s", exc)
            return None

    def is_market_open(self) -> bool:
        try:
            clock = self._trading_client.get_clock()
            return clock.is_open
        except Exception:
            return False

    # ── Market data ───────────────────────────────────────────────────────────

    def get_latest_bar(self, ticker: str) -> Optional[pd.Series]:
        try:
            from alpaca.data.requests import StockLatestBarRequest
            req = StockLatestBarRequest(symbol_or_symbols=ticker)
            bars = self._data_client.get_stock_latest_bar(req)
            bar  = bars[ticker]
            return pd.Series({
                "open":   bar.open,
                "high":   bar.high,
                "low":    bar.low,
                "close":  bar.close,
                "volume": bar.volume,
            })
        except Exception as exc:
            logger.error("get_latest_bar(%s) failed: %s", ticker, exc)
            return None

    def get_current_price(self, ticker: str) -> Optional[float]:
        bar = self.get_latest_bar(ticker)
        return float(bar["close"]) if bar is not None else None

    def get_bars(self, ticker: str, days: int) -> Optional[pd.DataFrame]:
        """Fetch daily OHLCV bars from Alpaca for the past `days` calendar days."""
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame

            end   = datetime.utcnow()
            start = end - timedelta(days=days + 7)  # buffer for weekends/holidays
            req   = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
            )
            bars = self._data_client.get_stock_bars(req)
            df   = bars.df
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(ticker, level="symbol")
            df.index = pd.to_datetime(df.index)
            df = df[["open", "high", "low", "close", "volume"]].dropna()
            return df.tail(days)
        except Exception as exc:
            logger.error("get_bars(%s, %d) failed: %s", ticker, days, exc)
            return None
