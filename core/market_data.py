"""
Market data provider — historical (yfinance) and real-time (Alpaca).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class MarketDataProvider:
    """Fetch OHLCV data for training and live bar updates."""

    # ── Historical (yfinance) ─────────────────────────────────────────────────

    def get_history(
        self,
        ticker: str,
        days: int = 504,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Return a DataFrame with columns: open, high, low, close, volume."""
        end = datetime.utcnow()
        # days is in trading days; multiply by ~1.5 to convert to calendar days
        # (252 trading days ≈ 365 calendar days) plus a holiday buffer.
        start = end - timedelta(days=int(days * 1.5) + 10)
        try:
            raw = yf.download(
                ticker, start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval=interval, progress=False, auto_adjust=True,
            )
            if raw.empty:
                raise ValueError(f"No data returned for {ticker}")
            # yfinance ≥0.2.x returns MultiIndex columns (field, ticker) for
            # single-ticker downloads — flatten to just the field level.
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
            df.index = pd.to_datetime(df.index)
            df = df.dropna().tail(days)
            logger.info("Fetched %d bars for %s", len(df), ticker)
            return df
        except Exception as exc:
            logger.error("Failed to fetch history for %s: %s", ticker, exc)
            raise

    # ── Live bar (Alpaca) ─────────────────────────────────────────────────────

    def get_latest_bar(self, ticker: str, alpaca_client) -> Optional[pd.Series]:
        """
        Fetch the most recent completed bar from Alpaca.
        alpaca_client should be an AlpacaBroker instance.
        """
        try:
            bar = alpaca_client.get_latest_bar(ticker)
            return bar
        except Exception as exc:
            logger.error("Failed to get latest bar for %s: %s", ticker, exc)
            return None

    # ── Rolling live window ───────────────────────────────────────────────────

    def build_live_window(
        self,
        ticker: str,
        lookback_days: int,
        alpaca_client=None,
        max_bar_age_hours: float = 2.0,
    ) -> pd.DataFrame:
        """
        Return a recent OHLCV window suitable for computing HMM features.
        Uses yfinance as fallback when Alpaca isn't available or data is stale.
        """
        if alpaca_client is not None:
            df = alpaca_client.get_bars(ticker, days=lookback_days)
            if df is not None and not df.empty:
                latest_ts = df.index[-1]
                if latest_ts.tzinfo is not None:
                    latest_ts = latest_ts.tz_convert("UTC").tz_localize(None)
                age_hours = (datetime.utcnow() - latest_ts).total_seconds() / 3600
                if age_hours > max_bar_age_hours:
                    logger.warning(
                        "Stale Alpaca data for %s: latest bar is %.1f hours old — "
                        "falling back to yfinance.",
                        ticker, age_hours,
                    )
                else:
                    logger.info("Live window for %s: %d bars from Alpaca", ticker, len(df))
                    return df
            else:
                logger.warning("Alpaca get_bars failed for %s — falling back to yfinance", ticker)
        return self.get_history(ticker, days=lookback_days)
