"""
Risk management layer — circuit breakers, half-Kelly position sizing,
trade caps, drawdown limits.  Has ABSOLUTE VETO POWER over every signal.
Works independently of the HMM / strategy engine.

Position sizing pipeline:
  1. Circuit-breaker gate (block if active).
  2. Daily trade-cap gate (block if MAX_DAILY_TRADES reached).
  3. Max-concurrent-positions gate.
  4. Correlation gate (block or scale if correlated position exists).
  5. Half-Kelly fraction computed from rolling win-rate and payoff ratio.
  6. ATR-based risk budget: stop distance = ATR × gap multiplier.
  7. Apply daily-loss halving if applicable.
  8. Enforce KELLY_MAX_POSITION_PCT hard cap.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class PositionSizeResult:
    approved: bool
    shares: int
    notional: float
    reason: str
    kelly_fraction: float = 0.0


@dataclass
class RiskStatus:
    circuit_breaker_active: bool = False
    daily_pnl_pct: float = 0.0
    weekly_pnl_pct: float = 0.0
    peak_drawdown_pct: float = 0.0
    leverage_ok: bool = True
    lock_file_exists: bool = False
    messages: List[str] = field(default_factory=list)


class RiskManager:
    """Enforces all risk limits. Must be checked before every order."""

    def __init__(self, initial_equity: float):
        self.initial_equity   = initial_equity
        self.peak_equity      = initial_equity
        self._day_start_equity:  Dict[date, float] = {}
        self._week_start_equity: Dict[int, float]  = {}
        self._daily_trade_count: Dict[date, int]   = {}

    # ── Main gate ─────────────────────────────────────────────────────────────

    def check_all(self, current_equity: float, open_positions: dict) -> RiskStatus:
        """Return full risk status. Call before placing any order."""
        status = RiskStatus()

        if os.path.exists(settings.LOCK_FILE_PATH):
            status.circuit_breaker_active = True
            status.lock_file_exists = True
            status.messages.append(
                f"LOCK FILE present at {settings.LOCK_FILE_PATH}. "
                "Delete it after reviewing to restart."
            )
            return status

        today    = date.today()
        iso_week = today.isocalendar()[1]

        self._day_start_equity.setdefault(today, current_equity)
        self._week_start_equity.setdefault(iso_week, current_equity)

        day_start  = self._day_start_equity[today]
        week_start = self._week_start_equity[iso_week]

        status.daily_pnl_pct  = (current_equity - day_start)  / day_start
        status.weekly_pnl_pct = (current_equity - week_start) / week_start

        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        status.peak_drawdown_pct = (current_equity - self.peak_equity) / self.peak_equity

        # Peak drawdown — hard stop, write lock file
        if status.peak_drawdown_pct <= settings.CB_PEAK_DRAWDOWN_STOP:
            self._write_lock_file(status)
            status.circuit_breaker_active = True
            return status

        # Daily loss thresholds
        if status.daily_pnl_pct <= settings.CB_DAILY_LOSS_CLOSE:
            status.circuit_breaker_active = True
            status.messages.append(
                f"Daily loss {status.daily_pnl_pct:.1%} hit close-all threshold "
                f"({settings.CB_DAILY_LOSS_CLOSE:.1%})."
            )
        elif status.daily_pnl_pct <= settings.CB_DAILY_LOSS_HALVE:
            status.messages.append(
                f"Daily loss {status.daily_pnl_pct:.1%} — cutting sizes in half."
            )

        # Weekly loss thresholds
        if status.weekly_pnl_pct <= settings.CB_WEEKLY_LOSS_CLOSE:
            status.circuit_breaker_active = True
            status.messages.append(
                f"Weekly loss {status.weekly_pnl_pct:.1%} hit close-all threshold "
                f"({settings.CB_WEEKLY_LOSS_CLOSE:.1%}) — halting week."
            )
        elif status.weekly_pnl_pct <= settings.CB_WEEKLY_LOSS_RESIZE:
            status.messages.append(
                f"Weekly loss {status.weekly_pnl_pct:.1%} — resizing all positions."
            )

        return status

    # ── Position sizing ───────────────────────────────────────────────────────

    def size_position(
        self,
        equity: float,
        price: float,
        signal_equity_pct: float,
        risk_status: RiskStatus,
        existing_positions: dict,
        ticker: str,
        recent_returns: Optional[pd.Series] = None,
        atr: Optional[float] = None,
        buying_power: Optional[float] = None,
    ) -> PositionSizeResult:
        """Return approved share count respecting all risk limits."""

        if risk_status.circuit_breaker_active:
            return PositionSizeResult(False, 0, 0.0, "Circuit breaker active.")

        # Daily trade cap
        today       = date.today()
        trade_count = self._daily_trade_count.get(today, 0)
        if trade_count >= settings.MAX_DAILY_TRADES:
            return PositionSizeResult(
                False, 0, 0.0,
                f"Daily trade cap ({settings.MAX_DAILY_TRADES}) reached.",
            )

        # Max concurrent positions
        if len(existing_positions) >= settings.MAX_CONCURRENT_POSITIONS:
            return PositionSizeResult(
                False, 0, 0.0,
                f"Max concurrent positions ({settings.MAX_CONCURRENT_POSITIONS}) reached.",
            )

        # Correlation gate
        corr_action = self._correlation_action(ticker, existing_positions)
        if corr_action == "block":
            return PositionSizeResult(
                False, 0, 0.0,
                f"{ticker} is too highly correlated with an existing position.",
            )

        # Compute half-Kelly notional
        kelly_f, notional = self._kelly_notional(
            equity, price, signal_equity_pct, risk_status, atr, recent_returns
        )

        # Scale down for correlated position
        if corr_action == "scale":
            notional *= (1.0 - settings.MAX_CORRELATION_SCALE)

        # Leverage cap
        total_notional = sum(
            p.get("notional", 0) for p in existing_positions.values()
        )
        if (total_notional + notional) > equity * settings.MAX_LEVERAGE:
            notional = max(0.0, equity * settings.MAX_LEVERAGE - total_notional)

        # Buying power guard — cap at 95% of available buying power
        if buying_power is not None and buying_power > 0:
            if notional > buying_power:
                logger.warning(
                    "[%s] Notional $%.0f exceeds buying power $%.0f — capping.",
                    ticker, notional, buying_power,
                )
                notional = buying_power * 0.95
            if notional <= 0:
                return PositionSizeResult(False, 0, 0.0, "Insufficient buying power.")

        # Minimum order notional guard
        if notional < settings.MIN_ORDER_NOTIONAL:
            return PositionSizeResult(
                False, 0, 0.0,
                f"Notional ${notional:.2f} below minimum ${settings.MIN_ORDER_NOTIONAL:.0f}.",
            )

        shares = int(notional / price) if price > 0 else 0
        if shares <= 0:
            return PositionSizeResult(False, 0, 0.0, "Notional too small after adjustments.")

        return PositionSizeResult(True, shares, shares * price, "Approved.", kelly_f)

    def _kelly_notional(
        self,
        equity: float,
        price: float,
        signal_equity_pct: float,
        risk_status: RiskStatus,
        atr: Optional[float],
        recent_returns: Optional[pd.Series],
    ):
        """
        Half-Kelly sizing.  Formula: f* = 0.5 × (p·b − q) / b
          p = fraction of recent bars with positive returns
          b = mean_win / |mean_loss|
        Falls back to signal_equity_pct when history is insufficient.
        """
        kelly_f = signal_equity_pct  # default

        if recent_returns is None or len(recent_returns) < 10:
            logger.debug(
                "Kelly fallback to signal_equity_pct=%.2f — insufficient history (%d bars).",
                signal_equity_pct,
                len(recent_returns) if recent_returns is not None else 0,
            )
        else:
            wins   = recent_returns[recent_returns > 0]
            losses = recent_returns[recent_returns < 0]
            if len(wins) > 0 and len(losses) > 0:
                p = len(wins) / len(recent_returns)
                q = 1 - p
                b = wins.mean() / abs(losses.mean())
                if b > 0:
                    raw_kelly = (p * b - q) / b
                    kelly_f   = max(raw_kelly, 0.0) * settings.KELLY_FRACTION

        # ATR-based risk budget (gap protection)
        if atr is not None and atr > 0:
            stop_dist       = atr * settings.ATR_GAP_MULTIPLIER
            risk_budget     = equity * settings.MAX_POSITION_RISK_PCT
            notional_by_atr = (risk_budget / stop_dist) * price
        else:
            notional_by_atr = equity * settings.KELLY_MAX_POSITION_PCT

        kelly_notional = equity * kelly_f

        # Apply daily-loss halving
        if risk_status.daily_pnl_pct <= settings.CB_DAILY_LOSS_HALVE:
            kelly_notional *= 0.50

        # Weekly resize
        if risk_status.weekly_pnl_pct <= settings.CB_WEEKLY_LOSS_RESIZE:
            kelly_notional *= 0.50

        notional = min(
            kelly_notional,
            notional_by_atr,
            equity * settings.KELLY_MAX_POSITION_PCT,
        )
        return kelly_f, max(notional, 0.0)

    def record_trade(self, ticker: str) -> None:
        """Increment today's trade counter after a fill is confirmed."""
        today = date.today()
        self._daily_trade_count[today] = self._daily_trade_count.get(today, 0) + 1

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _correlation_action(self, ticker: str, existing_positions: dict) -> str:
        """
        Returns 'block' if correlation >= MAX_CORRELATION_BLOCK,
                'scale' if >= MAX_CORRELATION_SCALE,
                'allow' otherwise.
        Uses a hard-coded pairs table (extend with real matrix for production).
        """
        high_corr_pairs = {
            frozenset(["SPY", "QQQ"]),
            frozenset(["SPY", "IVV"]),
            frozenset(["QQQ", "TQQQ"]),
            frozenset(["TLT", "IEF"]),
            frozenset(["GLD", "IAU"]),
        }
        moderate_corr_pairs = {
            frozenset(["SPY", "TLT"]),
            frozenset(["QQQ", "SPY"]),
        }
        for existing in existing_positions:
            pair = frozenset([ticker, existing])
            if pair in high_corr_pairs:
                return "block"
            if pair in moderate_corr_pairs:
                return "scale"
        return "allow"

    def _write_lock_file(self, status: RiskStatus) -> None:
        os.makedirs(os.path.dirname(settings.LOCK_FILE_PATH), exist_ok=True)
        msg = (
            f"CIRCUIT BREAKER TRIGGERED — {datetime.utcnow().isoformat()}\n"
            f"Peak drawdown: {status.peak_drawdown_pct:.2%}\n"
            f"Daily P&L:     {status.daily_pnl_pct:.2%}\n"
            f"Weekly P&L:    {status.weekly_pnl_pct:.2%}\n\n"
            "Review strategy before deleting this file to resume.\n"
        )
        with open(settings.LOCK_FILE_PATH, "w") as f:
            f.write(msg)
        logger.critical("Lock file written: %s", settings.LOCK_FILE_PATH)
        status.lock_file_exists = True
        status.messages.append(
            f"Peak drawdown {status.peak_drawdown_pct:.1%} exceeded "
            f"{settings.CB_PEAK_DRAWDOWN_STOP:.1%} limit — BOT STOPPED."
        )

    def get_size_multiplier(self, risk_status: RiskStatus) -> float:
        """Convenience multiplier for callers that size externally."""
        if risk_status.circuit_breaker_active:
            return 0.0
        if risk_status.daily_pnl_pct <= settings.CB_DAILY_LOSS_HALVE:
            return 0.5
        if risk_status.weekly_pnl_pct <= settings.CB_WEEKLY_LOSS_RESIZE:
            return 0.75
        return 1.0
