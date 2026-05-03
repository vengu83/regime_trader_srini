"""
Volatility-based allocation strategies with per-regime technical confirmation.

Signal flow:
  1. StrategyOrchestrator.get_signal() checks HMM confidence.
  2. Regime-specific strategy computes target allocation.
  3. TechnicalFilter validates the signal before it is acted on:
       - Bull:    RSI(14) in [50, 75]  AND  MACD histogram positive & growing.
       - Neutral: price in lower 20% of Bollinger Band (mean-reversion entry).
       - Crash / Bear / Euphoria: no technical gate — act on regime alone.
  4. If filter blocks the signal, the orchestrator holds current allocation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class SignalData:
    regime: str
    confidence: float
    target_equity_pct: float      # 0–1 fraction of portfolio to deploy
    leverage: float               # multiplier applied to equity_pct
    strategy_name: str
    notes: str = ""

    @property
    def effective_exposure(self) -> float:
        return min(self.target_equity_pct * self.leverage, 2.0)  # cap at 2×


# ── Base strategy ──────────────────────────────────────────────────────────────

class BaseStrategy:
    name: str = "base"

    def signal(self, regime: str, confidence: float, features: pd.DataFrame) -> SignalData:
        raise NotImplementedError


# ── Regime-specific strategies ─────────────────────────────────────────────────

class CrashStrategy(BaseStrategy):
    name = "defensive_crash"

    def signal(self, regime, confidence, features):
        alloc = settings.REGIME_ALLOCATIONS["crash"]
        return SignalData(
            regime=regime, confidence=confidence,
            target_equity_pct=alloc["equity_pct"], leverage=alloc["leverage"],
            strategy_name=self.name,
            notes="Crash detected — minimal exposure, capital preservation.",
        )


class BearStrategy(BaseStrategy):
    name = "defensive_bear"

    def signal(self, regime, confidence, features):
        alloc = settings.REGIME_ALLOCATIONS["bear"]
        mom = float(features["mom_21"].iloc[-1]) if "mom_21" in features else 0.0
        pct = alloc["equity_pct"] * (0.7 if mom < -0.05 else 1.0)
        return SignalData(
            regime=regime, confidence=confidence,
            target_equity_pct=pct, leverage=alloc["leverage"],
            strategy_name=self.name,
            notes=f"Bear market, mom_21={mom:.3f}.",
        )


class NeutralStrategy(BaseStrategy):
    name = "balanced_neutral"

    def signal(self, regime, confidence, features):
        alloc = settings.REGIME_ALLOCATIONS["neutral"]
        return SignalData(
            regime=regime, confidence=confidence,
            target_equity_pct=alloc["equity_pct"], leverage=alloc["leverage"],
            strategy_name=self.name,
            notes="Neutral/choppy market — balanced allocation.",
        )


class BullStrategy(BaseStrategy):
    name = "aggressive_bull"

    def signal(self, regime, confidence, features):
        alloc = settings.REGIME_ALLOCATIONS["bull"]
        mom_5 = float(features["mom_5"].iloc[-1]) if "mom_5" in features else 0.0
        pct   = alloc["equity_pct"] if mom_5 >= 0 else alloc["equity_pct"] * 0.75
        return SignalData(
            regime=regime, confidence=confidence,
            target_equity_pct=pct, leverage=alloc["leverage"],
            strategy_name=self.name,
            notes=f"Bull market, mom_5={mom_5:.3f}.",
        )


class EuphoriaStrategy(BaseStrategy):
    name = "trim_euphoria"

    def signal(self, regime, confidence, features):
        alloc = settings.REGIME_ALLOCATIONS["euphoria"]
        return SignalData(
            regime=regime, confidence=confidence,
            target_equity_pct=alloc["equity_pct"], leverage=alloc["leverage"],
            strategy_name=self.name,
            notes="Euphoria — trimming positions, reducing leverage.",
        )


_STRATEGY_MAP = {
    "crash":       CrashStrategy(),
    "bear":        BearStrategy(),
    "neutral":     NeutralStrategy(),
    "bull":        BullStrategy(),
    "euphoria":    EuphoriaStrategy(),
    "deep_bear":   BearStrategy(),
    "strong_bull": BullStrategy(),
}


# ── Technical confirmation filter ─────────────────────────────────────────────

class TechnicalFilter:
    """
    Per-regime technical gate. Returns True (signal passes) or False (hold).

    Bull regimes require momentum confirmation; neutral requires a
    mean-reversion setup. Defensive regimes (crash/bear/euphoria) have
    no gate — the HMM regime signal is sufficient.

    Falls back to True (pass) whenever a required column is absent,
    ensuring backward-compatibility with older feature sets.
    """

    def passes(self, regime: str, features: pd.DataFrame) -> bool:
        if features.empty:
            return True

        if regime in ("bull", "strong_bull"):
            return self._bull_confirmation(features)
        if regime == "neutral":
            return self._neutral_confirmation(features)
        return True   # crash, bear, deep_bear, euphoria: no gate

    def _bull_confirmation(self, features: pd.DataFrame) -> bool:
        """RSI(14) in [50, 75] AND MACD histogram positive and growing."""
        missing = {"rsi_14", "macd_hist", "macd_hist_chg"} - set(features.columns)
        if missing:
            logger.warning("Bull filter BLOCKED: required columns missing %s", missing)
            return False   # block signal — don't trade on incomplete data

        rsi       = float(features["rsi_14"].iloc[-1])
        macd_hist = float(features["macd_hist"].iloc[-1])
        macd_chg  = float(features["macd_hist_chg"].iloc[-1])

        rsi_ok  = settings.TECH_RSI_BULL_LO <= rsi <= settings.TECH_RSI_BULL_HI
        macd_ok = macd_hist > 0 and macd_chg > 0

        if not (rsi_ok and macd_ok):
            logger.debug(
                "Bull filter BLOCKED: rsi=%.1f macd_hist=%.4f macd_chg=%.4f",
                rsi, macd_hist, macd_chg,
            )
        return rsi_ok and macd_ok

    def _neutral_confirmation(self, features: pd.DataFrame) -> bool:
        """Price at or below lower 20% of Bollinger Band — mean-reversion entry."""
        if "bb_lower_pct" not in features.columns:
            logger.warning("Neutral filter BLOCKED: bb_lower_pct column missing")
            return False   # block signal — don't trade on incomplete data

        bb_pct = float(features["bb_lower_pct"].iloc[-1])
        passes = bb_pct <= settings.TECH_BB_ENTRY_PCT
        if not passes:
            logger.debug("Neutral filter BLOCKED: bb_lower_pct=%.3f", bb_pct)
        return passes


# ── Orchestrator ───────────────────────────────────────────────────────────────

class StrategyOrchestrator:
    """Routes detected regime → strategy → technical filter → SignalData."""

    def __init__(self):
        self._tech_filter = TechnicalFilter()

    def get_signal(
        self,
        regime: str,
        confidence: float,
        features: pd.DataFrame,
    ) -> SignalData:
        if confidence < settings.MIN_CONFIDENCE:
            logger.info(
                "Confidence %.2f below threshold %.2f — holding.",
                confidence, settings.MIN_CONFIDENCE,
            )
            return SignalData(
                regime=regime, confidence=confidence,
                target_equity_pct=0.50, leverage=1.0,
                strategy_name="hold_low_confidence",
                notes="Confidence too low to act.",
            )

        strategy = _STRATEGY_MAP.get(regime, NeutralStrategy())
        signal   = strategy.signal(regime, confidence, features)

        if not self._tech_filter.passes(regime, features):
            logger.info("Technical filter blocked signal for regime=%s", regime)
            return SignalData(
                regime=regime, confidence=confidence,
                target_equity_pct=0.50, leverage=1.0,
                strategy_name="hold_tech_filter",
                notes="Technical confirmation failed — holding.",
            )

        logger.info(
            "Strategy signal: regime=%s exposure=%.2f (%.0f%% × %.2f×)",
            regime, signal.effective_exposure,
            signal.target_equity_pct * 100, signal.leverage,
        )
        return signal

    def rebalance_needed(
        self,
        current_pct: float,
        target_pct: float,
        threshold: float = 0.05,
    ) -> bool:
        """Suppress micro-rebalances within the deadband."""
        return abs(current_pct - target_pct) >= threshold
